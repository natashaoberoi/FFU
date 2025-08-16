# === Fully DDP-safe 4-GPU training script (updated with latest changes) ===
import os, sys, json, math, time, random, uuid, csv
from datetime import datetime
from pathlib import Path
from itertools import chain

import regex
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from torch import amp
from PIL import Image
from tqdm.auto import tqdm
from collections import defaultdict as _dd

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

# ---------------------------------------------------------------------
# DDP env defaults for mp.spawn + init_method='env://'
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")
torch.backends.cudnn.benchmark = True
# ---------------------------------------------------------------------

# Repro
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Local site-packages path(s)
sys.path.insert(0, "/scratch/zt1/project/msml612/user/indro/my_site_packages")

# Project imports
from FFU import UTransformer, coco_name   # (we define GRefDataset locally)

# CLIP (local installation)
sys.path.insert(0, '/scratch/zt1/project/msml612/user/indro/open_clip_package')
import open_clip

# ---------------------------------------------------------------------
# Device (for single-GPU fallback and parent-process eval)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# OpenCLIP model/tokenizer (tokenizer used in probes; model kept for consistency)
clip_model, _, _ = open_clip.create_model_and_transforms(
    'ViT-B-32',
    pretrained='/scratch/zt1/project/msml612/user/indro/open_clip/models--laion--CLIP-ViT-B-32-laion2B-s34B-b79K/open_clip_pytorch_model.bin',
    cache_dir='/scratch/zt1/project/msml612/user/indro/open_clip'
)
clip_model = clip_model.eval().to(device)
clip_tok = open_clip.get_tokenizer(
    'ViT-B-32',  # match the model arch (no slash)
    cache_dir='/scratch/zt1/project/msml612/user/indro/open_clip'
)

# Paths
CKPT_DIR   = Path("/scratch/zt1/project/msml612/user/indro/checkpoints")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_FILE  = CKPT_DIR / "latest.pt"

ROOT       = '/scratch/zt1/project/msml612/user/indro/datasets'
GREF_JSON  = f'{ROOT}/grefs(unc).json'
INST_JSON  = f'{ROOT}/instances.json'
IMG_DIR    = f'{ROOT}/coco/train2014/gref_images'

# =====================================================================
# Tuned / latest hyper-parameters & knobs
# =====================================================================
IMG_SIZE   = 512
MAX_TOK    = 20

# Sampling / splits
ROWS_TOTAL   = 100
TRAIN_RATIO  = 0.9

# Note: TRAIN_RATIO is ignored when USE_FULL_DATASET is enabled; we use the
# dataset-defined train/test splits in full.

# Flag: use full dataset
USE_FULL_DATASET = True

# Batch & accumulation
PER_RANK_BATCH = 16          # per-GPU mini-batch
ACC_STEPS      = 2
# Run the caption-shuffle probe (full test set, non-sharded) every N epochs
SHUFFLE_PROBE_EVERY = 5

# LRs (tuned)
LR_IMG   = 8.24757533344303e-05
LR_TEXT  = 5.616604788742511e-04
LR_GAIN  = 3.32990803758666e-04
WEIGHT_DECAY = 3.4695916603302916e-04

# Regularizers / aux weights (tuned)
REG_W_FiLM = 3e-4
REG_W_AREA = 0.0610283292575495
REG_W_COM  = 0.042710778862625635
TV_W       = 0.08180147659224932

# λ-div warm-up epochs
DIV_WARMUP_EPOCHS = 8

# Model width/heads
BASE_CHANNELS = 80
NUM_HEADS     = 4

# Probe / diagnostics
TEXT_PROBE_FREEZE_EPOCH = 6
TEXT_PROBE_THAW_EPOCH   = 9
PROBE_BATCH             = 8
SHUFFLE_BATCHES         = 2    # (not computed in DDP loop)

# Grad clip policy around unfreeze
FREEZE_EPOCHS            = 1
CLIP_NORM_BASE           = 1.0
CLIP_NORM_AFTER_UNFREEZE = 0.5
SIGMA_TOL_FOR_TXT_LR     = 20.0

# Scheduler / early stop
EPOCHS        = 100
PATIENCE_MAX  = 15
min_delta     = 1e-4

WARMUP_EPOCHS = 2

# λ-div schedule
def lambda_div(ep: int, start: float = 0.0, full: float = 1.0, warmup: int = DIV_WARMUP_EPOCHS):
    return min(1.0, ep / warmup) * full

# Logging CSV
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_id = uuid.uuid4().hex[:8]
BASE_PATH = "/scratch/zt1/project/msml612/user/indro/"
LOG_DIR = os.path.join(BASE_PATH, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_CSV_PATH = os.path.join(LOG_DIR, f"training_log_{timestamp}_{run_id}.csv")

def init_log_csv():
    if not os.path.exists(LOG_CSV_PATH):
        with open(LOG_CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch","log_gain","xattn_H",
                "enc_γμ","enc_γσ","enc_βμ","enc_βσ",
                "dec_γμ","dec_γσ","dec_βμ","dec_βσ",
                "train_loss","combo_train","val_loss","lambda_div",
                "train_time","val_time","peak_gpu_mem_mb"
            ])

def log_metrics(epoch, log_gain, xattn_H,
                enc_γμ, enc_γσ, enc_βμ, enc_βσ,
                dec_γμ, dec_γσ, dec_βμ, dec_βσ,
                train_loss, combo_train, val_loss, lambda_div,
                train_time, val_time, peak_gpu_mem_mb):
    with open(LOG_CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch, log_gain, xattn_H,
            enc_γμ, enc_γσ, enc_βμ, enc_βσ,
            dec_γμ, dec_γσ, dec_βμ, dec_βσ,
            train_loss, combo_train, val_loss, lambda_div,
            train_time, val_time, peak_gpu_mem_mb
        ])

# ──────────────────────────────
# Per-epoch 2×2 visualization
# ──────────────────────────────
EPOCH_VIZ_DIR = Path("artifacts/epoch_viz")
EPOCH_VIZ_DIR.mkdir(parents=True, exist_ok=True)
# fixed samples so the *same* pair is visualized over time
VIS_TRAIN_IDX = 0
VIS_TEST_IDX  = 0

def _get_sample(dataset, idx, device):
    """dataset -> (img[1,C,H,W], ids[1,77], gt[1,1,H,W], caption:str, img_id:int)."""
    img, clip_ids, gt, cap, img_id, _sid = dataset[idx]
    return (img.unsqueeze(0).to(device),
            clip_ids.unsqueeze(0).to(device),
            gt.unsqueeze(0).to(device),
            cap, int(img_id))

def save_epoch_pair_figure(epoch: int, model_any, device, train_dataset, test_dataset):
    """Save 2×2: Train(GT/Pred), Test(GT/Pred). Print per-sample timings to stdout."""
    model = model_any.module if hasattr(model_any, "module") else model_any
    model.eval()
    with torch.no_grad():
        # Train sample (timed)
        t_img, t_ids, t_gt, t_cap, t_iid = _get_sample(train_dataset, VIS_TRAIN_IDX, device)
        if device.type == "cuda": torch.cuda.synchronize(device)
        _t0 = time.perf_counter()
        t_out = model(t_img, t_ids)     # model returns prob in shape [1,1,H,W]
        if device.type == "cuda": torch.cuda.synchronize(device)
        t_ms = (time.perf_counter() - _t0) * 1000.0
        # Ensure 2D arrays: [H,W]
        t_prob = t_out.squeeze().detach().cpu().numpy()         # [H,W]
        t_gt_np = t_gt.squeeze().detach().cpu().numpy()         # [H,W]
        t_base = (t_img.squeeze(0).detach().cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
        t_pred = (t_prob > 0.5).astype(np.uint8)

        # Test sample (timed)
        v_img, v_ids, v_gt, v_cap, v_iid = _get_sample(test_dataset, VIS_TEST_IDX, device)
        if device.type == "cuda": torch.cuda.synchronize(device)
        _s0 = time.perf_counter()
        v_out = model(v_img, v_ids)
        if device.type == "cuda": torch.cuda.synchronize(device)
        v_ms = (time.perf_counter() - _s0) * 1000.0
        v_prob = v_out.squeeze().detach().cpu().numpy()         # [H,W]
        v_gt_np = v_gt.squeeze().detach().cpu().numpy()         # [H,W]
        v_base = (v_img.squeeze(0).detach().cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
        v_pred = (v_prob > 0.5).astype(np.uint8)

    # Print timings to stdout
    print(f"[epoch_viz] Epoch {epoch}: train img {t_iid} = {t_ms:.2f} ms | test img {v_iid} = {v_ms:.2f} ms")

    # 2×2 layout (reuse existing overlay_mask)
    plt.figure(figsize=(10, 8))
    # Train row
    plt.subplot(2,2,1); plt.imshow(overlay_mask(t_gt_np, t_base)); plt.axis('off'); plt.title(f"Train GT (img {t_iid})")
    plt.subplot(2,2,2); plt.imshow(overlay_mask(t_pred,  t_base)); plt.axis('off'); plt.title("Train Pred")
    # Test row
    plt.subplot(2,2,3); plt.imshow(overlay_mask(v_gt_np, v_base)); plt.axis('off'); plt.title(f"Test GT (img {v_iid})")
    plt.subplot(2,2,4); plt.imshow(overlay_mask(v_pred,  v_base)); plt.axis('off'); plt.title("Test Pred")
    plt.suptitle(f"Train: “{t_cap}”\nTest:  “{v_cap}”", y=0.98)
    plt.tight_layout()
    out_path = EPOCH_VIZ_DIR / f"epoch_{epoch:03d}_train{t_iid}_test{v_iid}.png"
    plt.savefig(out_path.as_posix(), dpi=150); plt.close()
    print(f"[epoch_viz] saved {out_path.as_posix()}")
    model.train()

# =====================================================================
# Dataset prep (and local Dataset class with in-scope ann_ids_to_mask)
# =====================================================================
from pycocotools.coco import COCO
coco = COCO(INST_JSON)

# CLIP vocabulary size (dynamic from model's token embedding)
VOCAB  = int(clip_model.token_embedding.weight.shape[0])

def encode_text_clip(txt, ctx_len: int = 77) -> torch.LongTensor:
    """txt: str or list[str] → (B,77) long tensor on CPU."""
    if isinstance(txt, str):
        toks = clip_tok([txt], context_length=ctx_len)
        return toks[0]
    else:
        return clip_tok(txt, context_length=ctx_len)
    
# --- CLIP-only vocab gating helpers -----------------------------------------
# Strip CLIP special tokens: pad=0, sos=49406, eos=49407
CLIP_SPECIAL_TOKENS = {0, 49406, 49407}
def clip_vocab_ids_from_text(text: str):
    ids = encode_text_clip(text)  # shape: [77]
    return [int(t) for t in ids.tolist() if int(t) not in CLIP_SPECIAL_TOKENS]
def caption_in_clip_vocab(entry, vocab_ids: set[int]) -> bool:
    return all(set(clip_vocab_ids_from_text(s["sent"])).issubset(vocab_ids)
               for s in entry["sentences"])

def ann_ids_to_mask(ann_ids, wh):
    """Return the OR-union of all annotation masks."""
    W, H = wh
    m = np.zeros((H, W), np.uint8)
    for aid in ann_ids:
        if aid in coco.anns:
            m |= coco.annToMask(coco.anns[aid]).astype(np.uint8)
    return m

class GRefDataset(Dataset):
    def __init__(self, entries, img_dir, img_size):
        self.samples, self.img_dir, self.img_size = [], img_dir, img_size
        self._wh_cache = {}
        for e in entries:
            path = os.path.join(img_dir, coco_name(e["image_id"]))
            if not os.path.isfile(path): continue
            for s in e["sentences"]:
                self.samples.append(dict(
                    img_id  = e["image_id"],
                    path    = path,
                    ann_ids = e["ann_id"],
                    clip_ids= encode_text_clip(s["sent"]),
                    text    = s["sent"]
                ))
        if not self.samples:
            raise RuntimeError("No samples")
    def __len__(self): return len(self.samples)
    def _wh(self, iid, path):
        if iid not in self._wh_cache:
            with Image.open(path) as im: self._wh_cache[iid] = im.size
        return self._wh_cache[iid]
    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s["path"]).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        img = torch.from_numpy(np.array(img)).permute(2,0,1).float()/255.
        W, H = Image.open(s["path"]).size
        m = ann_ids_to_mask(s["ann_ids"], (W, H))
        m = Image.fromarray(m*255).resize((IMG_SIZE,IMG_SIZE), Image.NEAREST)
        m = torch.from_numpy(np.array(m)//255)[None].float()
        return img, s["clip_ids"], m, s["text"], s["img_id"], idx

# Parse gRef + split (match latest logic)
with open(GREF_JSON) as f:
    gref = json.load(f)

train_entries, test_entries = [], []
for e in gref:
    if e.get("no_target") or not e["sentences"] or not e["ann_id"]:
        continue
    img_path = os.path.join(IMG_DIR, coco_name(e["image_id"]))
    if not os.path.isfile(img_path):
        continue
    trg = train_entries if e.get("split", "train") == "train" else test_entries
    trg.append(e)

print(f"Full Corpus Details:")
print(f"# Training Segments = {len(train_entries):,}")
print(f"# Testing Segments = {len(test_entries):,}")

random.shuffle(train_entries)
random.shuffle(test_entries)

if USE_FULL_DATASET:
    # Use dataset-defined splits; ignore TRAIN_RATIO entirely.
    train_subset = train_entries
    test_seed    = test_entries   # keep name for downstream compatibility
else:
    subset_refs  = train_entries[:ROWS_TOTAL]
    split_idx    = int(len(subset_refs) * TRAIN_RATIO)
    train_subset = subset_refs[:split_idx]
    test_seed    = subset_refs[split_idx:]

# Build CLIP token-id vocabulary over TRAIN (no HF/BERT)
train_vocab_ids = set()
for e in train_subset:
    for s in e["sentences"]:
        train_vocab_ids |= set(clip_vocab_ids_from_text(s["sent"]))

if USE_FULL_DATASET:
    # Use ALL test refs; no vocab-based filtering in full-dataset mode.
    test_subset = test_entries
    print(f"Final  train refs = {len(train_subset):,}  (dataset train split)")
    print(f"Final  test  refs = {len(test_subset):,}  (dataset test split)")
else:
    test_subset  = [e for e in test_seed if caption_in_clip_vocab(e, train_vocab_ids)]
    print(f"Final train refs = {len(train_subset):,} (= {len(train_subset)/ROWS_TOTAL:.0%} of {ROWS_TOTAL})")
    print(f"Final test  refs = {len(test_subset):,} (= {len(test_subset)/ROWS_TOTAL:.0%} of {ROWS_TOTAL})")

# =====================================================================
# Model + tv adapter + optim/sched
# =====================================================================
model_to = UTransformer(VOCAB, in_channels=3, num_classes=1,
                        base_channels=BASE_CHANNELS, num_heads=NUM_HEADS).to(device)

# Add small adapter so pooled C3 feature → 512 for TV loss
TXT_DIM = int(clip_model.text_projection.shape[1])  # usually 512
C3_DIM  = BASE_CHANNELS * 8                         # matches UTransformer c3 width
model_to.tv_proj = torch.nn.Linear(C3_DIM, TXT_DIM, bias=False).to(device)

# Freeze full CLIP text tower, unfreeze last 2 blocks
for p in model_to.film.clip.parameters():
    p.requires_grad_(False)
for blk in model_to.film.clip.transformer.resblocks[-2:]:
    for p in blk.parameters():
        p.requires_grad_(True)

# Backbone pooling helper (no grad)
def backbone_pool(model: UTransformer, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        x0 = model.inc(x)
        x0 = model.patch_embed(x0)
        x1 = model.down1(x0)
        x2 = model.down2(x1)
        x3 = model.down3(x2)          # (B, C3, H/32, W/32)
        feat = x3.mean(dim=(2, 3))    # (B, C3)
    return feat

# Losses (logits -> sigmoid inside for main combo; area/COM expect probs)

# ---------------------------------------------------------------------
# Visualization helpers shared by all viz functions
def overlay_mask(mask: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """
    Overlay a binary mask (HxW, {0,1}) onto an RGB image (HxWx3) with a red tint.
    """
    out = rgb.copy()
    sel = (mask.astype(np.uint8) > 0)
    out[sel] = (0.6 * out[sel] + np.array([255, 0, 0]) * 0.4).astype(np.uint8)
    return out

def dice_loss(logits, gt, eps=1e-6):
    p = torch.sigmoid(logits.float()); gt = gt.float()
    inter  = (p*gt).sum((2,3))
    union  = p.sum((2,3)) + gt.sum((2,3))
    return 1. - (2*inter+eps)/(union+eps)

def miou_loss(logits, gt, eps=1e-6):
    p = torch.sigmoid(logits.float()); gt = gt.float()
    inter  = (p*gt).sum((2,3))
    union  = (p+gt-p*gt).sum((2,3))
    return 1. - (inter+eps)/(union+eps)

def focal_loss(logits, gt, alpha=0.25, gamma=2.0, eps=1e-6):
    p  = torch.sigmoid(logits.float()).clamp(eps, 1-eps)
    ce = -(alpha*gt*torch.log(p) + (1-alpha)*(1-gt)*torch.log(1-p))
    mod = (1-p)**gamma
    return (mod * ce).mean(dim=(2,3))

def combo_loss(logits, gt, α=0.45, β=0.45, γ=0.10):
    d = dice_loss (logits, gt)
    i = miou_loss (logits, gt)
    f = focal_loss(logits, gt)
    return (α*d + β*i + γ*f).mean()

def mask_area_loss(prob, gt, eps: float = 1e-6):
    area_p = prob.float().mean((2, 3))
    area_g = gt.float().mean((2, 3))
    loss = -(area_g * (area_p.clamp_min(eps)).log()
             + (1.0 - area_g) * (1.0 - area_p).clamp_min(eps).log())
    return loss.mean()

def com_alignment_loss(prob, gt, eps=1e-6):
    B,_,H,W = prob.shape
    w = prob.clamp_min(eps)
    yy, xx = torch.meshgrid(
        torch.linspace(0,1,H,device=prob.device),
        torch.linspace(0,1,W,device=prob.device),
        indexing='ij')
    xx = xx[None,None]; yy = yy[None,None]
    cx_p = (w*xx).sum((2,3)) / w.sum((2,3))
    cy_p = (w*yy).sum((2,3)) / w.sum((2,3))
    cx_g = (gt*xx).sum((2,3)) / gt.sum((2,3))
    cy_g = (gt*yy).sum((2,3)) / gt.sum((2,3))
    return ((cx_p-cx_g)**2 + (cy_p-cy_g)**2).mean()

def contrastive_tv_loss(vis_feat, txt_feat, T=0.07):
    vis = F.normalize(vis_feat, dim=-1)
    txt = F.normalize(txt_feat, dim=-1)
    return 1. - (vis * txt).sum(-1).mean()

def film_reg(gammas, betas, w=REG_W_FiLM):
    reg = 0.
    for g, b in zip(gammas, betas):
        reg += ((g - 1).abs() + b.abs()).mean()
    return w * reg

# Smooth squashing used for FiLM diagnostics/reg (matches latest)
def squash_film_lists(gE, bE, gD, bD):
    def _squash_list(xs, lo, hi):
        mid  = 0.5 * (hi + lo)
        half = 0.5 * (hi - lo)
        return [mid + half * torch.tanh((x - mid) / half) for x in xs]
    gE = _squash_list(gE, 0.5, 2.0);  bE = [torch.tanh(x) for x in bE]
    gD = _squash_list(gD, 0.5, 2.5);  bD = [torch.tanh(x) for x in bD]
    return gE, bE, gD, bD

def freeze_backbone(m, flag=True):
    for name, sub in m.named_children():
        if name != "film":
            for p in sub.parameters():
                p.requires_grad_(not flag)
            # Keep BN layers in eval when frozen; train when unfrozen
            if flag:
                sub.apply(lambda mod: mod.eval() if isinstance(mod, (torch.nn.BatchNorm2d,
                                                                     torch.nn.SyncBatchNorm)) else None)
            else:
                sub.apply(lambda mod: mod.train() if isinstance(mod, (torch.nn.BatchNorm2d,
                                                                      torch.nn.SyncBatchNorm)) else None)

# Histories
hist = _dd(list)
loss_history = {"train": [], "val": [], "shuffle": [], "combo": []}

# Probe helper
def _probe_gamma_beta(clip_tok_batch: torch.Tensor):
    with torch.no_grad():
        gE, bE, gD, bD = model_to.film(clip_tok_batch.to(device))
    enc_stats = ([(g.mean().item(), g.std().item()) for g in gE],
                 [(b.mean().item(), b.std().item()) for b in bE])
    dec_stats = ([(g.mean().item(), g.std().item()) for g in gD],
                 [(b.mean().item(), b.std().item()) for b in bD])
    return enc_stats, dec_stats

# Save/Load checkpoint (DDP-friendly)
def save_checkpoint(
    ep: int,
    model,
    opt,
    scheduler,
    best_val: float,
    patience: int,
    extra_unfrozen: bool,
    hist_obj,
    loss_hist_obj,
    warmup_done: bool = None,
    warmup_epochs_completed: int = None,
    keep_history: bool = False,
    save_best: bool = False,
    ckpt_file=CKPT_FILE,
    ckpt_dir=CKPT_DIR
):
    state = {
        "epoch"          : ep,
        "model"          : model.state_dict(),
        "opt"            : opt.state_dict(),
        "sched"          : scheduler.state_dict(),
        "best_val"       : best_val,
        "patience"       : patience,
        "extra_unfrozen" : extra_unfrozen,
        "hist"           : hist_obj,
        "loss_history"   : loss_hist_obj,
    }
    if warmup_done is not None:
        state["warmup_done"] = bool(warmup_done)
    if warmup_epochs_completed is not None:
        state["warmup_epochs_completed"] = int(warmup_epochs_completed)
    torch.save(state, ckpt_file)
    if save_best:
        torch.save(state, ckpt_dir / "best.pt")
    if keep_history:
        torch.save(state, ckpt_dir / f"epoch_{ep:03d}.pt")

# ---------------------------------------------------------------------
def setup_ddp(rank, world_size):
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)

def cleanup_ddp():
    destroy_process_group()

# ---------------------------------------------------------------------
# Visualization
def run_visualization_and_timing(model, device, test_dl):
    model.eval()
    preds, gts, caps, bases = {}, {}, {}, {}
    inf_times = []
    with torch.no_grad():
        for img, ids, gt, cap, img_id, _sid in test_dl:
            img, ids = img.to(device), ids.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
                t0 = torch.cuda.Event(enable_timing=True)
                t1 = torch.cuda.Event(enable_timing=True)
                t0.record()
                with amp.autocast('cuda', dtype=torch.float16):
                    prob_batch = model(img, ids)   # model currently returns prob
                t1.record()
                torch.cuda.synchronize()
                batch_ms = t0.elapsed_time(t1)
            else:
                t0 = time.perf_counter()
                prob_batch = model(img, ids)
                batch_ms = (time.perf_counter() - t0) * 1000

            inf_times.extend([batch_ms / img.size(0)] * img.size(0))
            prob = prob_batch.cpu()

            for b in range(img.size(0)):
                iid = int(img_id[b])
                preds.setdefault(iid, []).append(prob[b, 0].numpy())
                gts.setdefault(iid, []).append(gt[b, 0].numpy())
                caps.setdefault(iid, []).append(cap[b])
                if iid not in bases:
                    rgb = (img[b].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    bases[iid] = rgb

    avg_ms = np.mean(inf_times); std_ms = np.std(inf_times)
    print(f"\n Average Forward-Pass Time  : {avg_ms:6.2f} ms / image (± {std_ms:4.2f} ms  •  {len(inf_times)} images)")


    for iid in preds:
        base = bases[iid]
        # print(f"\n=== Test Image {iid} ===")
        for prob, gt_mask, caption in zip(preds[iid], gts[iid], caps[iid]):
            pred_bin = (prob > 0.5).astype(np.uint8)
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(overlay_mask(gt_mask, base)); plt.axis('Off'); plt.title("Ground Truth")
            plt.subplot(1, 2, 2)
            plt.imshow(overlay_mask(pred_bin, base)); plt.axis('Off'); plt.title("Prediction")
            plt.suptitle(caption, y=0.95)
            plt.tight_layout()
            os.makedirs("artifacts/vis", exist_ok=True)
            out_path = os.path.join("artifacts/vis", f"{iid}_cap{len(preds[iid])-1}.png")
            plt.savefig(out_path, dpi=150)
            # print(f"[viz] saved {out_path}")
            plt.close()

# ---------------------------------------------------------------------
# DDP worker
def main_worker(rank, world_size):
    try:
        if rank == 0: print("[DDP] Starting processes …")
        setup_ddp(rank, world_size)
        dev = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(rank)
        torch.manual_seed(42); np.random.seed(42); random.seed(42)

        # Model to device and wrap DDP
        model_to.to(dev)
        model_ddp = DDP(model_to, device_ids=[rank], output_device=rank, find_unused_parameters=True)

        # Build datasets (keep handles for DDP caption-shuffle)
        train_dataset = GRefDataset(train_subset, IMG_DIR, IMG_SIZE)
        test_dataset  = GRefDataset(test_subset,  IMG_DIR, IMG_SIZE)

        train_sampler = DistributedSampler(train_subset, num_replicas=world_size, rank=rank, shuffle=True)
        test_sampler  = DistributedSampler(test_subset,  num_replicas=world_size, rank=rank, shuffle=False)

        train_dl = DataLoader(
            train_dataset,
            batch_size=PER_RANK_BATCH,
            sampler=train_sampler,
            num_workers=0,
            pin_memory=True
        )
        test_dl = DataLoader(
            test_dataset,
            batch_size=PER_RANK_BATCH,
            sampler=test_sampler,
            num_workers=0,
            pin_memory=True
        )

        # Param groups
        vis_params, text_params = [], []
        gain_param   = [model_ddp.module.film_gain_log]
        for n, p in model_ddp.module.named_parameters():
            if p is model_ddp.module.film_gain_log:
                continue
            if ("film." in n) or ("embed" in n) or ("xattn" in n):
                text_params.append(p)
            else:
                vis_params.append(p)

        opt = torch.optim.AdamW(
            [
                {"params": vis_params,  "lr": LR_IMG},
                {"params": text_params, "lr": LR_TEXT},
                {"params": gain_param,  "lr": LR_GAIN},
            ],
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3, min_lr=1e-6)

        # State vars
        start_epoch = 1
        best_val = float("inf")
        patience = 0
        extra_clip_unfrozen = False
        warmup_done = False
        warmup_epochs_completed = 0

        # Optional resume (rank 0 loads; then broadcast)
        if rank == 0 and CKPT_FILE.exists():
            print(f"[resume] loading {CKPT_FILE.name}")
            state = torch.load(CKPT_FILE, map_location=dev)
            model_ddp.module.load_state_dict(state["model"])
            opt.load_state_dict(state["opt"])
            scheduler.load_state_dict(state["sched"])
            start_epoch         = int(state.get("epoch", 0)) + 1
            best_val            = float(state.get("best_val", float("inf")))
            patience            = int(state.get("patience", 0))
            extra_clip_unfrozen = bool(state.get("extra_unfrozen", False))
            warmup_done         = bool(state.get("warmup_done", False))
            warmup_epochs_completed = int(state.get("warmup_epochs_completed", 0))
            try:
                for k,v in state.get("hist", {}).items(): hist[k] = v
                for k,v in state.get("loss_history", {}).items(): loss_history[k] = v
            except Exception:
                pass
            print(f"↪︎ resumed from epoch {state.get('epoch', 0)}  (best val = {best_val:.4f})")

        # Broadcast scalars to all ranks
        start_ep_t = torch.tensor(start_epoch, device=dev, dtype=torch.long)
        best_val_t = torch.tensor(best_val,   device=dev, dtype=torch.float32)
        patience_t = torch.tensor(patience,   device=dev, dtype=torch.long)
        unfrozen_t = torch.tensor(1 if extra_clip_unfrozen else 0, device=dev, dtype=torch.long)
        wdone_t    = torch.tensor(1 if warmup_done else 0, device=dev, dtype=torch.long)
        wcomp_t    = torch.tensor(warmup_epochs_completed, device=dev, dtype=torch.long)
        dist.broadcast(start_ep_t, src=0)
        dist.broadcast(best_val_t, src=0)
        dist.broadcast(patience_t, src=0)
        dist.broadcast(unfrozen_t, src=0)
        dist.broadcast(wdone_t, src=0)
        dist.broadcast(wcomp_t, src=0)
        start_epoch = int(start_ep_t.item())
        best_val    = float(best_val_t.item())
        patience    = int(patience_t.item())
        extra_clip_unfrozen = bool(unfrozen_t.item())
        warmup_done = bool(wdone_t.item())
        warmup_epochs_completed = int(wcomp_t.item())

        # Initial freeze policy (text tower frozen; last 2 blocks unfrozen)
        for p in model_ddp.module.film.clip.parameters():
            p.requires_grad_(False)
        for blk in model_ddp.module.film.clip.transformer.resblocks[-2:]:
            for p in blk.parameters():
                p.requires_grad_(True)

        # Warmup (resumable): text-only cosine sim, backbone frozen; vision LR=0
        freeze_backbone(model_ddp.module, True)
        for p in model_ddp.module.film.parameters():
            p.requires_grad_(True)

        vis_group, text_group, gain_group = opt.param_groups
        orig_lr_vis = vis_group["lr"]
        if not warmup_done:
            vis_group["lr"] = 0.0

        if rank == 0: init_log_csv()

        
        # Resume-aware warmup loop
        if not warmup_done:
            for warm_ep in range(warmup_epochs_completed, WARMUP_EPOCHS):
                model_ddp.train()
                train_sampler.set_epoch(warm_ep)
                cum_loss = n = 0
                pbar_w = tqdm(train_dl, desc=f"[Rank {rank}] warm-up {warm_ep+1}/{WARMUP_EPOCHS}", leave=False)
                for img, clip_ids, *_ in pbar_w:
                    img, clip_ids = img.to(dev), clip_ids.to(dev)
                    with torch.no_grad():
                        vis_feat = backbone_pool(model_ddp.module, img)  # (B, C3)
                        vis_feat = model_ddp.module.tv_proj(vis_feat)    # (B, 512)
                    txt_feat = model_ddp.module.film.clip.encode_text(clip_ids)
                    loss = contrastive_tv_loss(vis_feat, txt_feat)
                    loss.backward()
                    opt.step(); opt.zero_grad()
                    cum_loss += loss.item() * img.size(0); n += img.size(0)
                    pbar_w.set_postfix(loss=f"{loss.item():.3f}")
                # update warmup progress and checkpoint (rank0)
                warmup_epochs_completed = warm_ep + 1
                if rank == 0:
                    print(f"  Warm-Up Epoch {warm_ep+1}: Loss {cum_loss/n:.4f}")
                    save_checkpoint(
                        ep=0, model=model_ddp.module, opt=opt, scheduler=scheduler,
                        best_val=best_val, patience=patience, extra_unfrozen=extra_clip_unfrozen,
                        hist_obj=hist, loss_hist_obj=loss_history,
                        warmup_done=False, warmup_epochs_completed=warmup_epochs_completed,
                        keep_history=False, save_best=False
                    )
            warmup_done = True
            vis_group["lr"] = orig_lr_vis
            if rank == 0:
                print(f"Warm-Up Done: Backbone frozen for next {FREEZE_EPOCHS} epochs")
                save_checkpoint(
                    ep=0, model=model_ddp.module, opt=opt, scheduler=scheduler,
                    best_val=best_val, patience=patience, extra_unfrozen=extra_clip_unfrozen,
                    hist_obj=hist, loss_hist_obj=loss_history,
                    warmup_done=True, warmup_epochs_completed=warmup_epochs_completed,
                    keep_history=False, save_best=False
                )

        # Unfreeze everything after warmup
        for p in model_ddp.parameters():
            p.requires_grad_(True)

        for ep in range(start_epoch, EPOCHS + 1):
            print(f"\n[Rank {rank}] Starting Epoch {ep}/{EPOCHS}")
            train_sampler.set_epoch(ep)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(dev)

            start_train = time.time()

            if ep == FREEZE_EPOCHS + 1:
                freeze_backbone(model_ddp.module, False)
                # reduce vision LR for two epochs after unfreeze
                opt.param_groups[0]["lr"] = LR_IMG * 0.5
                if rank == 0:
                    print(f"[Epoch {ep}] Backbone Unfrozen; vision LR temporarily halved")

            # choose grad-clip based on post-unfreeze window
            clip_norm_cur = CLIP_NORM_AFTER_UNFREEZE if (FREEZE_EPOCHS+1 <= ep <= FREEZE_EPOCHS+2) else CLIP_NORM_BASE
            if ep == FREEZE_EPOCHS + 3:
                opt.param_groups[0]["lr"] = LR_IMG   # restore vision LR

            λ_div = lambda_div(ep)

            # ------------------ TRAIN ------------------
            model_ddp.train()
            opt.zero_grad(set_to_none=True)
            run, run_main, tot = 0., 0., 0
            pbar = tqdm(train_dl, desc=f"[Rank {rank}] Ep{ep}", leave=False)
            for step, (img, clip_ids, gt, _, img_id, _sid) in enumerate(pbar, 1):
                img, clip_ids, gt = img.to(dev), clip_ids.to(dev), gt.to(dev)
                prob = model_ddp(img, clip_ids)  # model currently returns prob
                # turn prob -> logits safely so combo_loss sees logits
                eps = 1e-6
                logits = torch.log(prob.clamp(eps,1-eps) / (1 - prob.clamp(eps,1-eps)))

                loss_main = combo_loss(logits, gt)
                run_main += loss_main.item() * img.size(0)

                # TV loss (small)
                with torch.no_grad():
                    vis_feat = backbone_pool(model_ddp.module, img)
                    vis_feat = model_ddp.module.tv_proj(vis_feat)
                txt_feat = model_ddp.module.film.clip.encode_text(clip_ids)
                loss_tv   = TV_W       * contrastive_tv_loss(vis_feat, txt_feat)

                # Priors (use prob)
                loss_area = REG_W_AREA * mask_area_loss(prob, gt)
                loss_com  = REG_W_COM  * com_alignment_loss(prob, gt)

                # Divergence per image-id (use prob)
                idx_by_img = {}; div_acc = pairs = 0.
                for k, iid in enumerate(img_id): idx_by_img.setdefault(int(iid), []).append(k)
                for idxs in idx_by_img.values():
                    for i in range(len(idxs)):
                        for j in range(i+1, len(idxs)):
                            i1, i2 = idxs[i], idxs[j]
                            if torch.equal(gt[i1], gt[i2]): continue
                            div_acc += (prob[i1] - prob[i2]).abs().mean()
                            pairs += 1
                div_loss = (div_acc / pairs) if pairs else torch.tensor(0., device=dev)

                # FiLM regulariser on squashed values
                gE, bE, gD, bD = model_ddp.module.film(clip_ids)
                gE, bE, gD, bD = squash_film_lists(gE, bE, gD, bD)
                reg_term = film_reg(gE + gD, bE + bD, w=REG_W_FiLM)

                loss = (loss_main
                        + λ_div * div_loss
                        + reg_term + loss_tv + loss_area + loss_com)

                loss.backward()
                if step % ACC_STEPS == 0 or step == len(train_dl):
                    torch.nn.utils.clip_grad_norm_(model_ddp.parameters(), clip_norm_cur)
                    opt.step()
                    # (we rely on squashing; no hard clamp here)
                    opt.zero_grad(set_to_none=True)

                run += loss.item() * img.size(0)
                tot += img.size(0)
                pbar.set_postfix(
                    Lmain=f"{loss_main.item():.3f}",
                    TV=f"{loss_tv.item():.3e}",
                    area=f"{loss_area.item():.3e}",
                    COM=f"{loss_com.item():.3e}",
                    div=f"{div_loss.item():.3e}",
                    λ=f"{λ_div:.2f}",
                )

            train_loss = run / tot
            combo_train = run_main / tot
            end_train = time.time()
            train_time = end_train - start_train

            # ---------------- VALIDATION --------------
            start_val = time.time()
            model_ddp.eval()
            v_sum = torch.tensor(0., device=dev)
            v_n   = torch.tensor(0., device=dev)
            with torch.no_grad():
                pbar_v = tqdm(test_dl, desc=f"[Rank {rank}] Val Ep{ep}", leave=False)
                for img, clip_ids, gt, _, img_id, _sid in pbar_v:
                    img, clip_ids, gt = img.to(dev), clip_ids.to(dev), gt.to(dev)
                    prob_val = model_ddp(img, clip_ids)
                    eps = 1e-6
                    logits_val = torch.log(prob_val.clamp(eps,1-eps) / (1 - prob_val.clamp(eps,1-eps)))
                    v_loss = combo_loss(logits_val, gt)
                    v_sum += v_loss * img.size(0)
                    v_n   += img.size(0)
            dist.all_reduce(v_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(v_n,   op=dist.ReduceOp.SUM)
            val_loss = (v_sum / v_n).item()
            end_val = time.time()
            val_time = end_val - start_val

            # ---------------- CAPTION-SHUFFLE PROBE (every 5 epochs, DDP) -----------
            shuffle_loss_val = float('nan')
            if ((ep-1) % SHUFFLE_PROBE_EVERY) == 0:
                # Build ALL clip token ids on rank 0, then broadcast to peers
                if rank == 0:
                    with torch.no_grad():
                        all_clip_ids = torch.stack(
                            [test_dataset.samples[i]["clip_ids"] for i in range(len(test_dataset))],
                            dim=0
                        ).to(dev)  # [N,77]
                    perm = torch.randperm(all_clip_ids.size(0), device=dev)
                else:
                    # placeholders for broadcast
                    all_clip_ids = torch.empty((len(test_dataset), 77), dtype=torch.long, device=dev)
                    perm = torch.empty((len(test_dataset),), dtype=torch.long, device=dev)
                dist.broadcast(all_clip_ids, src=0)
                dist.broadcast(perm,         src=0)

                s_sum = torch.tensor(0., device=dev)
                s_n   = torch.tensor(0., device=dev)
                eps   = 1e-6
                with torch.no_grad():
                    pbar_s = tqdm(test_dl, desc=f"[Rank {rank}] Shuffle Ep{ep}", leave=False)
                    for img, _, gt, _, _img_id, sid in pbar_s:
                        img, gt = img.to(dev), gt.to(dev)
                        sid = sid.to(dev) if torch.is_tensor(sid) else torch.as_tensor(sid, device=dev)
                        # mismatched captions via global permutation
                        clip_ids_shuf = all_clip_ids[perm[sid]]
                        prob_s   = model_ddp(img, clip_ids_shuf)
                        logits_s = torch.log(prob_s.clamp(eps,1-eps) / (1 - prob_s.clamp(eps,1-eps)))
                        s_loss   = combo_loss(logits_s, gt)
                        s_sum   += s_loss * img.size(0)
                        s_n     += img.size(0)
                # global average over ranks
                dist.all_reduce(s_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(s_n,   op=dist.ReduceOp.SUM)
                shuffle_loss_val = (s_sum / s_n).item()

            local_max_mem = torch.tensor(torch.cuda.max_memory_allocated(dev), device=dev)
            dist.all_reduce(local_max_mem, op=dist.ReduceOp.MAX)
            max_mem_mb = local_max_mem.item() / 1024**2

            # -------------- Rank 0 logging/metrics --------------
            if rank == 0:
                try:
                    run_visualization_and_timing(model=model_ddp.module, device=dev, test_dl=test_dl)
                    if (ep % 5) == 0:
                        save_epoch_pair_figure(ep, model_ddp, dev, train_dataset, test_dataset)
                except Exception as e:
                    print(f"[Rank 0] Visualization Failed: {e}")

                print(
                    f"Epoch {ep:02d} | Training Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f} | "
                    f"λ={λ_div:.2f} | Training Time: {train_time:.1f}s | Validation Time: {val_time:.1f}s | "
                    f"Peak GPU Memory across All Ranks: {max_mem_mb:.1f} MB"
                )

                # Attn entropy (if collected)
                xH = np.mean(model_ddp.module.attn_entropy) if len(getattr(model_ddp.module, "attn_entropy", [])) else 0.0
                if hasattr(model_ddp.module, "attn_entropy"): model_ddp.module.attn_entropy.clear()

                # Probe γ/β
                sample_caps = [train_subset[i]["sentences"][0]["sent"]
                               for i in range(min(PROBE_BATCH, len(train_subset)))]
                clip_tok_batch = encode_text_clip(sample_caps).to(dev)
                (enc_g, enc_b), (dec_g, dec_b) = _probe_gamma_beta(clip_tok_batch)

                # Optional: damp text LR if σ spikes
                max_sigma = max([s for _,s in enc_g] + [s for _,s in enc_b] +
                                [s for _,s in dec_g] + [s for _,s in dec_b]) if enc_g else 0.0
                if max_sigma > SIGMA_TOL_FOR_TXT_LR and abs(opt.param_groups[1]["lr"] - LR_TEXT*0.5) > 1e-12:
                    opt.param_groups[1]["lr"] = LR_TEXT * 0.5
                    print(f"[INFO] large FiLM σ={max_sigma:.1f} → halving text LR to {opt.param_groups[1]['lr']:.2e}")

                # Hist
                hist["log_gain"].append(model_ddp.module.film_gain_log.item())
                hist["xattn_H"].append(xH)
                hist["enc_γμ"].append([m for m, _ in enc_g]); hist["enc_γσ"].append([s for _, s in enc_g])
                hist["enc_βμ"].append([m for m, _ in enc_b]); hist["enc_βσ"].append([s for _, s in enc_b])
                hist["dec_γμ"].append([m for m, _ in dec_g]); hist["dec_γσ"].append([s for _, s in dec_g])
                hist["dec_βμ"].append([m for m, _ in dec_b]); hist["dec_βσ"].append([s for _, s in dec_b])

                # Loss history for post-run plots
                loss_history["train"].append(train_loss)
                loss_history["val"].append(val_loss)
                loss_history["shuffle"].append(shuffle_loss_val)
                loss_history["combo"].append(combo_train)

                # CSV
                log_metrics(
                    epoch=ep,
                    log_gain=model_ddp.module.film_gain_log.item(),
                    xattn_H=xH,
                    enc_γμ=hist["enc_γμ"][-1] if hist["enc_γμ"] else [],
                    enc_γσ=hist["enc_γσ"][-1] if hist["enc_γσ"] else [],
                    enc_βμ=hist["enc_βμ"][-1] if hist["enc_βμ"] else [],
                    enc_βσ=hist["enc_βσ"][-1] if hist["enc_βσ"] else [],
                    dec_γμ=hist["dec_γμ"][-1] if hist["dec_γμ"] else [],
                    dec_γσ=hist["dec_γσ"][-1] if hist["dec_γσ"] else [],
                    dec_βμ=hist["dec_βμ"][-1] if hist["dec_βμ"] else [],
                    dec_βσ=hist["dec_βσ"][-1] if hist["dec_βσ"] else [],
                    train_loss=train_loss, combo_train=combo_train, val_loss=val_loss,
                    lambda_div=λ_div,
                    train_time=train_time, val_time=val_time,
                    peak_gpu_mem_mb=max_mem_mb
                )

                # Best/patience/unfreeze + save
                improved = (val_loss < best_val - min_delta)
                if improved:
                    best_val = val_loss
                    patience = 0
                else:
                    patience += 1
                    if patience == 8 and not extra_clip_unfrozen:
                        for blk in model_ddp.module.film.clip.transformer.resblocks[-4:]:
                            for p in blk.parameters():
                                p.requires_grad_(True)
                        extra_clip_unfrozen = True
                        print("[INFO] last-4 CLIP blocks unfrozen")

                save_checkpoint(
                    ep,
                    model=model_ddp.module,
                    opt=opt,
                    scheduler=scheduler,
                    best_val=best_val,
                    patience=patience,
                    extra_unfrozen=extra_clip_unfrozen,
                    hist_obj=hist,
                    loss_hist_obj=loss_history,
                    warmup_done=warmup_done,
                    warmup_epochs_completed=warmup_epochs_completed,
                    keep_history=False,
                    save_best=improved
                )

                # --- Early stopping decision (rank 0 decides, broadcast to all) ---
                if rank == 0 and patience >= PATIENCE_MAX:
                    print(f"[EarlyStop] No val improvement for {patience} epochs "
                          f"(≥ PATIENCE_MAX={PATIENCE_MAX}). Stopping.")
                stop_t = torch.tensor(1 if (rank == 0 and patience >= PATIENCE_MAX) else 0,
                                      device=dev, dtype=torch.long)
                dist.broadcast(stop_t, src=0)
                if stop_t.item() == 1:
                    break

            # Broadcast updated small state so all ranks match decisions
            best_val_t = torch.tensor(best_val, device=dev, dtype=torch.float32)
            patience_t = torch.tensor(patience, device=dev, dtype=torch.long)
            unfrozen_t = torch.tensor(1 if extra_clip_unfrozen else 0, device=dev, dtype=torch.long)
            dist.broadcast(best_val_t, src=0)
            dist.broadcast(patience_t, src=0)
            dist.broadcast(unfrozen_t, src=0)
            best_val = float(best_val_t.item())
            patience = int(patience_t.item())
            extra_clip_unfrozen = bool(unfrozen_t.item())

            # If unfreeze was toggled, ensure all ranks apply it
            if extra_clip_unfrozen:
                for blk in model_ddp.module.film.clip.transformer.resblocks[-4:]:
                    for p in blk.parameters():
                        p.requires_grad_(True)

            # Step scheduler on all ranks with reduced val_loss
            scheduler.step(val_loss)

    except Exception as e:
        print(f"[Rank {rank}] Exception occurred: {e}")
        import traceback; traceback.print_exc()
    finally:
        cleanup_ddp()

# ---------------------------------------------------------------------
# Single-GPU fallback (kept, aligned with latest logic)
def run_fallback_training(dev):
    print("[Fallback]: Running Single-GPU Training Loop.")
    global hist, loss_history

    train_dl = DataLoader(
        GRefDataset(train_subset, IMG_DIR, IMG_SIZE),
        batch_size=PER_RANK_BATCH,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    test_dl = DataLoader(
        GRefDataset(test_subset, IMG_DIR, IMG_SIZE),
        batch_size=PER_RANK_BATCH,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Build optimizer/scheduler here (no globals)
    vis_params, text_params = [], []
    gain_param   = [model_to.film_gain_log]
    for n, p in model_to.named_parameters():
        if p is model_to.film_gain_log:
            continue
        if ("film." in n) or ("embed" in n) or ("xattn" in n):
            text_params.append(p)
        else:
            vis_params.append(p)

    opt = torch.optim.AdamW(
        [
            {"params": vis_params,  "lr": LR_IMG},
            {"params": text_params, "lr": LR_TEXT},
            {"params": gain_param,  "lr": LR_GAIN},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3, min_lr=1e-6)

    # Optional resume
    start_epoch = 1
    best_val = float('inf')
    patience = 0
    extra_clip_unfrozen = False
    warmup_done = False
    warmup_epochs_completed = 0
    if CKPT_FILE.exists():
        print(f"[resume] loading {CKPT_FILE.name}")
        state = torch.load(CKPT_FILE, map_location=dev)
        model_to.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        scheduler.load_state_dict(state["sched"])
        start_epoch         = int(state.get("epoch", 0)) + 1
        best_val            = float(state.get("best_val", float("inf")))
        patience            = int(state.get("patience", 0))
        extra_clip_unfrozen = bool(state.get("extra_unfrozen", False))
        warmup_done         = bool(state.get("warmup_done", False))
        warmup_epochs_completed = int(state.get("warmup_epochs_completed", 0))
        try:
            for k,v in state.get("hist", {}).items(): hist[k] = v
            for k,v in state.get("loss_history", {}).items(): loss_history[k] = v
        except Exception:
            pass
        print(f"↪︎ resumed from epoch {state.get('epoch', 0)}  (best val = {best_val:.4f})")

    # Initial freeze policy
    for p in model_to.film.clip.parameters(): p.requires_grad_(False)
    for blk in model_to.film.clip.transformer.resblocks[-2:]:
        for p in blk.parameters(): p.requires_grad_(True)

    # Warmup
    freeze_backbone(model_to, True)
    for p in model_to.film.parameters(): p.requires_grad_(True)
    vis_group, text_group, gain_group = opt.param_groups
    orig_lr_vis = vis_group["lr"]
    if not warmup_done:
        vis_group["lr"]  = 0.0

    if not warmup_done:
        print(f"Running {WARMUP_EPOCHS}-epoch cosine-similarity warm-up … (resume @ {warmup_epochs_completed}/{WARMUP_EPOCHS})")
        for warm_ep in range(warmup_epochs_completed, WARMUP_EPOCHS):
            cum_loss = n = 0
            pbar_w = tqdm(train_dl, desc=f"[Rank 0] WARMUP {warm_ep+1}/{WARMUP_EPOCHS}", leave=False)
            for img, clip_ids, *_ in pbar_w:
                img, clip_ids = img.to(dev), clip_ids.to(dev)
                with torch.no_grad():
                    vis_feat = backbone_pool(model_to, img)
                    vis_feat = model_to.tv_proj(vis_feat)
                txt_feat = model_to.film.clip.encode_text(clip_ids)
                loss = contrastive_tv_loss(vis_feat, txt_feat)
                loss.backward()
                opt.step(); opt.zero_grad()
                cum_loss += loss.item() * img.size(0); n += img.size(0)
                pbar_w.set_postfix(loss=f"{loss.item():.3f}")
            warmup_epochs_completed = warm_ep + 1
            print(f"  warm-up epoch {warm_ep+1}:  loss {cum_loss/n:.4f}")
            save_checkpoint(
                ep=0, model=model_to, opt=opt, scheduler=scheduler,
                best_val=best_val, patience=patience, extra_unfrozen=extra_clip_unfrozen,
                hist_obj=hist, loss_hist_obj=loss_history,
                warmup_done=False, warmup_epochs_completed=warmup_epochs_completed,
                keep_history=False, save_best=False
            )
        warmup_done = True
        vis_group["lr"]  = orig_lr_vis
        print(f"Warm-up Done — Backbone will stay frozen for the next {FREEZE_EPOCHS} epochs\n")
        save_checkpoint(
            ep=0, model=model_to, opt=opt, scheduler=scheduler,
            best_val=best_val, patience=patience, extra_unfrozen=extra_clip_unfrozen,
            hist_obj=hist, loss_hist_obj=loss_history,
            warmup_done=True, warmup_epochs_completed=warmup_epochs_completed,
            keep_history=False, save_best=False
        )

    # Unfreeze all
    for p in model_to.parameters(): p.requires_grad_(True)

    for ep in range(start_epoch, EPOCHS + 1):
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(dev)
        start_train = time.time()

        if ep == FREEZE_EPOCHS+1:
            freeze_backbone(model_to, False)
            opt.param_groups[0]["lr"] = LR_IMG * 0.5
            print(f"[Epoch {ep}] backbone unfrozen (vision LR 0.5x for 2 epochs)")
        clip_norm_cur = CLIP_NORM_AFTER_UNFREEZE if (FREEZE_EPOCHS+1 <= ep <= FREEZE_EPOCHS+2) else CLIP_NORM_BASE
        if ep == FREEZE_EPOCHS+3:
            opt.param_groups[0]["lr"] = LR_IMG

        λ_div = lambda_div(ep)
        # TRAIN
        model_to.train(); run, run_main, tot = 0., 0., 0
        opt.zero_grad(set_to_none=True)
        pbar = tqdm(train_dl, desc=f"[Rank 0] Ep{ep}", leave=False)
        for step, (img, clip_ids, gt, _, img_id, _sid) in enumerate(pbar, 1):
            img, clip_ids, gt = img.to(dev), clip_ids.to(dev), gt.to(dev)
            prob = model_to(img, clip_ids)
            eps=1e-6; logits = torch.log(prob.clamp(eps,1-eps)/(1-prob.clamp(eps,1-eps)))
            loss_main = combo_loss(logits, gt)
            run_main += loss_main.item()*img.size(0)

            with torch.no_grad():
                vis_feat = backbone_pool(model_to, img)
                vis_feat = model_to.tv_proj(vis_feat)
            txt_feat = model_to.film.clip.encode_text(clip_ids)
            loss_tv   = TV_W       * contrastive_tv_loss(vis_feat, txt_feat)
            loss_area = REG_W_AREA * mask_area_loss(prob, gt)
            loss_com  = REG_W_COM  * com_alignment_loss(prob, gt)

            idx_by_img={}; div_acc=pairs=0.
            for k,iid in enumerate(img_id): idx_by_img.setdefault(int(iid),[]).append(k)
            for idxs in idx_by_img.values():
                for i in range(len(idxs)):
                    for j in range(i+1,len(idxs)):
                        i1,i2=idxs[i],idxs[j]
                        if torch.equal(gt[i1],gt[i2]): continue
                        div_acc += (prob[i1]-prob[i2]).abs().mean()
                        pairs+=1
            div_loss = (div_acc/pairs) if pairs else torch.tensor(0.,device=dev)

            gE, bE, gD, bD = model_to.film(clip_ids)
            gE, bE, gD, bD = squash_film_lists(gE, bE, gD, bD)
            reg_term = film_reg(gE + gD, bE + bD, w=REG_W_FiLM)

            loss = (loss_main + λ_div*div_loss + reg_term + loss_tv + loss_area + loss_com)
            loss.backward()
            if step%ACC_STEPS==0 or step==len(train_dl):
                torch.nn.utils.clip_grad_norm_(model_to.parameters(), clip_norm_cur)
                opt.step(); opt.zero_grad(set_to_none=True)

            run += loss.item()*img.size(0); tot+=img.size(0)
            pbar.set_postfix(
                Lmain=f"{loss_main.item():.3f}",
                TV=f"{loss_tv.item():.3e}",
                area=f"{loss_area.item():.3e}",
                COM=f"{loss_com.item():.3e}",
                div=f"{div_loss.item():.3e}",
                λ=f"{λ_div:.2f}"
            )

        train_loss = run/tot
        end_train=time.time()
        train_time = end_train - start_train

        # VALIDATION
        start_val = time.time()
        model_to.eval(); v_sum=v_n=0
        with torch.no_grad():
            pbar_v = tqdm(test_dl, desc=f"[Rank 0] Val Ep{ep}", leave=False)
            for img, clip_ids, gt, _, img_id, _sid in pbar_v:
                img, clip_ids, gt = img.to(dev), clip_ids.to(dev), gt.to(dev)
                prob_val = model_to(img, clip_ids)
                eps=1e-6; logits_val = torch.log(prob_val.clamp(eps,1-eps)/(1-prob_val.clamp(eps,1-eps)))
                v_loss = combo_loss(logits_val, gt)
                v_sum += v_loss.item()*img.size(0); v_n += img.size(0)
        val_loss = v_sum/v_n
        end_val = time.time()
        val_time = end_val - start_val

        max_mem_mb = (torch.cuda.max_memory_allocated(dev) / 1024**2) if torch.cuda.is_available() else 0
        print(
            f"Epoch {ep:02d} | Training Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f} | "
            f"λ={λ_div:.2f} | Training Time: {train_time:.1f}s | Validation Time: {val_time:.1f}s | "
            f"Peak GPU Memory: {max_mem_mb:.1f} MB"
        )

        # Probe stats
        hist["log_gain"].append(model_to.film_gain_log.item())
        hist["xattn_H"].append( np.mean(getattr(model_to, "attn_entropy", [])) if hasattr(model_to, "attn_entropy") else 0.0 )
        if hasattr(model_to, "attn_entropy"): model_to.attn_entropy.clear()

        sample_caps = [train_subset[i]["sentences"][0]["sent"] for i in range(min(PROBE_BATCH, len(train_subset)))]
        clip_tok_batch = encode_text_clip(sample_caps).to(dev)
        (enc_g, enc_b), (dec_g, dec_b) = _probe_gamma_beta(clip_tok_batch)

        hist["enc_γμ"].append([m for m,_ in enc_g]); hist["enc_γσ"].append([s for _,s in enc_g])
        hist["enc_βμ"].append([m for m,_ in enc_b]); hist["enc_βσ"].append([s for _,s in enc_b])
        hist["dec_γμ"].append([m for m,_ in dec_g]); hist["dec_γσ"].append([s for _,s in dec_g])
        hist["dec_βμ"].append([m for m,_ in dec_b]); hist["dec_βσ"].append([s for _,s in dec_b])

        # Best/patience/unfreeze + save
        improved = (val_loss < best_val - min_delta)
        if improved:
            best_val  = val_loss
            patience  = 0
        else:
            patience += 1
            if patience == 8 and not extra_clip_unfrozen:
                for blk in model_to.film.clip.transformer.resblocks[-4:]:
                    for p in blk.parameters():
                        p.requires_grad_(True)
                extra_clip_unfrozen = True
                print("[INFO] last-4 CLIP blocks unfrozen")

        save_checkpoint(
            ep,
            model=model_to,
            opt=opt,
            scheduler=scheduler,
            best_val=best_val,
            patience=patience,
            extra_unfrozen=extra_clip_unfrozen,
            hist_obj=hist,
            loss_hist_obj=loss_history,
            keep_history=False,
            save_best=improved
        )

        scheduler.step(val_loss)

        # --- Early stopping in single-GPU fallback ---
        if patience >= PATIENCE_MAX:
            print(f"[EarlyStop] No val improvement for {patience} epochs "
                  f"(≥ PATIENCE_MAX={PATIENCE_MAX}). Stopping.")
            break

# ---------------------------------------------------------------------
def main():
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    try:
        print("Launching DDP training...")
        mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)
    except Exception as e:
        import traceback
        print(">> [WARNING] DDP training failed. Falling back to single-GPU training.")
        print(">> Error traceback:")
        traceback.print_exc()

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        run_fallback_training(dev)

        # Post-fallback visualization
        test_dl = DataLoader(
            GRefDataset(test_subset, IMG_DIR, IMG_SIZE),
            batch_size=PER_RANK_BATCH, shuffle=False, num_workers=0, pin_memory=True
        )
        run_visualization_and_timing(model=model_to.to(dev).eval(), device=dev, test_dl=test_dl)

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()

    # -------- Post-run plots / evaluation in parent process --------
    # Load latest checkpoint to get hist/loss_history for plotting
    if CKPT_FILE.exists():
        state = torch.load(CKPT_FILE, map_location=device)
        try:
            hist = state.get("hist", hist)
            loss_history = state.get("loss_history", loss_history)
            model_to.load_state_dict(state["model"])
        except Exception:
            pass

    epochs = range(1, len(loss_history["train"])+1)

    if len(epochs) > 0:
        # 1) Loss Curves
        plt.figure(figsize=(6,4))
        plt.plot(epochs, loss_history["train"],   label="train")
        plt.plot(epochs, loss_history["combo"],   label="combo (train)")
        plt.plot(epochs, loss_history["val"],     label="val")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title("Loss curves")
        plt.tight_layout()
        os.makedirs("artifacts/plots", exist_ok=True)
        out_path = os.path.join("artifacts/plots", f"loss_curve_ep{len(epochs)}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()

        # 1b) Save a separate plot: Shuffle vs Val at probe epochs (1,6,11,…)
        probe_epochs = [i+1 for i, v in enumerate(loss_history["shuffle"]) if (not math.isnan(v))]
        if len(probe_epochs) > 0:
            y_shuffle = [loss_history["shuffle"][e-1] for e in probe_epochs]
            y_val     = [loss_history["val"][e-1]     for e in probe_epochs]
            plt.figure(figsize=(6,4))
            plt.plot(probe_epochs, y_shuffle, marker="o", label="caption shuffle")
            plt.plot(probe_epochs, y_val,     marker="o", label="val")
            plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title("Shuffle vs Val (every 5 epochs)")
            os.makedirs("artifacts/plots", exist_ok=True)
            plt.tight_layout(); plt.savefig("artifacts/plots/shuffle_vs_val.png", dpi=150); plt.close()

        # 2) γ / β evolution
        for s in range(4):
            γμ = [row[s] for row in hist.get("enc_γμ", [])] if len(hist.get("enc_γμ", [])) else []
            γσ = [row[s] for row in hist.get("enc_γσ", [])] if len(hist.get("enc_γσ", [])) else []
            βμ = [row[s] for row in hist.get("enc_βμ", [])] if len(hist.get("enc_βμ", [])) else []
            βσ = [row[s] for row in hist.get("enc_βσ", [])] if len(hist.get("enc_βσ", [])) else []
            if γμ:
                plt.figure(figsize=(5,2))
                plt.errorbar(epochs, γμ, yerr=γσ, label="γ", capsize=3)
                plt.errorbar(epochs, βμ, yerr=βσ, label="β", capsize=3)
                plt.axhline(1); plt.axhline(0)
                plt.title(f"Encoder stage {s} γ/β"); plt.tight_layout()
                out_path = os.path.join("artifacts/plots", f"encoder_stage_{s}_ep{len(epochs)}.png")
                plt.savefig(out_path, dpi=150)
                plt.close()

        for s in range(3):
            γμ = [row[s] for row in hist.get("dec_γμ", [])] if len(hist.get("dec_γμ", [])) else []
            γσ = [row[s] for row in hist.get("dec_γσ", [])] if len(hist.get("dec_γσ", [])) else []
            βμ = [row[s] for row in hist.get("dec_βμ", [])] if len(hist.get("dec_βμ", [])) else []
            βσ = [row[s] for row in hist.get("dec_βσ", [])] if len(hist.get("dec_βσ", [])) else []
            if γμ:
                plt.figure(figsize=(5,2))
                plt.errorbar(epochs, γμ, yerr=γσ, label="γ", capsize=3)
                plt.errorbar(epochs, βμ, yerr=βσ, label="β", capsize=3)
                plt.axhline(1); plt.axhline(0)
                plt.title(f"Decoder gate {s} γ/β"); plt.tight_layout()
                os.makedirs("artifacts/plots", exist_ok=True)
                out_path = os.path.join("artifacts/plots", f"decoder_gate_{s}_ep{len(epochs)}.png")
                plt.savefig(out_path, dpi=150)
                plt.close()

        if "log_gain" in hist and len(hist["log_gain"]):
            plt.figure(figsize=(6,3))
            plt.plot(epochs, [math.exp(lg) for lg in hist["log_gain"]], label="global gain")
            if len(hist.get("xattn_H", [])):
                plt.plot(epochs, hist["xattn_H"], label="X-Attn entropy ↓")
            plt.legend(); plt.xlabel("epoch"); plt.tight_layout()
            os.makedirs("artifacts/plots", exist_ok=True)
            out_path = os.path.join("artifacts/plots", f"log_gain_curve_ep{len(epochs)}.png")
            plt.savefig(out_path, dpi=150)
            print(f"[plot] saved {out_path}")
            plt.close()

    # Optional final quick eval visualization
    if CKPT_FILE.exists():
        dev_eval = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_to.to(dev_eval).eval()
        test_dl_eval = DataLoader(
            GRefDataset(test_subset, IMG_DIR, IMG_SIZE),
            batch_size=PER_RANK_BATCH, shuffle=False, num_workers=0, pin_memory=True
        )
        run_visualization_and_timing(model=model_to, device=dev_eval, test_dl=test_dl_eval)
        print("Post-run evaluation visualizations complete.")
