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
sys.path.insert(0, "/scratch/zt1/project/msml612/user/noberoi1/my_site_packages")

# Project imports
from FFU import UTransformer, coco_name   # (we define GRefDataset locally)

# Tokenizers
from transformers import AutoTokenizer

# CLIP (local installation)
sys.path.insert(0, '/scratch/zt1/project/msml612/user/noberoi1/open_clip_package')
import open_clip

# ---------------------------------------------------------------------
# Device (for single-GPU fallback and parent-process eval)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# OpenCLIP model/tokenizer (tokenizer used in probes; model kept for consistency)
clip_model, _, _ = open_clip.create_model_and_transforms(
    'ViT-B-32',
    pretrained='/scratch/zt1/project/msml612/user/noberoi1/open_clip/models--laion--CLIP-ViT-B-32-laion2B-s34B-b79K/open_clip_pytorch_model.bin',
    cache_dir='/scratch/zt1/project/msml612/user/noberoi1/open_clip'
)
clip_model = clip_model.eval().to(device)
clip_tok = open_clip.get_tokenizer(
    'ViT-B-32',  # match the model arch (no slash)
    cache_dir='/scratch/zt1/project/msml612/user/noberoi1/open_clip'
)

# HuggingFace tokenizer (for your dataset captions to ids)
tok = AutoTokenizer.from_pretrained(
    "bert-base-uncased",
    cache_dir="/scratch/zt1/project/msml612/user/noberoi1/hf_cache"
)

# Paths
CKPT_DIR   = Path("/scratch/zt1/project/msml612/user/noberoi1/checkpoints")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_FILE  = CKPT_DIR / "latest.pt"

ROOT       = '/scratch/zt1/project/msml612/user/noberoi1/datasets'
GREF_JSON  = f'{ROOT}/grefs(unc).json'
INST_JSON  = f'{ROOT}/instances.json'
IMG_DIR    = f'{ROOT}/coco/train2014/gref_images'

# =====================================================================
# Tuned / latest hyper-parameters & knobs
# =====================================================================
IMG_SIZE   = 512
MAX_TOK    = 20

# Sampling / splits
ROWS_TOTAL   = 10_000
TRAIN_RATIO  = 0.9

# Batch & accumulation
PER_RANK_BATCH = 16          # per-GPU mini-batch
ACC_STEPS      = 2

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
EPOCHS        = 5
PATIENCE_MAX  = 15
min_delta     = 1e-4

# λ-div schedule
def lambda_div(ep: int, start: float = 0.0, full: float = 1.0, warmup: int = DIV_WARMUP_EPOCHS):
    return min(1.0, ep / warmup) * full

# Logging CSV
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_id = uuid.uuid4().hex[:8]
BASE_PATH = "/scratch/zt1/project/msml612/user/noberoi1/"
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
                "train_loss","val_loss","lambda_div",
                "train_time","val_time","peak_gpu_mem_mb"
            ])

def log_metrics(epoch, log_gain, xattn_H,
                enc_γμ, enc_γσ, enc_βμ, enc_βσ,
                dec_γμ, dec_γσ, dec_βμ, dec_βσ,
                train_loss, val_loss, lambda_div,
                train_time, val_time, peak_gpu_mem_mb):
    with open(LOG_CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch, log_gain, xattn_H,
            enc_γμ, enc_γσ, enc_βμ, enc_βσ,
            dec_γμ, dec_γσ, dec_βμ, dec_βσ,
            train_loss, val_loss, lambda_div,
            train_time, val_time, peak_gpu_mem_mb
        ])

# =====================================================================
# Dataset prep (and local Dataset class with in-scope ann_ids_to_mask)
# =====================================================================
from pycocotools.coco import COCO
coco = COCO(INST_JSON)
VOCAB  = len(tok)

def encode_text(t: str):
    return tok(t, padding="max_length", truncation=True,
               max_length=MAX_TOK, return_tensors="pt").input_ids[0]

def encode_text_clip(txt, ctx_len: int = 77) -> torch.LongTensor:
    """txt: str or list[str] → (B,77) long tensor on CPU."""
    if isinstance(txt, str):
        toks = clip_tok([txt], context_length=ctx_len)
        return toks[0]
    else:
        return clip_tok(txt, context_length=ctx_len)

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
        return img, s["clip_ids"], m, s["text"], s["img_id"]

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
subset_refs  = train_entries[:ROWS_TOTAL]
split_idx    = int(len(subset_refs) * TRAIN_RATIO)
train_subset = subset_refs[:split_idx]
test_seed    = subset_refs[split_idx:]

train_vocab = set()
for e in train_subset:
    for s in e["sentences"]:
        train_vocab |= set(tok.tokenize(s["sent"].lower()))

def caption_in_vocab(entry, vocab):
    return all(
        set(tok.tokenize(s["sent"].lower())).issubset(vocab)
        for s in entry["sentences"]
    )

test_subset  = [e for e in test_seed if caption_in_vocab(e, train_vocab)]
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
loss_history = {"train": [], "val": [], "shuffle": []}

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
        for img, ids, gt, cap, img_id in test_dl:
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

    def overlay_mask(mask, rgb):
        o = rgb.copy()
        o[mask == 1] = (0.6 * o[mask == 1] + np.array([255, 0, 0]) * 0.4).astype(np.uint8)
        return o

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
            plt.show()
            os.makedirs("artifacts/vis", exist_ok=True)
            out_path = os.path.join("artifacts/vis", f"{iid}_cap{len(preds[iid])-1}.png")
            plt.savefig(out_path, dpi=150)
            # print(f"[viz] saved {out_path}")
            plt.close()

# ---------------------------------------------------------------------
# DDP worker
def main_worker(rank, world_size):
    try:
        print(f"[Rank {rank}] Starting Process")
        setup_ddp(rank, world_size)
        dev = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(rank)
        torch.manual_seed(42); np.random.seed(42); random.seed(42)

        # Model to device and wrap DDP
        model_to.to(dev)
        model_ddp = DDP(model_to, device_ids=[rank], output_device=rank, find_unused_parameters=True)

        # Build distributed data
        train_sampler = DistributedSampler(train_subset, num_replicas=world_size, rank=rank, shuffle=True)
        test_sampler  = DistributedSampler(test_subset,  num_replicas=world_size, rank=rank, shuffle=False)

        train_dl = DataLoader(
            GRefDataset(train_subset, IMG_DIR, IMG_SIZE),
            batch_size=PER_RANK_BATCH,
            sampler=train_sampler,
            num_workers=0,
            pin_memory=True
        )
        test_dl = DataLoader(
            GRefDataset(test_subset, IMG_DIR, IMG_SIZE),
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
        dist.broadcast(start_ep_t, src=0)
        dist.broadcast(best_val_t, src=0)
        dist.broadcast(patience_t, src=0)
        dist.broadcast(unfrozen_t, src=0)
        start_epoch = int(start_ep_t.item())
        best_val    = float(best_val_t.item())
        patience    = int(patience_t.item())
        extra_clip_unfrozen = bool(unfrozen_t.item())

        # Initial freeze policy (text tower frozen; last 2 blocks unfrozen)
        for p in model_ddp.module.film.clip.parameters():
            p.requires_grad_(False)
        for blk in model_ddp.module.film.clip.transformer.resblocks[-2:]:
            for p in blk.parameters():
                p.requires_grad_(True)

        # Warmup: text-only cosine sim, backbone frozen; set vision LR=0
        freeze_backbone(model_ddp.module, True)
        for p in model_ddp.module.film.parameters():
            p.requires_grad_(True)

        vis_group, text_group, gain_group = opt.param_groups
        orig_lr_vis, vis_group["lr"] = vis_group["lr"], 0.0

        if rank == 0: init_log_csv()

        WARMUP_EPOCHS = 2
        for warm_ep in range(WARMUP_EPOCHS):
            model_ddp.train()
            train_sampler.set_epoch(warm_ep)
            cum_loss = n = 0
            pbar_w = tqdm(train_dl, desc=f"[Rank {rank}] warm-up {warm_ep+1}/2", leave=False)
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
            if rank == 0:
                print(f"  Warm-Up Epoch {warm_ep+1}: Loss {cum_loss/n:.4f}")

        vis_group["lr"] = orig_lr_vis
        if rank == 0:
            print(f"Warm-Up Done: Backbone frozen for next {FREEZE_EPOCHS} epochs")

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

            # Probe freeze/thaw
            if ep == TEXT_PROBE_FREEZE_EPOCH:
                for n, p in model_ddp.named_parameters():
                    if ('film' in n) or ('embed' in n):
                        p.requires_grad_(False)
                if rank == 0: print(">> PROBE: Text Branch **FROZEN**")

            if ep == TEXT_PROBE_THAW_EPOCH:
                for n, p in model_ddp.named_parameters():
                    if ('film' in n) or ('embed' in n):
                        p.requires_grad_(True)
                if rank == 0: print(">> PROBE: Text Branch **UNFROZEN**")

            # ------------------ TRAIN ------------------
            model_ddp.train()
            opt.zero_grad(set_to_none=True)
            run, run_main, tot = 0., 0., 0
            pbar = tqdm(train_dl, desc=f"[Rank {rank}] Ep{ep}", leave=False)
            for step, (img, clip_ids, gt, _, img_id) in enumerate(pbar, 1):
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
                for img, clip_ids, gt, _, img_id in test_dl:
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

            local_max_mem = torch.tensor(torch.cuda.max_memory_allocated(dev), device=dev)
            dist.all_reduce(local_max_mem, op=dist.ReduceOp.MAX)
            max_mem_mb = local_max_mem.item() / 1024**2

            # -------------- Rank 0 logging/metrics --------------
            if rank == 0:
                try:
                    run_visualization_and_timing(model=model_ddp.module, device=dev, test_dl=test_dl)
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
                loss_history["shuffle"].append(float('nan'))  # not computed in DDP loop

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
                    train_loss=train_loss, val_loss=val_loss,
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
                    keep_history=False,
                    save_best=improved
                )

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
    orig_lr_vis,  vis_group["lr"]  = vis_group["lr"],  0.0

    print("Running 2-epoch cosine-similarity warm-up …")
    for warm_ep in range(2):
        cum_loss = n = 0
        pbar_w = tqdm(train_dl, desc=f"warm-up {warm_ep+1}/2", leave=False)
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
        print(f"  warm-up epoch {warm_ep+1}:  loss {cum_loss/n:.4f}")

    vis_group["lr"]  = orig_lr_vis
    print(f"Warm-up Done — Backbone will stay frozen for the next {FREEZE_EPOCHS} epochs\n")

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

        # Probe freeze/thaw
        if ep == TEXT_PROBE_FREEZE_EPOCH:
            for n, p in model_to.named_parameters():
                if ('film' in n) or ('embed' in n): p.requires_grad_(False)
            print(">> PROBE: text branch **FROZEN**")
        if ep == TEXT_PROBE_THAW_EPOCH:
            for n, p in model_to.named_parameters():
                if ('film' in n) or ('embed' in n): p.requires_grad_(True)
            print(">> PROBE: text branch **UNFROZEN**")

        # TRAIN
        model_to.train(); run, run_main, tot = 0., 0., 0
        opt.zero_grad(set_to_none=True)
        pbar = tqdm(train_dl, desc=f"Ep{ep}", leave=False)
        for step, (img, clip_ids, gt, _, img_id) in enumerate(pbar, 1):
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
            for img, clip_ids, gt, _, img_id in test_dl:
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

# ---------------------------------------------------------------------
def main():
    world_size = 4
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
        plt.plot(epochs, loss_history["val"],     label="val")
        plt.plot(epochs, loss_history["shuffle"], label="caption shuffle")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title("Loss curves")
        plt.tight_layout(); plt.show()

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
                plt.title(f"Encoder stage {s} γ/β"); plt.tight_layout(); plt.show()

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
                plt.title(f"Decoder gate {s} γ/β"); plt.tight_layout(); plt.show()

        if "log_gain" in hist and len(hist["log_gain"]):
            plt.figure(figsize=(6,3))
            plt.plot(epochs, [math.exp(lg) for lg in hist["log_gain"]], label="global gain")
            if len(hist.get("xattn_H", [])):
                plt.plot(epochs, hist["xattn_H"], label="X-Attn entropy ↓")
            plt.legend(); plt.xlabel("epoch"); plt.tight_layout(); plt.show()

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