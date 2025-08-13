import regex
from FFU import UTransformer, GRefDataset, coco_name
import sys
sys.path.insert(0, "/scratch/zt1/project/msml612/user/noberoi1/my_site_packages")
import torch

from datetime import datetime
import uuid
from itertools import chain
import time, torchvision.transforms.functional as TF

from pathlib import Path
import json, os, random, math
from pycocotools.coco import COCO
import torch.nn.functional as F
from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch import amp
import torchvision.transforms.functional as TF
from tqdm.auto import tqdm
from collections import defaultdict as _dd
import matplotlib.pyplot as plt
import numpy as np
import csv

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DistributedSampler
import torch.multiprocessing as mp
import traceback

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

sys.path.insert(0, '/scratch/zt1/project/msml612/user/noberoi1/open_clip_package')
import open_clip
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _, _ = open_clip.create_model_and_transforms(
    'ViT-B-32',
    pretrained='/scratch/zt1/project/msml612/user/noberoi1/open_clip/models--laion--CLIP-ViT-B-32-laion2B-s34B-b79K/open_clip_pytorch_model.bin',
    cache_dir='/scratch/zt1/project/msml612/user/noberoi1/open_clip'
)
clip_model = clip_model.eval().to(device)
clip_tok = open_clip.get_tokenizer(
    'ViT-B/32',
    cache_dir='/scratch/zt1/project/msml612/user/noberoi1/open_clip'
)

def setup_ddp(rank, world_size):
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()


sys.path.insert(0, "/scratch/zt1/project/msml612/user/noberoi1/my_site_packages")
from transformers import AutoTokenizer
# Load tokenizer from hf_cache
tok = AutoTokenizer.from_pretrained(
    "bert-base-uncased",
    cache_dir="/scratch/zt1/project/msml612/user/noberoi1/hf_cache"
)

CKPT_DIR   = Path("/scratch/zt1/project/msml612/user/noberoi1/checkpoints") # change as needed
CKPT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_FILE  = CKPT_DIR / "latest.pt"

ROOT       = '/scratch/zt1/project/msml612/user/noberoi1/datasets' # change as needed
GREF_JSON  = f'{ROOT}/grefs(unc).json'
INST_JSON  = f'{ROOT}/instances.json'
IMG_DIR    = f'{ROOT}/coco/train2014/gref_images' # change as needed

IMG_SIZE   = 512        # crop fed into U‑Transformer
MAX_TOK    = 20         # caption trunc / pad length

# Probe Hyper‑Parameters
TEXT_PROBE_FREEZE_EPOCH = 6      # freeze FiLM+embed here
TEXT_PROBE_THAW_EPOCH   = 9      # un‑freeze here
PROBE_BATCH             = 8      # samples for γ/β stats
SHUFFLE_BATCHES         = 2      # how many val batches to test


# Storage for Loss and History
hist = _dd(list)       # epoch → list of γσ, βσ
loss_history = {"train": [], "val": [], "shuffle": []}



# Tiny Utility (used each epoch)
def _probe_gamma_beta(clip_tok_batch: torch.Tensor):
    """
    Returns
        g_enc, b_enc : list[(μ,σ)]  (len 4)
        g_dec, b_dec : list[(μ,σ)]  (len 3)
    """
    with torch.no_grad():
        gE, bE, gD, bD = model_to.film(clip_tok_batch.to(device))

    enc_stats = ([(g.mean().item(), g.std().item()) for g in gE],
                 [(b.mean().item(), b.std().item()) for b in bE])
    dec_stats = ([(g.mean().item(), g.std().item()) for g in gD],
                 [(b.mean().item(), b.std().item()) for b in bD])
    return enc_stats, dec_stats


# Parse gRefCOCO to Build Full Train / Test Lists
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

coco = COCO(INST_JSON)
# PAD_ID = tok.pad_token_id
VOCAB  = len(tok)

def ann_ids_to_mask(ann_ids, wh):
    """Return the OR‑union of all annotation masks."""
    W, H = wh                  # wh comes as (W , H)
    m = np.zeros((H, W), np.uint8)   # <- correct rectangle
    for aid in ann_ids:
        if aid in coco.anns:
            m |= coco.annToMask(coco.anns[aid]).astype(np.uint8)
    return m

def encode_text(t:str):
    return tok(t, padding="max_length", truncation=True,
               max_length=MAX_TOK, return_tensors="pt").input_ids[0]

model_to = UTransformer(VOCAB, 3, 1, base_channels=64).to(device)

# Dual‑LR Optimiser
def is_text_param(n): return ('film' in n) or ('embed' in n)

LR_IMG, LR_TEXT, LR_GAIN = 1e-4, 5e-4, 5e-3 # ADJUST

vis_params   = []
text_params  = []                      # FiLM + token‑side blocks + xAttn
gain_param   = [model_to.film_gain_log]

for n, p in model_to.named_parameters():

    if p is model_to.film_gain_log:          # singleton -> own group
        continue

    if ("film." in n) or ("embed" in n) or ("xattn" in n):
        text_params.append(p)                # ← add xattn here
    else:
        vis_params.append(p)

opt = torch.optim.AdamW(
    [
        {"params": vis_params,  "lr": LR_IMG},
        {"params": text_params, "lr": LR_TEXT},
        {"params": gain_param,  "lr": LR_GAIN},
    ],
    weight_decay=1e-2, # ADJUST
)

# Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=0.5, patience=3, min_lr=1e-6) # ADJUST

start_epoch            = 1
best_val               = float("inf")
save_best              = False # EDITED
patience               = 0
extra_clip_unfrozen    = False        # will flip later

if CKPT_FILE.exists():
    print(f"[resume] loading {CKPT_FILE.name}")
    state            = torch.load(CKPT_FILE, map_location=device)

    model_to.load_state_dict(state["model"])
    opt.load_state_dict(state["opt"])
    scheduler.load_state_dict(state["sched"])

    start_epoch         = state["epoch"] + 1
    best_val            = state["best_val"]
    save_best              = False # EDITED
    patience            = state["patience"]
    extra_clip_unfrozen = state["extra_unfrozen"]

    hist          = state["hist"]          # <-- histories back in memory
    loss_history  = state["loss_history"]

    # Un-freeze extra blocks if needed
    if extra_clip_unfrozen:
        for blk in model_to.film.clip.transformer.resblocks[-4:]:
            for p in blk.parameters():
                p.requires_grad_(True)

    print(f"↪︎ resumed from epoch {state['epoch']}  (best val = {best_val:.4f})")


# ─ Freeze entire CLIP text tower first
for p in model_to.film.clip.parameters():
    p.requires_grad_(False)

# ─ Unfreeze last two transformer blocks for fine‑tuning
for blk in model_to.film.clip.transformer.resblocks[-2:]:
    for p in blk.parameters():
        p.requires_grad_(True)


# A) Quick Global‑Avg Pool that Exposes the Deepest Vision Feature
def backbone_pool(model: UTransformer, x: torch.Tensor) -> torch.Tensor:
    """
    Forward only the encoder, return a (B, C) pooled feature.
    """
    with torch.no_grad():
        # reuse the encoder blocks you already have
        x0 = model.inc(x)
        x0 = model.patch_embed(x0)
        x1 = model.down1(x0)
        x2 = model.down2(x1)
        x3 = model.down3(x2)          # deepest conv feature (B, C3, H/32, W/32)
        feat = x3.mean(dim=(2, 3))    # global‑avg pool → (B, C3)
    return feat                        # **no grad**



# b) A Tiny Wrapper that Converts a List[str] → CLIP tokens tensor
def encode_text_clip(txt, ctx_len: int = 77) -> torch.LongTensor:
    """
    txt : str  or  list[str]
    out : (B,77) LongTensor on **CPU**
    """
    if isinstance(txt, str):
        toks = clip_tok([txt], context_length=ctx_len)   # (1,77)
        return toks[0]                                   # (77,)
    else:                                                # list[str]
        return clip_tok(txt, context_length=ctx_len)     # (B,77)



#  Small‑Subset Experiment: 10,000 Reference Rows
ROWS_TOTAL   = 10000          # <- hard cap
TRAIN_RATIO  = 0.9            # 80 % train, 20 % test
BATCH_SIZE = 32
ACC_STEPS  = 4

# Shuffle Full Reference Lists once for Reproducibility
random.shuffle(train_entries)
random.shuffle(test_entries)

# 1) Take the first ROWS_TOTAL refs from the *training* side
subset_refs = train_entries[:ROWS_TOTAL]            # list of dicts

# 2) Split them 80 / 20
split_idx     = int(len(subset_refs) * TRAIN_RATIO)
train_subset  = subset_refs[:split_idx]
test_seed     = subset_refs[split_idx:]             # provisional test

# 3) Build train‑vocabulary
train_vocab = set()
for e in train_subset:
    for s in e["sentences"]:
        train_vocab |= set(tok.tokenize(s["sent"].lower()))

# 4) Ensure test captions use only that vocabulary
def caption_in_vocab(entry, vocab):
    return all(
        set(tok.tokenize(s["sent"].lower())).issubset(vocab)
        for s in entry["sentences"]
    )

test_subset  = [e for e in test_seed if caption_in_vocab(e, train_vocab)]

print(f"Final train refs = {len(train_subset):,}  "
      f"(= {len(train_subset)/ROWS_TOTAL:.0%} of {ROWS_TOTAL})")
print(f"Final test  refs = {len(test_subset):,}  "
      f"(= {len(test_subset)/ROWS_TOTAL:.0%} of {ROWS_TOTAL})")

## Regulariser & Schedule Adjustments
REG_W_FiLM   = 2.5e-3          #   ↘  (was 5e-3)
REG_W_AREA   = 0.10            #   ↘  (was 0.20)
REG_W_COM    = 0.05            #   ↘  (was 0.10)
TV_W         = 0.05            # unchanged

# Div-Weight Ramps a Little Faster (full value by epoch-5 instead of -10)
def lambda_div(ep, start=0.0, full=1.0, warmup=5):
    return min(1.0, ep / warmup) * full



# Loss Functions
def dice_loss(logits, gt, eps=1.):
    p = torch.sigmoid(logits.float()); gt = gt.float()
    inter  = (p*gt).sum((2,3))
    union  = p.sum((2,3)) + gt.sum((2,3))
    return 1. - (2*inter+eps)/(union+eps)          # shape (B,)

def miou_loss(logits, gt, eps=1e-6):
    p = torch.sigmoid(logits.float()); gt = gt.float()
    inter  = (p*gt).sum((2,3))
    union  = (p+gt-p*gt).sum((2,3))
    return 1. - (inter+eps)/(union+eps)            # shape (B,)

def focal_loss(logits, gt, alpha=0.25, gamma=2.0, eps=1e-6):
    p  = torch.sigmoid(logits.float()).clamp(eps, 1-eps)
    ce = -(alpha*gt*torch.log(p) + (1-alpha)*(1-gt)*torch.log(1-p))
    mod = (1-p)**gamma
    return (mod * ce).mean(dim=(2,3))              # shape (B,)

def combo_loss(logits, gt, α=0.45, β=0.45, γ=0.10): # ADJUST RATIO
    d = dice_loss (logits, gt)    # (B,)
    i = miou_loss (logits, gt)    # (B,)
    f = focal_loss(logits, gt)    # (B,)
    return (α*d + β*i + γ*f).mean()   # **scalar**

def film_reg(gammas, betas, w=REG_W_FiLM):
    # L1 penalty keeps gradients alive
    reg = 0.
    for g, b in zip(gammas, betas):
        reg += ((g - 1).abs() + b.abs()).mean()
    return w * reg



# Contrastive Text–Vision Loss
def contrastive_tv_loss(vis_feat, txt_feat, T=0.07):
    """
    vis_feat : (B,C) pooled encoder feature  (no grad‑stopping)
    txt_feat : (B,512) CLIP text feature     (fwd pass already done)
    """
    vis = F.normalize(vis_feat, dim=-1)
    txt = F.normalize(txt_feat, dim=-1)
    # cosine‑sim on the *matching* pairs (diagonal of the B×B matrix)
    return 1. - (vis * txt).sum(-1).mean()



# Mask‑Area Prior (BCE on area)
def mask_area_loss(pred, gt):
    """
    pred, gt : (B,1,H,W) after *sigmoid*  (keep them float in 0‑1)
    """
    B,_,H,W = pred.shape
    area_p = pred.mean((2,3))             # (B,)  ∈ (0,1)
    area_g = gt.mean((2,3))
    return F.binary_cross_entropy(area_p, area_g)

# Centre‑of‑Mass Alignment
def com_alignment_loss(pred, gt, eps=1e-6):
    """
    Fast ∑( |c_pred − c_gt|² )  over the batch.
    """
    B,_,H,W = pred.shape
    # add a tiny ε so argmax won’t explode on empty masks
    w = pred.clamp_min(eps)
    # coordinates 0‥1   (two 1×H×W buffers reused for entire batch)
    yy, xx = torch.meshgrid(
        torch.linspace(0,1,H,device=pred.device),
        torch.linspace(0,1,W,device=pred.device),
        indexing='ij')
    xx = xx[None,None]; yy = yy[None,None]        # (1,1,H,W)
    cx_p = (w*xx).sum((2,3)) / w.sum((2,3))       # (B,1)
    cy_p = (w*yy).sum((2,3)) / w.sum((2,3))
    cx_g = (gt*xx).sum((2,3)) / gt.sum((2,3))
    cy_g = (gt*yy).sum((2,3)) / gt.sum((2,3))
    return ((cx_p-cx_g)**2 + (cy_p-cy_g)**2).mean()

@torch.no_grad()
def clamp_film_params(model,
                      enc_g=(0.5, 2.0), dec_g=(0.5, 2.5),
                      b_lim=(-1.0, 1.0)):
    """
    Encoders stay conservative, decoders get wider head-room.
    """
    for n, p in model.named_parameters():
        if ".film." not in n or p.ndim != 4:
            continue
        if "gamma" in n:
            if "dec" in n:   p.clamp_(*dec_g)
            else:            p.clamp_(*enc_g)
        else:
            p.clamp_(*b_lim)

best_val = float('inf')
patience = 0
extra_clip_unfrozen = False
PATIENCE_MAX = 10          # or any value you like

def save_checkpoint(ep: int, *, keep_history: bool = False, save_best = False): # EDITED
    state = {
        "epoch"          : ep,
        "model"          : model_to.state_dict(),
        "opt"            : opt.state_dict(),
        "sched"          : scheduler.state_dict(),
        "best_val"       : best_val,
        "patience"       : patience,
        "extra_unfrozen" : extra_clip_unfrozen,
        "hist"           : hist,           # <─ NEW
        "loss_history"   : loss_history,   # <─ NEW
    }

    torch.save(state, CKPT_FILE)                    # always overwrites
    if save_best:
        torch.save(state, CKPT_DIR / "best.pt") # EDITED
    if keep_history:
        torch.save(state, CKPT_DIR / f"epoch_{ep:03d}.pt")
        
def freeze_backbone(m, flag=True):
            for name, sub in m.named_children():
                if name != "film":
                    for p in sub.parameters(): p.requires_grad_(not flag)


# --------------------------------- make sure this works ---------------------------
def flatten(x):
    return list(chain.from_iterable(x)) if isinstance(x[0], (list, tuple)) else x

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
                "epoch",
                "log_gain",
                "xattn_H",
                "enc_γμ",
                "enc_γσ",
                "enc_βμ",
                "enc_βσ",
                "dec_γμ",
                "dec_γσ",
                "dec_βμ",
                "dec_βσ",
                "train_loss",
                "val_loss",
                "lambda_div",
                "train_time",
                "val_time",
                "peak_gpu_mem_mb"
            ])

def log_metrics(
    epoch,
    log_gain,
    xattn_H,
    enc_γμ,
    enc_γσ,
    enc_βμ,
    enc_βσ,
    dec_γμ,
    dec_γσ,
    dec_βμ,
    dec_βσ,
    train_loss,
    val_loss,
    lambda_div,
    train_time,
    val_time,
    peak_gpu_mem_mb
):

    with open(LOG_CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            log_gain,
            xattn_H,
            enc_γμ,
            enc_γσ,
            enc_βμ,
            enc_βσ,
            dec_γμ,
            dec_γσ,
            dec_βμ,
            dec_βσ,
            train_loss,
            val_loss,
            lambda_div,
            train_time,
            val_time,
            peak_gpu_mem_mb
        ])
# ----------------------------------------------------------------------------------------

## Segmentation Visualisation + Average Inference Timing
def run_visualization_and_timing(model, device, test_dl):

    model.eval()
    preds, gts, caps, bases = {}, {}, {}, {}
    inf_times = []

    with torch.no_grad():
        for img, ids, gt, cap, img_id in test_dl:
            img, ids = img.to(device), ids.to(device)

            # Forward‑Pass Timing
            if device.type == "cuda":
                torch.cuda.synchronize()
                t0 = torch.cuda.Event(enable_timing=True)
                t1 = torch.cuda.Event(enable_timing=True)
                t0.record()
                with amp.autocast('cuda', dtype=torch.float16):
                    p_batch = model(img, ids)
                t1.record()
                torch.cuda.synchronize()
                batch_ms = t0.elapsed_time(t1)
            else:
                t0 = time.perf_counter()
                p_batch = model(img, ids)
                batch_ms = (time.perf_counter() - t0) * 1000

            inf_times.extend([batch_ms / img.size(0)] * img.size(0))
            prob = p_batch.cpu()

            for b in range(img.size(0)):
                iid = int(img_id[b])
                preds.setdefault(iid, []).append(prob[b, 0].numpy())
                gts.setdefault(iid, []).append(gt[b, 0].numpy())
                caps.setdefault(iid, []).append(cap[b])

                if iid not in bases:
                    rgb = (img[b].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    bases[iid] = rgb

    avg_ms = np.mean(inf_times)
    std_ms = np.std(inf_times)
    print(f"\n Average Forward‑Pass Time  : {avg_ms:6.2f} ms / image "
          f"(± {std_ms:4.2f} ms  •  {len(inf_times)} images)")

    def overlay_mask(mask, rgb):
        o = rgb.copy()
        o[mask == 1] = (0.6 * o[mask == 1] + np.array([255, 0, 0]) * 0.4).astype(np.uint8)
        return o

    for iid in preds:
        base = bases[iid]
        print(f"\n=== Test Image {iid} ===")
        for prob, gt_mask, caption in zip(preds[iid], gts[iid], caps[iid]):
            pred_bin = (prob > 0.5).astype(np.uint8)
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(overlay_mask(gt_mask, base))
            plt.axis('Off')
            plt.title("Ground Truth")

            plt.subplot(1, 2, 2)
            plt.imshow(overlay_mask(pred_bin, base))
            plt.axis('Off')
            plt.title("Prediction")

            plt.suptitle(caption, y=0.95)
            plt.tight_layout()
            plt.show()
            plt.close()

## DDP CODE
def main_worker(rank, world_size):
    try:
        print(f"[Rank {rank}] Starting Process")
        setup_ddp(rank, world_size)
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(rank)
        
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        
        if rank == 0:
            hist = {
                "log_gain": [],
                "xattn_H": [],
                "enc_γμ": [],
                "enc_γσ": [],
                "enc_βμ": [],
                "enc_βσ": [],
                "dec_γμ": [],
                "dec_γσ": [],
                "dec_βμ": [],
                "dec_βσ": [],
                "train_loss": [],
                "val_loss": [],
                "reg_term": []
            }
        else:
            hist = None

        # Move Model to Device
        model_to.to(device)
        
        # Wrap Model with DDP
        model_ddp = torch.nn.parallel.DistributedDataParallel(
            model_to, device_ids=[rank], output_device=rank, find_unused_parameters=True
        )
        
        # Create distributed samplers
        train_sampler = DistributedSampler(train_subset, num_replicas=world_size, rank=rank, shuffle=True)
        test_sampler = DistributedSampler(test_subset, num_replicas=world_size, rank=rank, shuffle=False)
        
        train_dl = DataLoader(
            GRefDataset(train_subset, IMG_DIR, IMG_SIZE),
            batch_size=BATCH_SIZE // world_size,
            sampler=train_sampler,
            num_workers=0,
            pin_memory=True
        )
        
        test_dl = DataLoader(
            GRefDataset(test_subset, IMG_DIR, IMG_SIZE),
            batch_size=BATCH_SIZE // world_size,
            sampler=test_sampler,
            num_workers=0,
            pin_memory=True
        )
        
        
        # Setup Optimizer, Scheduler on Model_DDP.parameters()
        vis_params = []
        text_params = []
        gain_param = [model_ddp.module.film_gain_log]

        for n, p in model_ddp.module.named_parameters():
            if p is model_ddp.module.film_gain_log:
                continue

            if ("film." in n) or ("embed" in n) or ("xattn" in n):
                text_params.append(p)
            else:
                vis_params.append(p)
        
        
        ### Optimizer
        opt = torch.optim.AdamW(
            [
                {"params": vis_params,  "lr": LR_IMG},
                {"params": text_params, "lr": LR_TEXT},
                {"params": gain_param,  "lr": LR_GAIN},
            ],
            weight_decay=1e-2,
        )
        
        
        ### Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        opt, mode="min", factor=0.5, patience=3, min_lr=1e-6)
        
        
        ### Adjustments
        FREEZE_EPOCHS, CLIP_NORM = 1, 1.
        LAMBDA_DIV0, EPOCHS, PATIENCE_MAX = 1.0, 23, 15
        ACC_STEPS = 1
        
        best_val = float('inf')
        patience = 0
        extra_clip_unfrozen = False
        
        freeze_backbone(model_ddp.module, True)

        best, patience = math.inf, 0
        
        # Warm-up phase remains mostly unchanged, just use model_ddp.module and device:
        for p in model_ddp.module.film.parameters():
            p.requires_grad_(True)

        vis_group, text_group, gain_group = opt.param_groups
        orig_lr_vis, vis_group["lr"] = vis_group["lr"], 0.0
        
        WARMUP_EPOCHS = 2
        for warm_ep in range(WARMUP_EPOCHS):
            model_ddp.train()
            train_sampler.set_epoch(warm_ep)
            cum_loss = n = 0
            pbar_w = tqdm(train_dl, desc=f"[Rank {rank}] warm-up {warm_ep+1}/2", leave=False)
            for img, clip_ids, *_ in pbar_w:
                img, clip_ids = img.to(device), clip_ids.to(device)
                with torch.no_grad():
                    vis_feat = backbone_pool(model_ddp.module, img)
                txt_feat = model_ddp.module.film.clip.encode_text(clip_ids)
                loss = contrastive_tv_loss(vis_feat, txt_feat)
                
                loss.backward()
                opt.step(); opt.zero_grad()
                
                cum_loss += loss.item() * img.size(0); n += img.size(0)
                pbar_w.set_postfix(loss=f"{loss.item():.3f}")

            if rank == 0:
                print(f"  Warm-Up Epoch {warm_ep+1}: Loss {cum_loss/n:.4f}")

        ## Restore Original Learning-Rates
        vis_group["lr"] = orig_lr_vis
        
        if rank == 0:
            print(f"Warm-Up Done: Backbone frozen for next {FREEZE_EPOCHS} epochs")

        # Unfreeze all parameters after warm-up
        for p in model_ddp.parameters():
            p.requires_grad_(True)

        for ep in range(start_epoch, EPOCHS + 1):
            print(f"\n[Rank {rank}] Starting Epoch {ep}/{EPOCHS}")
            train_sampler.set_epoch(ep)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)

            start_train = time.time()
            
            if ep == FREEZE_EPOCHS + 1:
                freeze_backbone(model_ddp.module, False)
                if rank == 0:
                    print(f"[Epoch {ep}] Backbone Unfrozen")

            λ_div = lambda_div(ep)
            
            if ep == TEXT_PROBE_FREEZE_EPOCH:
                for n, p in model_ddp.named_parameters():
                    if ('film' in n) or ('embed' in n):
                        p.requires_grad_(False)
                if rank == 0:
                    print(">> PROBE: Text Branch **FROZEN**")
            
            if ep == TEXT_PROBE_THAW_EPOCH:
                for n, p in model_ddp.named_parameters():
                    if ('film' in n) or ('embed' in n):
                        p.requires_grad_(True)
                if rank == 0:
                    print(">> PROBE: Text Branch **UNFROZEN**")
            
            # --- TRAIN -------
            model_ddp.train()
            opt.zero_grad(set_to_none=True)
            run, tot = 0., 0
            pbar = tqdm(train_dl, desc=f"[Rank {rank}] Ep{ep}", leave=False)
            for step, (img, clip_ids, gt, _, img_id) in enumerate(pbar, 1):
                img, clip_ids, gt = img.to(device), clip_ids.to(device), gt.to(device)
                p = model_ddp(img, clip_ids)
                loss_main = combo_loss(p, gt)
                    
                # Auxiliary 1: Contrastive TV  (very small weight)
                with torch.no_grad():
                    vis_feat = backbone_pool(model_ddp.module, img)
                    txt_feat = model_ddp.module.film.clip.encode_text(clip_ids)
                loss_tv = TV_W * contrastive_tv_loss(vis_feat, txt_feat)
                
                # Auxiliary 2: Mask‑Area Prior  (weight 0.2)
                loss_area = REG_W_AREA * mask_area_loss(p, gt)
                
                # Auxiliary 3: COM alignment       (weight 0.1)
                loss_com = REG_W_COM * com_alignment_loss(p, gt)

                # Divergence over ALL differing pairs
                idx_by_img = {}
                div_acc = pairs = 0.
                for k, iid in enumerate(img_id):
                    idx_by_img.setdefault(int(iid), []).append(k)
                for idxs in idx_by_img.values():
                    for i in range(len(idxs)):
                        for j in range(i + 1, len(idxs)):
                            i1, i2 = idxs[i], idxs[j]
                            if torch.equal(gt[i1], gt[i2]):
                                continue
                            div_acc += (p[i1] - p[i2]).abs().mean()
                            pairs += 1
                div_loss = (div_acc / pairs) if pairs else torch.tensor(0., device=device)

                with torch.no_grad():
                    gE, bE, gD, bD = model_ddp.module.film(clip_ids)  # lists of tensors

                # FiLM regulariser (unchanged)
                reg_term = film_reg(gE + gD, bE + bD, w=1e-3)
                reg_loss = loss_main + λ_div * div_loss

                ## Total Loss
                loss = loss_main + div_loss + reg_term + loss_tv + loss_area + loss_com

                loss.backward()

                if step % ACC_STEPS == 0 or step == len(train_dl):
                    torch.nn.utils.clip_grad_norm_(model_ddp.parameters(), CLIP_NORM)
                    opt.step()
                    clamp_film_params(model_ddp.module)
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
            end_train = time.time()
            train_time = end_train - start_train

            # VALIDATION
            start_val = time.time()
            model_ddp.eval()
            v_sum = torch.tensor(0., device=device) #-- different from training loop (v_sum = v_n = 0)
            v_n = torch.tensor(0., device=device)  #-- different from training loop (v_sum = v_n = 0)
            with torch.no_grad():
                for img, clip_ids, gt, _, img_id in test_dl:
                    img, clip_ids, gt = img.to(device), clip_ids.to(device), gt.to(device)
                    p_val = model_ddp(img, clip_ids)
                    v_loss = combo_loss(p_val, gt)
                    v_sum += v_loss * img.size(0)
                    v_n += img.size(0)

            # Reduce Validation Sums and Counts Across all GPUs
            dist.all_reduce(v_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(v_n, op=dist.ReduceOp.SUM)
            val_loss = (v_sum / v_n).item()
            end_val = time.time()
            val_time = end_val - start_val

            local_max_mem = torch.tensor(torch.cuda.max_memory_allocated(device), device=device)
            dist.all_reduce(local_max_mem, op=dist.ReduceOp.MAX)
                    
            # Print only on Rank 0
            if rank == 0:
                model_to_eval = model_ddp.module  # Unwrap from DDP
                
                try:
                    run_visualization_and_timing(model=model_to_eval, device=device, test_dl=test_dl)
                except Exception as e:
                    print(f"[Rank 0] Visualization Failed: {e}")
                
                max_mem_mb = local_max_mem.item() / 1024**2
                
                print(
                    f"Epoch {ep:02d} | Training Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f} | "
                    f"λ={λ_div:.2f} | Training Time: {train_time:.1f}s | Validation Time: {val_time:.1f}s | "
                    f"Peak GPU Memory across All Ranks: {max_mem_mb:.1f} MB"
                )
                
                log_metrics(
                    epoch=ep,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    lambda_div=λ_div,
                    train_time=train_time,
                    val_time=val_time,
                    peak_gpu_mem_mb=max_mem_mb,
                    log_gain=model_ddp.module.film_gain_log.item(),
                    xattn_H=np.mean(model_ddp.module.attn_entropy) if len(model_ddp.module.attn_entropy) > 0 else 0,
                )
                
                hist["log_gain"].append(model_ddp.module.film_gain_log.item())
                hist["xattn_H"].append(np.mean(model_ddp.module.attn_entropy))
                model_ddp.module.attn_entropy.clear()

                sample_caps = [train_subset[i]["sentences"][0]["sent"]
                            for i in range(min(PROBE_BATCH, len(train_subset)))]

                clip_tok_batch = encode_text_clip(sample_caps).to(device)
                (enc_g, enc_b), (dec_g, dec_b) = _probe_gamma_beta(clip_tok_batch)

                hist["enc_γμ"].append([m for m, _ in enc_g])
                hist["enc_γσ"].append([s for _, s in enc_g])
                hist["enc_βμ"].append([m for m, _ in enc_b])
                hist["enc_βσ"].append([s for _, s in enc_b])

                hist["dec_γμ"].append([m for m, _ in dec_g])
                hist["dec_γσ"].append([s for _, s in dec_g])
                hist["dec_βμ"].append([m for m, _ in dec_b])
                hist["dec_βσ"].append([s for _, s in dec_b])

                # Save checkpoint on Improved Validation Loss
                if best_val > val_loss:
                    patience = 0
                    best_val = val_loss
                    torch.save(
                        {
                            "model": model_ddp.module.state_dict(),
                            "optim": opt.state_dict(),
                            "sched": scheduler.state_dict(),
                            "epoch": ep,
                            "hist": hist,
                        },
                        "best_ckpt.pth",
                    )
                    print(f"-> best val {best_val:.4f}")
                else:
                    patience += 1

            # Scheduler step
            scheduler.step(val_loss)

    except Exception as e:
        print(f"[Rank {rank}] Exception occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_ddp()
# Training Loop – freeze, accumulation, λ‑warm‑up, sanity plots
def run_fallback_training(device):
    global best_val, patience, extra_clip_unfrozen, save_best
    print("[Fallback]: Running Single-GPU Training Loop.")
    global model_to, opt, scheduler, hist, loss_history, start_epoch
    train_dl = DataLoader(
        GRefDataset(train_subset, IMG_DIR, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,          # <- one process, no IPC
        pin_memory=True,
    )

    test_dl = DataLoader(
        GRefDataset(test_subset, IMG_DIR, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    FREEZE_EPOCHS, CLIP_NORM = 1, 1.
    LAMBDA_DIV0, EPOCHS, PATIENCE_MAX = 1.0, 23, 15

    freeze_backbone(model_to, True)

    best, patience = math.inf, 0


    # Text-Only Cosine Warm‑Up (2 epochs)
    print("Running 2‑epoch cosine‑similarity warm‑up …")

    model_to.train()                           # ensure dropout/BN in train mode
    freeze_backbone(model_to, True)            # vision frozen

    for p in model_to.film.parameters():       # FiLM + (unfrozen) text blocks
        p.requires_grad_(True)

    # handles to the three param‑groups you created earlier
    vis_group, text_group, gain_group = opt.param_groups

    # temporarily zero‑out the LRs of groups we DON’T want to move
    orig_lr_vis,  vis_group["lr"]  = vis_group["lr"],  0.0
    # orig_lr_gain, gain_group["lr"] = gain_group["lr"], 0.0

    for warm_ep in range(2):
        cum_loss = n = 0
        pbar_w = tqdm(train_dl, desc=f"warm‑up {warm_ep+1}/2", leave=False)

        for img, clip_ids, *_ in pbar_w:
            img, clip_ids = img.to(device), clip_ids.to(device)

            with torch.no_grad():                    # vision stays frozen
                vis_feat = backbone_pool(model_to, img)   # (B,C)

            txt_feat = model_to.film.clip.encode_text(clip_ids)  # grads flow here
            loss = contrastive_tv_loss(vis_feat, txt_feat)

            loss.backward()
            opt.step();  opt.zero_grad()

            cum_loss += loss.item() * img.size(0);  n += img.size(0)
            pbar_w.set_postfix(loss=f"{loss.item():.3f}")

        print(f"  warm‑up epoch {warm_ep+1}:  loss {cum_loss/n:.4f}")

    # Restore Original Learning‑Rates
    vis_group["lr"]  = orig_lr_vis
    # gain_group["lr"] = orig_lr_gain

    print(f"Warm‑up Done — Backbone will stay frozen for the next "
          f"{FREEZE_EPOCHS} epochs\n")


    # 3) Normal Training (unfrozen training)
    for p in model_to.parameters():
        p.requires_grad_(True)

    for ep in range(start_epoch, EPOCHS + 1):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            
        start_train = time.time()
        
        if ep == FREEZE_EPOCHS+1:
            freeze_backbone(model_to, False)
            print(f"[Epoch {ep}] backbone unfrozen")
        λ_div     = lambda_div(ep)     # replaces the old one-liner

        # Text‑Branch Freeze / Thaw Probe
        if ep == TEXT_PROBE_FREEZE_EPOCH:
            for n, p in model_to.named_parameters():
                if ('film' in n) or ('embed' in n):
                    p.requires_grad_(False)
            print(">> PROBE: text branch **FROZEN**")

        if ep == TEXT_PROBE_THAW_EPOCH:
            for n, p in model_to.named_parameters():
                if ('film' in n) or ('embed' in n):
                    p.requires_grad_(True)
            print(">> PROBE: text branch **UNFROZEN**")


        # TRAIN
        model_to.train(); run, tot = 0., 0
        opt.zero_grad(set_to_none=True)
        pbar = tqdm(train_dl, desc=f"Ep{ep}", leave=False)
        for step, (img, clip_ids, gt, _, img_id) in enumerate(pbar, 1):
            img, clip_ids, gt = img.to(device), clip_ids.to(device), gt.to(device)
            p = model_to(img, clip_ids)
            loss_main = combo_loss(p, gt)        # no clamp / sigmoid

            # Auxiliary 1: Contrastive TV  (very small weight)
            with torch.no_grad():
                vis_feat = backbone_pool(model_to, img)            # (B,C)
                txt_feat = model_to.film.clip.encode_text(clip_ids)
            loss_tv   = TV_W   * contrastive_tv_loss(vis_feat, txt_feat)

            # Auxiliary 2: Mask‑Area Prior  (weight 0.2)
            loss_area = REG_W_AREA * mask_area_loss(p, gt)

            # Auxiliary 3: COM alignment       (weight 0.1)
            loss_com  = REG_W_COM  * com_alignment_loss(p, gt)


            # Divergence over ALL differing pairs
            idx_by_img={}; div_acc=pairs=0.
            for k,iid in enumerate(img_id): idx_by_img.setdefault(int(iid),[]).append(k)
            for idxs in idx_by_img.values():
                for i in range(len(idxs)):
                    for j in range(i+1,len(idxs)):
                        i1,i2=idxs[i],idxs[j]
                        if torch.equal(gt[i1],gt[i2]): continue
                        div_acc += (p[i1] - p[i2]).abs().mean()   # ← using prob tensors
                        pairs+=1
            div_loss = (div_acc/pairs) if pairs else torch.tensor(0.,device=device)


            with torch.no_grad():
                gE, bE, gD, bD = model_to.film(clip_ids)  # lists of tensors

            # FiLM regulariser (unchanged)
            reg_term = film_reg(gE + gD, bE + bD, w=1e-3)
            loss = loss_main + λ_div*div_loss


            # Total Loss
            loss = loss_main + div_loss + reg_term + loss_tv + loss_area + loss_com

            loss.backward()

            if step%ACC_STEPS==0 or step==len(train_dl):
                torch.nn.utils.clip_grad_norm_(model_to.parameters(),CLIP_NORM)
                opt.step()
                clamp_film_params(model_to)
                opt.zero_grad(set_to_none=True)

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
        # scheduler.step()                       # LR decay

        # VALIDATION
        start_val = time.time()
        model_to.eval(); v_sum=v_n=0
        with torch.no_grad():
            for img, clip_ids, gt, _, img_id in test_dl:
                img, clip_ids, gt = img.to(device), clip_ids.to(device), gt.to(device)
                p_val = model_to(img, clip_ids)
                v_loss = combo_loss(p_val, gt)
                v_sum += v_loss.item()*img.size(0); v_n += img.size(0)
        val_loss = v_sum/v_n
        end_val = time.time()
        val_time = end_val - start_val

        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        else:
            max_mem_mb = 0
            
        print(
            f"Epoch {ep:02d} | Training Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f} | "
            f"λ={λ_div:.2f} | Training Time: {train_time:.1f}s | Validation Time: {val_time:.1f}s | "
            f"Peak GPU Memory: {max_mem_mb:.1f} MB")

        hist["log_gain"].append(model_to.film_gain_log.item())
        hist["xattn_H"].append( np.mean(model_to.attn_entropy) )
        model_to.attn_entropy.clear()          # reset for next epoch

        # 1)  γ / β  Statistics
        sample_caps = [train_subset[i]["sentences"][0]["sent"]
                   for i in range(min(PROBE_BATCH, len(train_subset)))]

        clip_tok_batch = encode_text_clip(sample_caps).to(device)
        (enc_g, enc_b), (dec_g, dec_b) = _probe_gamma_beta(clip_tok_batch)

        hist["enc_γμ"].append([m for m,_ in enc_g])
        hist["enc_γσ"].append([s for _,s in enc_g])
        hist["enc_βμ"].append([m for m,_ in enc_b])
        hist["enc_βσ"].append([s for _,s in enc_b])

        hist["dec_γμ"].append([m for m,_ in dec_g])
        hist["dec_γσ"].append([s for _,s in dec_g])
        hist["dec_βμ"].append([m for m,_ in dec_b])
        hist["dec_βσ"].append([s for _,s in dec_b])


        # Inline Console Print
        row = lambda lst: "  ".join(f"{m:+.3f}±{s:.3f}" for m,s in lst)
        print(f"  enc γ: {row(enc_g)}   β: {row(enc_b)}")
        print(f"  dec γ: {row(dec_g)}   β: {row(dec_b)}")

        # 2) Caption‑Shuffle Probe (one or few validation batches)
        sh_sum = sh_n = 0
        with torch.no_grad():
            for _b,(img_v, ids_v, gt_v, _, _) in enumerate(test_dl):
                img_v, ids_v, gt_v = img_v.to(device), ids_v.to(device), gt_v.to(device)
                ids_shuf = ids_v[torch.randperm(ids_v.size(0))]
                p_shuf = model_to(img_v, ids_shuf)
                sh_loss = combo_loss(p_shuf, gt_v)
                sh_sum += sh_loss.item() * img_v.size(0)
                sh_n   += img_v.size(0)
                if _b+1 >= SHUFFLE_BATCHES: break
        shuffle_loss = sh_sum / sh_n
        delta = shuffle_loss - val_loss
        print(f"  shuffle‑loss {shuffle_loss:.4f}  (Δ {delta:+.4f})")

        # History Record for Later Plotting
        loss_history["train"].append(train_loss)
        loss_history["val"].append(val_loss)
        loss_history["shuffle"].append(shuffle_loss)

        # Visual Sanity Check (every 5 epochs)
        if ep % 5 == 0:
            def overlay(mask, rgb):
                out = rgb.copy()
                out[mask == 1] = (0.6 * out[mask == 1] +
                                  np.array([255, 0, 0]) * 0.4).astype(np.uint8)
                return out

            fig, ax = plt.subplots(2, 2, figsize=(6, 6))      # 2 rows (train / test)

            for row, (dl, split_name) in enumerate([(train_dl, "train"),
                                                    (test_dl,  "test")]):

                # One Mini-Batch
                img_t, ids_t, gt_t, cap_t, _ = next(iter(dl))
                img_t, ids_t = img_t.to(device), ids_t.to(device)

                # Run the Model on the **First Sample Only**
                with torch.no_grad():
                    p = model_to(img_t[:1], ids_t[:1])          # ← already prob (B,1,H,W)
                    pred = p[0, 0].cpu().numpy()                    # remove extra sigmoid call

                # Prepare Visuals
                base = (img_t[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                gt   = gt_t[0, 0].cpu().numpy()

                # GT
                ax[row, 0].imshow(overlay(gt, base))
                ax[row, 0].set_title("GT")
                ax[row, 0].axis("off")

                # Prediction
                ax[row, 1].imshow(overlay(pred > 0.5, base))    # threshold as before
                ax[row, 1].set_title("Pred")
                ax[row, 1].axis("off")

                # Row Caption
                ax[row, 0].annotate(f"{split_name}:  {cap_t[0]}", xy=(0.5, -0.25),
                                    xycoords="axes fraction", ha="center", fontsize=8)

            plt.tight_layout()
            plt.show()

        # save_checkpoint(ep)

        min_delta   = 1e-4          # how much val must improve
        # best_val    = getattr(best_val, 'value', float('inf'))   # one‑time init

        if val_loss < best_val - min_delta:
            best_val  = val_loss
            save_best = True
            patience  = 0
        else:
            save_best = False
            patience += 1
            if patience == 8 and not extra_clip_unfrozen:
                for blk in model_to.film.clip.transformer.resblocks[-4:]:
                      for p in blk.parameters():
                              p.requires_grad_(True)
                extra_clip_unfrozen = True
                print("[INFO] last-4 CLIP blocks unfrozen")
        
        save_checkpoint(ep, keep_history = False, save_best = save_best)

        # optional: ReduceLRonPlateau
        scheduler.step(val_loss)

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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        global model_to, opt, scheduler, hist, loss_history, start_epoch
        start_epoch = 1

        hist = {
            "log_gain": [],
            "xattn_H": [],
            "enc_γμ": [],
            "enc_γσ": [],
            "enc_βμ": [],
            "enc_βσ": [],
            "dec_γμ": [],
            "dec_γσ": [],
            "dec_βμ": [],
            "dec_βσ": [],
        }
        loss_history = {
            "train": [],
            "val": [],
            "shuffle": [],
        }

        vis_params = []
        text_params = []
        gain_param = [model_to.film_gain_log]

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
            weight_decay=1e-2,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=3, min_lr=1e-6
        )

        run_fallback_training(device=device)
        test_dl = DataLoader(GRefDataset(test_subset, IMG_DIR, IMG_SIZE),
                     batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
        run_visualization_and_timing(model=model_to, device=device, test_dl=test_dl)

if __name__ == "__main__":
    main()

    epochs = range(1, len(loss_history["train"])+1)

    # 1) Loss Curves
    plt.figure(figsize=(6,4))
    plt.plot(epochs, loss_history["train"],   label="train")
    plt.plot(epochs, loss_history["val"],     label="val")
    plt.plot(epochs, loss_history["shuffle"], label="caption shuffle")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title("Loss curves")
    plt.tight_layout(); plt.show()

    # 2) γ / β evolution
    # Encoder
    for s in range(4):
        γμ = [row[s] for row in hist["enc_γμ"]];  γσ = [row[s] for row in hist["enc_γσ"]]
        βμ = [row[s] for row in hist["enc_βμ"]];  βσ = [row[s] for row in hist["enc_βσ"]]
        plt.figure(figsize=(5,2))
        plt.errorbar(epochs, γμ, yerr=γσ, label="γ", capsize=3)
        plt.errorbar(epochs, βμ, yerr=βσ, label="β", capsize=3)
        plt.axhline(1); plt.axhline(0)
        plt.title(f"Encoder stage {s} γ/β"); plt.tight_layout(); plt.show()

    # Decoder:
    for s in range(3):
        γμ = [row[s] for row in hist["dec_γμ"]];  γσ = [row[s] for row in hist["dec_γσ"]]
        βμ = [row[s] for row in hist["dec_βμ"]];  βσ = [row[s] for row in hist["dec_βσ"]]
        plt.figure(figsize=(5,2))
        plt.errorbar(epochs, γμ, yerr=γσ, label="γ", capsize=3)
        plt.errorbar(epochs, βμ, yerr=βσ, label="β", capsize=3)
        plt.axhline(1); plt.axhline(0)
        plt.title(f"Decoder gate {s} γ/β"); plt.tight_layout(); plt.show()

    plt.figure(figsize=(6,3))
    plt.plot(epochs, [math.exp(lg) for lg in hist["log_gain"]], label="global gain")
    plt.plot(epochs, hist["xattn_H"],                   label="X‑Attn entropy ↓")
    plt.legend(); plt.xlabel("epoch"); plt.tight_layout(); plt.show()

    model_to.eval()
    preds, gts, caps, bases = {}, {}, {}, {}

    # Timing Containers
    inf_times = []                   # ms per image (forward only)

    with torch.no_grad():
        for img, ids, gt, cap, img_id in test_dl:
            img, ids = img.to(device), ids.to(device)

            # ---------- forward‑pass timing ------------------------
            if device == "cuda":
                torch.cuda.synchronize()
                t0 = torch.cuda.Event(enable_timing=True)
                t1 = torch.cuda.Event(enable_timing=True)
                t0.record()
                with amp.autocast('cuda', dtype=torch.float16):
                    p_batch = model_to(img, ids)
                t1.record(); torch.cuda.synchronize()
                batch_ms = t0.elapsed_time(t1)      # ms for the whole batch
            else:
                t0 = time.perf_counter()
                p_batch = model_to(img, ids)
                batch_ms = (time.perf_counter() - t0) * 1000
            inf_times.extend([batch_ms / img.size(0)] * img.size(0))
            # -------------------------------------------------------

            prob = p_batch.cpu()       # ← already probabilities

            for b in range(img.size(0)):
                iid = int(img_id[b])
                preds.setdefault(iid, []).append(prob[b,0].numpy())
                gts.setdefault(iid,   []).append(gt[b,0].numpy())
                caps.setdefault(iid,  []).append(cap[b])

                if iid not in bases:                        # save RGB only once
                    rgb = (img[b].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
                    bases[iid] = rgb

    # Average Inference Latency
    avg_ms = np.mean(inf_times)
    std_ms = np.std(inf_times)
    print(f"\n Average forward‑pass time  : {avg_ms:6.2f} ms / image "
        f"(± {std_ms:4.2f} ms  •  {len(inf_times)} images)")



    # Helper to Overlay Mask on Image
    def overlay_mask(mask, rgb):
        o = rgb.copy()
        o[mask==1] = (0.6*o[mask==1] + np.array([255,0,0])*0.4).astype(np.uint8)
        return o


    # Visualise All Captions per Test Image
    for iid in preds:
        base = bases[iid]
        print(f"\n=== Test image {iid} ===")
        for prob, gt_mask, caption in zip(preds[iid], gts[iid], caps[iid]):
            pred_bin = (prob > 0.5).astype(np.uint8)
            plt.figure(figsize=(10,4))
            plt.subplot(1,2,1); plt.imshow(overlay_mask(gt_mask,  base)); plt.axis('off'); plt.title("Ground truth")
            plt.subplot(1,2,2); plt.imshow(overlay_mask(pred_bin, base)); plt.axis('off'); plt.title("Prediction")
            plt.suptitle(caption, y=0.95); plt.tight_layout(); plt.show()
