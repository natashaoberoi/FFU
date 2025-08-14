# Imports
import sys
sys.path.insert(0, "/scratch/zt1/project/msml612/user/noberoi1/my_site_packages")
import torch

# sys.path.insert(0, '/scratch/zt1/project/msml612/user/noberoi1/ftfy_package')
# sys.path.insert(0, '/scratch/zt1/project/msml612/user/noberoi1/regex_package')
sys.path.insert(0, '/scratch/zt1/project/msml612/user/noberoi1/open_clip_package')
import open_clip
print(open_clip.__file__)
dir(open_clip)

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

import time
import os, math
import torch.nn.functional as F
from typing import Tuple
from torch import Tensor
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torch import amp
import numpy as np

IMG_SIZE   = 512

# Functions
def s2p(
    x: torch.Tensor,
    n_bins: int,
    pad_mode: str = "zeros",
) -> Tuple[torch.Tensor, int]:
    """
    Serial-to-Parallel: split a B×L×D sequence into B×n_bins×R×D, with
    optional right-padding when L is not divisible by n_bins.

    Returns
    -------
    chunks : torch.Tensor
        Tensor of shape (B, n_bins, R, D) where R = ceil(L / n_bins).
    orig_len : int
        The original (unpadded) sequence length, needed by `p2s`.
    """
    if x.dim() != 3:
        raise ValueError("x must have shape (batch, length, dim)")

    B, L, D = x.shape
    if n_bins <= 0:
        raise ValueError("n_bins must be a positive integer")

    # --- handle length not divisible by n_bins -----------------------------
    R = (L + n_bins - 1) // n_bins                # ceil division
    target_len = R * n_bins
    if target_len > L:
        pad_len = target_len - L
        if pad_mode == "zeros":
            x = F.pad(x, (0, 0, 0, pad_len))      # pad on sequence dim
        elif pad_mode == "reflect":
            x = F.pad(x, (0, 0, 0, pad_len), mode="reflect")
        else:
            raise ValueError(f"Unknown pad_mode: {pad_mode}")

    # --- reshape to (B, n_bins, R, D) --------------------------------------
    chunks = x.view(B, n_bins, R, D).contiguous()  # contiguous guarantees safe .view()
    return chunks, L                               # keep original length for P2S

def p2s(chunks: torch.Tensor, orig_len: int) -> torch.Tensor:
    """
    Parallel-to-Serial: inverse of `s2p`.

    Parameters
    ----------
    chunks : torch.Tensor
        Tensor coming from `s2p`, of shape (B, n_bins, R, D).
    orig_len : int
        Length of the sequence *before* any padding was added in `s2p`.
    """
    if chunks.dim() != 4:
        raise ValueError("chunks must have shape (batch, n_bins, R, dim)")

    B, n_bins, R, D = chunks.shape
    x = chunks.contiguous().view(B, n_bins * R, D)   # <- safened
    return x[:, :orig_len, :]      # strip any padding off the right

#  FFT_chunked / IFFT_chunked
# --------------------------------------------------------------------------- #
def fft_chunked(
        chunks: torch.Tensor,
        dim: int = 2,
        normalize: str = "backward",
    ) -> Tuple[torch.Tensor, int]:
    """
    Forward FFT on each chunk in the serial-to-parallel tensor.

    Parameters
    ----------
    chunks : (B, N_bins, R, D) real-valued tensor
        Output of `s2p()`. FFT is applied along the `R` axis.
    dim : int
        Axis along which to take the FFT (default 2 == R).
    normalize : {'backward', 'ortho', 'forward'}
        Normalisation mode passed to `torch.fft.rfft`.

    Returns
    -------
    X_f : (B, N_bins, R//2 + 1, D) complex tensor
        Frequency-domain representation (positive frequencies only).
    orig_len : int
        The original chunk length `R`; needed for `ifft_chunked`.
    """
    if chunks.dim() != 4:
        raise ValueError("Expect (B, N_bins, R, D) from s2p()")
    R = chunks.size(dim)
    # Real-to-complex FFT along the chunk length axis
    X_f = torch.fft.rfft(chunks, dim=dim, norm=normalize)
    return X_f, R

def ifft_chunked(
        X_f: torch.Tensor,
        orig_len: int,
        dim: int = 2,
        normalize: str = "backward",
    ) -> torch.Tensor:
    """
    Inverse FFT that restores time-domain chunks.

    Parameters
    ----------
    X_f : (B, N_bins, R//2 + 1, D) complex tensor
        Output of `fft_chunked`.
    orig_len : int
        Chunk length `R` before FFT (needed because rFFT drops negatives).
    dim : int
        Axis that held the FFT length.
    normalize : {'backward', 'ortho', 'forward'}
        Normalisation mode passed to `torch.fft.irfft`.

    Returns
    -------
    chunks_time : (B, N_bins, R, D) real-valued tensor
        Time-domain signal that should match the input to `fft_chunked`
        up to numerical precision.
    """
    if X_f.dim() != 4:
        raise ValueError("Expect (B, N_bins, R//2+1, D) from fft_chunked()")
    chunks_time = torch.fft.irfft(X_f, n=orig_len, dim=dim, norm=normalize)
    return chunks_time


#  Corrected IIR filter-bank
# --------------------------------------------------------------------------- #
def apply_iir_bank(
    X_f: Tensor,           # (B, N_bins, Freq, D)  complex
    theta: Tensor,         # (B, N_bins, F, D, 2)  real in (0,1)
    R: int,                # chunk length (time-domain)
    causal: bool = True,
) -> Tensor:
    B, N, Freq, D = X_f.shape
    B2, N2, F,  D2, two = theta.shape
    assert (B2, N2, D2, two) == (B, N, D, 2), "Θ shape mismatch"

    # causal shift (Θ_{n-1})
    if causal:
        z = torch.zeros_like(theta[:, :1])            # pad for n=0
        theta = torch.cat([z, theta[:, :-1]], dim=1)  # shift left

    θ0, θ1 = theta[..., 0], theta[..., 1]             # (B, N, F, D)

    # add a frequency axis in the *fourth* position → (B, N, F, 1, D)
    θ0 = θ0.unsqueeze(3)
    θ1 = θ1.unsqueeze(3)

    # build e^{-jω} vector with matching axis layout → (1,1,1,Freq,1)
    k  = torch.arange(Freq, device=X_f.device)        # 0 … R//2
    S  = torch.exp(-2j * math.pi * k / R).view(1, 1, 1, Freq, 1)
    S2 = S * S

    denom = 1 + θ0 * S + θ1 * S2                     # (B, N, F, Freq, D)
    H_f   = (1 / denom).sum(2)                       # sum over F → (B, N, Freq, D)

    return X_f * H_f                                 # shapes now match

def hyperfan_in_linear(layer: nn.Linear, fan_in: int, fan_out: int, *, relu=False):
    """
    HyperFan-in (Chang et al., 2020) for a linear *output* layer of a hyper-net.
    Guarantees Var(W) = 1 / fan_in  (see paper Eq. (2) + Table 1).
    """
    gain = math.sqrt(2.0) if relu else 1.0
    std  = gain * math.sqrt(1.0 / (fan_in * fan_out))
    nn.init.normal_(layer.weight, mean=0.0, std=std)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)

def film_apply(x: torch.Tensor,
               gamma: torch.Tensor,
               beta: torch.Tensor,
               log_gain: torch.Tensor) -> torch.Tensor:
    """
    Apply FiLM modulation with a *learnable* global gain.
        x       : (B,C,H,W)    – feature map
        gamma   : (B,C,1,1)    – scale
        beta    : (B,C,1,1)    – shift
        log_gain: ()           – scalar (log‑space) so gain≥1 after exp
    """
    gain = log_gain.exp()                 # ensures ≥1.0
    return gain * (gamma * x + beta)

def _check_cuda():
    # NOTE: Enforcing CUDA at import‑time restricts portability; consider letting the caller
    #       decide the device and moving tensors accordingly instead of hard‑failing.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required to run this model")
def coco_name(iid: int) -> str:
    return f"COCO_train2014_{iid:012d}.jpg"

def encode_text_clip(txt, ctx_len: int = 77) -> torch.LongTensor:
    if isinstance(txt, str):
        toks = clip_tok([txt], context_length=ctx_len)   # (1,77)
        return toks[0]
    else:
        return clip_tok(txt, context_length=ctx_len)     # (B,77)

def ann_ids_to_mask(ann_ids, wh):
    W, H = wh
    m = np.zeros((H, W), np.uint8)
    for aid in ann_ids:
        if aid in coco.anns:
            m |= coco.annToMask(coco.anns[aid]).astype(np.uint8)
    return m

# Classes
class CLIPFiLM(nn.Module):
    """
    Frozen CLIP text tower  →  MLP  →  γ | β per UNet stage.
    """
    def __init__(self, clip_text_encoder, enc_channels, d_hidden=512):
        """
        enc_channels : iterable with the C for each FiLM‑gated stage
                       e.g. (c0, c1, c2, c3)
        """
        super().__init__()
        self.clip_text_encoder = clip_text_encoder          # frozen
        self.enc_ch = tuple(enc_channels)                   # ←  store !

        d_clip = clip_text_encoder.text_projection.shape[1] # 512 for ViT‑B/32

        # small 2‑layer head that outputs γ‖β
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_clip),
            nn.Linear(d_clip, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 2 * sum(self.enc_ch))       # γ‖β
        )
        self._init_gamma_bias()

        # learnable global gain = how “loud” the text modulation is
        self.film_gain = nn.Parameter(torch.tensor(2.0))

    # ------------------------------------------------------------------
    def _init_gamma_bias(self):
        last = self.mlp[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        # γ starts at 1, β at 0 ⇒ network is vanilla UNet at step 0
        with torch.no_grad():
            last.bias[:sum(self.enc_ch)] = 1.0

    # ------------------------------------------------------------------
    def forward(self, clip_tokens):
        """
        clip_tokens: (B, 77) already tokenised by CLIP’s tokenizer.
        Returns two lists, each length len(enc_ch):
            gammas = [ (B,Ci,1,1), ... ]
            betas  = [ (B,Ci,1,1), ... ]
        """
        with torch.no_grad():
            z = self.clip_text_encoder.encode_text(clip_tokens)   # (B,512)

        film = self.mlp(z)                                        # (B,2ΣC)
        γ_vec, β_vec = torch.split(film, film.size(1)//2, dim=-1)

        gammas, betas, idx = [], [], 0
        for C in self.enc_ch:
            gammas.append(γ_vec[:, idx:idx+C].unsqueeze(-1).unsqueeze(-1))
            betas .append(β_vec[:, idx:idx+C].unsqueeze(-1).unsqueeze(-1))
            idx += C

        return gammas, betas, self.film_gain

class TwinCLIPFiLM(nn.Module):
    """
    frozen‑CLIP → two small MLP heads
    head_enc → γ_enc‖β_enc  (for the 4 encoder stages)
    head_dec → γ_dec‖β_dec  (for the 3 decoder gates)
    """
    def __init__(self, clip_text_encoder, enc_ch, dec_ch, d_hid=512):
        super().__init__()
        self.clip     = clip_text_encoder          # frozen – no grads
        self.enc_ch   = tuple(enc_ch)              # 💾 keep a private copy
        self.dec_ch   = tuple(dec_ch)

        d_clip = clip_text_encoder.text_projection.shape[1]  # 512 for ViT‑B/32

        def make_head(out_dim):
            mlp = nn.Sequential(
                nn.LayerNorm(d_clip),
                nn.Linear(d_clip, d_hid),
                nn.GELU(),
                nn.Linear(d_hid, out_dim)          # outputs γ‖β
            )
            nn.init.zeros_(mlp[-1].weight)
            nn.init.zeros_(mlp[-1].bias)
            with torch.no_grad():
                mlp[-1].bias[: out_dim // 2] = 1.0   # γ starts at 1
            return mlp

        self.head_enc = nn.Sequential(
            nn.LayerNorm(d_clip),
            nn.Linear(d_clip, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 2 * sum(enc_ch)),
        )

        self.head_dec = nn.Sequential(
            nn.LayerNorm(d_clip),
            nn.Linear(d_clip, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 2 * sum(dec_ch)),
        )
        self.head_enc[-1].bias.data[:  sum(self.enc_ch)] = 1.0   # γ half
        self.head_dec[-1].bias.data[:  sum(self.dec_ch)] = 1.0

        # global learnable gain
        # self.log_gain = nn.Parameter(torch.zeros(()))   # starts at gain = 1

    # --------------------------------------------------------------
    def _split(self, vec, ch_list):
        """split  (B, ΣC)  →  list[(B,C,1,1), …]"""
        outs, idx = [], 0
        for C in ch_list:
            outs.append(vec[:, idx:idx+C].unsqueeze(-1).unsqueeze(-1))
            idx += C
        return outs

    # --------------------------------------------------------------
    def forward(self, clip_tokens):
        with torch.no_grad():
            z = self.clip.encode_text(clip_tokens)      # (B,512)

        enc_vec = self.head_enc(z)                      # (B,2ΣCenc)
        dec_vec = self.head_dec(z)                      # (B,2ΣCdec)

        γ_enc, β_enc = torch.chunk(enc_vec, 2, dim=-1)
        γ_dec, β_dec = torch.chunk(dec_vec, 2, dim=-1)

        gammas_enc = [t.contiguous() for t in self._split(γ_enc, self.enc_ch)]
        betas_enc  = [t.contiguous() for t in self._split(β_enc, self.enc_ch)]
        gammas_dec = [t.contiguous() for t in self._split(γ_dec, self.dec_ch)]
        betas_dec  = [t.contiguous() for t in self._split(β_dec, self.dec_ch)]

        return gammas_enc, betas_enc, gammas_dec, betas_dec

class FocusAttentionHead(nn.Module):
    """
    Single Focus attention head (no gating) that matches
    yᵢ = Atten(Q xᵢ , K xᵢ_f , V xᵢ_f )

    Inputs:
        x_raw_chunks : Tensor[B, N_bins, R, D]   # xᵢ   (raw)
        x_f_chunks   : Tensor[B, N_bins, R, D]   # xᵢ_f (filtered)

    Output:
        Tensor[B, N_bins, R, D]  # still chunked; use p2s() afterward
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x_raw_chunks: torch.Tensor, x_f_chunks: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        x_raw_chunks : (B, N, R, D)
        x_f_chunks   : (B, N, R, D)

        Returns
        -------
        out : (B, N, R, D)
        """
        B, N, R, D = x_raw_chunks.shape
        assert x_f_chunks.shape == (B, N, R, D), "shape mismatch between raw and filtered chunks"
        assert D == self.d_model

        # Q from raw chunks, K and V from filtered ones
        Q = self.q_proj(x_raw_chunks)          # (B,N,R,D)
        K = self.k_proj(x_f_chunks)            # (B,N,R,D)
        V = self.v_proj(x_f_chunks)            # (B,N,R,D)

        # fold (B,N) into batch for efficient matmul
        Q = Q.view(B * N, R, D)
        K = K.view(B * N, R, D)
        V = V.view(B * N, R, D)

        attn_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(D)  # (B*N, R, R)
        attn_probs  = torch.softmax(attn_scores, dim=-1)
        attn_probs  = self.attn_dropout(attn_probs)

        Z = attn_probs @ V                                      # (B*N, R, D)
        Z = Z.view(B, N, R, D)

        return self.o_proj(Z)                                   # (B,N,R,D)

class GlobalConvEmbedder(nn.Module):
    """
    Fu et al. (2023) style 'regularised' depth-wise 1-D conv over the *sequence* axis,
    followed by adaptive max-pool that gives an oversampled embedding of size
    (O × Nbins) per channel.

    output: e  – shape (B, Nbins, D, O)
    """
    def __init__(self, d_model: int, oversample: int, n_bins: int,
                 kernel_size: int = 5):
        super().__init__()
        self.n_bins   = n_bins
        self.oversamp = oversample

        # depth-wise conv (groups = channels) ⇒ no channel mixing here
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,             # “regularised parameterisation”
            bias=False,
        )
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        returns e: (B, Nbins, D, O)
        """
        B, L, D = x.shape
        x_t = x.transpose(1, 2)            # (B, D, L)
        feat = self.conv(x_t)              # (B, D, L)

        # pool to (B, D, O * Nbins)
        pooled = F.adaptive_max_pool1d(
            feat, output_size=self.oversamp * self.n_bins
        )                                   # (B, D, O*Nbins)

        # reshape → (B, Nbins, D, O)
        pooled = pooled.view(B, D, self.n_bins, self.oversamp)
        e = pooled.permute(0, 2, 1, 3).contiguous()
        return e

class FilterMLP(nn.Module):
    """
    2-layer MLP that turns the shared embedding e (latent dim = O) into
    adaptive IIR kernels Θ of shape (B, Nbins, D, F, 2).

    A separate instance lives inside every Focus layer; parameters are *not*
    shared across layers.
    """
    def __init__(
        self,
        d_model: int,
        oversample: int,
        n_bins: int,
        f_bank: int,
        hidden: int = 128,
    ):
        super().__init__()
        self.n_bins  = n_bins
        self.d_model = d_model
        self.f_bank  = f_bank
        self.o_dim   = oversample

        # layer dims
        in_dim  = oversample          # O
        hid_dim = hidden
        out_dim = f_bank * 2          # F × 2

        # linear layers
        self.fc1 = nn.Linear(in_dim,  hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)

        # init: Xavier on first, HyperFan-in on output
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

        hyperfan_in_linear(
            self.fc2, fan_in=hid_dim, fan_out=out_dim, relu=False
        )

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        e : Tensor[B, Nbins, D, O]

        Returns
        -------
        Θ : Tensor[B, Nbins, D, F, 2]   with values in (0, 1)
        """
        B, N, D, O = e.shape
        assert (N, D, O) == (self.n_bins, self.d_model, self.o_dim), \
            "FilterMLP: embedding shape mismatch"

        # Flatten (B, N, D) → batch for MLP
        x = e.view(B * N * D, O)          # (B*N*D, O)

        h          = F.silu(self.fc1(x))  # use F.silu for broader PyTorch compat
        theta_raw  = self.fc2(h)          # (B*N*D, F*2)
        theta      = torch.sigmoid(theta_raw)

        # Reshape back to (B, N, D, F, 2)
        theta = theta.view(B, N, self.f_bank, D, 2)
        return theta

class HyperNetwork(nn.Module):
    """
    Orchestrates   GlobalConvEmbedder  (shared)  +  FilterMLP (per-Focus-layer).

    Usage:
        shared_embedder = GlobalConvEmbedder(D, O, Nbins)
        hnet_layer1     = HyperNetwork(shared_embedder, D, O, Nbins, F)

    Forward:
        Θ = hnet_layer1(x)      # (B, Nbins, D, F, 2)
    """

    def __init__(self,
                 shared_conv: GlobalConvEmbedder,
                 d_model: int,
                 oversample: int,
                 n_bins: int,
                 f_bank: int,
                 causal: bool = True):
        super().__init__()
        self.shared_conv = shared_conv         # pointer, not copy
        self.mlp         = FilterMLP(
            d_model, oversample, n_bins, f_bank
        )
        self.causal = causal

    @torch.no_grad()
    def _shift_right(self, theta: torch.Tensor) -> torch.Tensor:
        # causal shift: bin i uses Θ from bin i-1
        if not self.causal:
            return theta
        zeros = torch.zeros_like(theta[:, :1])
        return torch.cat([zeros, theta[:, :-1]], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, L, D)
        returns Θ : (B, Nbins, D, F, 2)   with causal shift applied
        """
        e = self.shared_conv(x)          # (B, Nbins, D, O)
        theta = self.mlp(e)              # raw Θ
        theta = self._shift_right(theta)
        return theta                     # ready for apply_iir_bank()

class FocusLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_bins: int,
        f_bank: int,
        oversample: int,
        shared_conv: "GlobalConvEmbedder",
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_bins   = n_bins
        self.d_model  = d_model

        # 1) hyper-network (shared conv + private 2-layer MLP)
        self.hyper = HyperNetwork(
            shared_conv, d_model, oversample, n_bins, f_bank, causal=True
        )

        # 2) Focus head
        self.attn  = FocusAttentionHead(d_model, dropout=dropout)

        # 3) Gate parameters (γ reset, ϕ update)
        self.gate_gamma = nn.Linear(d_model, d_model, bias=True)
        self.gate_phi   = nn.Linear(d_model, d_model, bias=True)

        # 4) LayerNorms
        self.ln_after_gate = nn.LayerNorm(d_model)
        self.ln_after_mlp  = nn.LayerNorm(d_model)

        # 5) Position-wise MLP
        hidden_dim = d_model * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    # --------------------------------------------------------------------- #
    #  forward
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : Tensor[B, L, D]
        returns y : Tensor[B, L, D]
        """
        B, L, D = x.shape
        # 1) Serial → parallel chunks
        chunks_raw, orig_len = s2p(x, self.n_bins)           # (B,N,R,D)

        # 2) Θ from hyper-network
        theta = self.hyper(x)                                # (B,N,F,D,2)

        # 3) FFT  -> IIR -> iFFT
        X_f, R            = fft_chunked(chunks_raw)          # (B,N,Freq,D)
        X_f_filt          = apply_iir_bank(X_f, theta, R)
        chunks_filt       = ifft_chunked(X_f_filt, R).contiguous()

        # 4) Focus attention head (Q=xᵢ, K/V=xᵢ_f)
        y_chunks          = self.attn(chunks_raw, chunks_filt)
        y_serial          = p2s(y_chunks, orig_len)          # (B,L,D)

        # ---- Focus head residual ----------------------------------------
        h1 = x + y_serial                                    # residual add

        # 5)  Reset γ  &  Update ϕ   (both from filtered sequence)
        xf_serial = p2s(chunks_filt, orig_len)               # (B,L,D)
        gamma = F.silu(self.gate_gamma(xf_serial))           # γ  in (−, +)
        phi   = torch.sigmoid(self.gate_phi(xf_serial))      # ϕ  in (0,1)

        # Convex combination     y_g = ϕ ⊙ h1  +  (1-ϕ) ⊙ γ
        y_g = phi * h1 + (1.0 - phi) * gamma

        # LayerNorm after gating
        y_norm = self.ln_after_gate(y_g)

        # 6) Position-wise MLP (+ dropout)  with residual
        y_mlp  = self.mlp(y_norm)
        y_out  = self.ln_after_mlp(y_norm + y_mlp)           # final output

        return y_out

# ───────────────────────────────────────────────────────────────
# Focus ×3 → FiLM (γ, β) generator  – with learnable scaling
# ───────────────────────────────────────────────────────────────
class FocusFiLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 96,
        n_bins: int = 8,
        f_bank: int = 4,
        oversample: int = 2,
        encoder_channels: tuple[int, int, int, int] = (64, 128, 256, 512),
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # shared conv + three Focus layers
        self.shared_conv = GlobalConvEmbedder(d_model, oversample, n_bins, 5)
        self.focus_stack = nn.ModuleList([
            FocusLayer(d_model, n_bins, f_bank, oversample, self.shared_conv)
            for _ in range(8)
        ])

        self.enc_ch   = encoder_channels
        self.film_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4*d_model),   # wider
            nn.GELU(),
            nn.Linear(4*d_model, 4*d_model), # one more hidden layer
            nn.GELU(),
            nn.Linear(4*d_model, 2*sum(self.enc_ch))  # final γ‖β
        )
        self._init_film_head()

        # learnable scalar that multiplies the γ/β deviation
        self.scale = nn.Parameter(torch.tensor(2.0))

        self.film_gain = nn.Parameter(torch.tensor(2.0))

    def _init_film_head(self):
        last = self.film_head[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        with torch.no_grad():
            last.bias[: sum(self.enc_ch)] = 1.0  # γ ≈ 1 at start

    # -----------------------------------------------------------
    def forward(self, token_ids: torch.LongTensor):
        """
        token_ids : (B, L)
        Returns
        -------
        y, gammas, betas  – lists of (B, Cᵢ, 1, 1) tensors
        """
        x = self.embed(token_ids)                # (B,L,D)
        for layer in self.focus_stack:
            x = layer(x)

        y = x.mean(1)                            # (B,D)

        film = self.film_head(y)                 # (B, 2ΣC)
        gamma_raw, beta_raw = torch.split(film, sum(self.enc_ch), -1)

        gammas, betas = [], []
        idx = 0
        for C in self.enc_ch:
            g = gamma_raw[:, idx:idx+C]          # (B,C)
            b = beta_raw[:, idx:idx+C]
            # scale deviations
            g = 1.0 + self.scale * (g - 1.0)
            b =       self.scale * b
            gammas.append(g.unsqueeze(-1).unsqueeze(-1))
            betas.append(b.unsqueeze(-1).unsqueeze(-1))
            idx += C
        return y, gammas, betas

class ConvBNReLU(nn.Module):
    """3×3 convolution + BatchNorm + ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.block(x)

class DoubleConv(nn.Module):
    """Two consecutive ConvBNReLU layers."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_ch, out_ch),
            ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.block(x)

class DownBlock(nn.Module):
    """Max‑pool → DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.block(x)

class UpBlock(nn.Module):
    """Upsample → conv → concat skip → DoubleConv."""

    def __init__(self, x_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        # NOTE: Nearest‑neighbor upsampling can introduce checkerboard artefacts; a learnable
        #       transposed convolution might give smoother results on medical images.
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(x_ch, out_ch, kernel_size=3, padding=1, bias=False),
        )
        self.double_conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y != 0 or diff_x != 0:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.double_conv(x)

class PositionalEncoding2D(nn.Module):
    """Fixed 2‑D sinusoidal positional encoding."""

    def __init__(self, channels: int):
        super().__init__()
        if channels % 4 != 0:
            raise ValueError("Channels must be divisible by 4")
        self.channels = channels
        # TODO: Pre‑compute the positional grid once and register as a buffer to save
        #       allocation time on every forward call.

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        b, c, h, w = x.shape
        device = x.device
        pe = torch.zeros(1, c, h, w, device=device, dtype=x.dtype)
        c_quarter = c // 4
        div_term = torch.exp(
            torch.arange(0, c_quarter, device=device, dtype=x.dtype) * (-math.log(10000.0) / c_quarter)
        )
        y_pos = torch.arange(h, device=device, dtype=x.dtype).unsqueeze(1)
        x_pos = torch.arange(w, device=device, dtype=x.dtype).unsqueeze(1)

        y_emb = y_pos * div_term.unsqueeze(0)
        x_emb = x_pos * div_term.unsqueeze(0)

        pe[:, 0:c_quarter] = torch.sin(y_emb).permute(1, 0).unsqueeze(2).repeat(1, 1, w)
        pe[:, c_quarter:2 * c_quarter] = torch.cos(y_emb).permute(1, 0).unsqueeze(2).repeat(1, 1, w)
        pe[:, 2 * c_quarter:3 * c_quarter] = torch.sin(x_emb).permute(1, 0).unsqueeze(1).repeat(1, h, 1)
        pe[:, 3 * c_quarter:] = torch.cos(x_emb).permute(1, 0).unsqueeze(1).repeat(1, h, 1)
        return x + pe

## Multi-Head Self Attention
class MHSA(nn.Module):
    """Multi‑Head Self‑Attention over flattened H×W tokens."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.pos_enc = PositionalEncoding2D(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        b, c, h, w = x.shape
        x = self.pos_enc(x)
        x = x.view(b, c, h * w).permute(0, 2, 1)  # (B, HW, C)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        x = x.permute(0, 2, 1).contiguous().view(b, c, h, w)
        return x

## Multi-Head Cross Attention
class MHCA(nn.Module):
    """Multi‑Head Cross‑Attention gating skip features S with Y."""

    def __init__(self, S_ch: int, Y_ch: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.S_proj = nn.Sequential(
            nn.Conv2d(S_ch, S_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(S_ch),
            nn.ReLU(inplace=True),
        )
        self.Y_proj = nn.Sequential(
            nn.Conv2d(Y_ch, S_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(S_ch),
            nn.ReLU(inplace=True),
        )
        self.pos_enc = PositionalEncoding2D(S_ch)
        self.attn = nn.MultiheadAttention(S_ch, num_heads, dropout=dropout, batch_first=True)
        self.filter_conv = nn.Sequential(
            nn.Conv2d(S_ch, S_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(S_ch),
            nn.Sigmoid(),
        )

    @staticmethod
    def _flatten_hw(t: torch.Tensor) -> torch.Tensor:
        b, c, h, w = t.shape
        return t.view(b, c, h * w).permute(0, 2, 1)  # (B, HW, C)

    def forward(self, S: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:  # noqa: D401
        b, Cs, Hs, Ws = S.shape
        Y_up = F.interpolate(Y, size=(Hs, Ws), mode="bilinear", align_corners=False)
        Y_up = self.Y_proj(Y_up)
        S_proj = self.S_proj(S)
        S_proj = self.pos_enc(S_proj)

        Q = self._flatten_hw(Y_up)
        K = Q
        V = self._flatten_hw(S_proj)
        attn_out, _ = self.attn(
            Q, K, V,
            need_weights=False,            # <- do NOT return the big matrix
            average_attn_weights=False
        )

        attn_out = attn_out.permute(0, 2, 1).contiguous().view(b, Cs, Hs, Ws)
        Z = self.filter_conv(attn_out)
        S_filtered = S * Z
        out = torch.cat([S_filtered, Y_up], dim=1)
        return out

class Text2VisXAttn(nn.Module):
    """
    Cross‑attention     Q ← text‑tokens (77×512)      KV ← visual map (H·W×C)
    Returns             fused_vis  and  mean entropy  (lower → more focused)
    """
    def __init__(self, vis_ch: int, txt_dim: int = 512, heads: int = 8):
        super().__init__()
        self.q    = nn.Linear(txt_dim, vis_ch)          # (B,77,C)
        self.kv   = nn.Conv2d(vis_ch, 2 * vis_ch, 1)
        self.attn = nn.MultiheadAttention(vis_ch, heads, batch_first=True)
        self.out  = nn.Conv2d(vis_ch, vis_ch, 1)

    def forward(self, vis, txt):                        # vis (B,C,H,W)   txt (B,77,512)
        B, C, H, W = vis.shape

        # --- KV from image -----------------------------------------------------
        k, v = self.kv(vis).chunk(2, 1)                 # (B,C,H,W) each
        k = k.flatten(2).transpose(1, 2)                # (B,HW,C)
        v = v.flatten(2).transpose(1, 2)

        # --- per‑token Q from text --------------------------------------------
        q = self.q(txt)                                 # (B,77,C)

        z, attn_w = self.attn(q, k, v, need_weights=True)  # attn_w (B,77,HW)

        # token‑wise entropy  H = −Σ p log p , then mean over B & tokens
        entropy = -(attn_w * attn_w.clamp_min(1e-9).log()).sum(-1).mean()

        # collapse 77 token embeddings back to one C‑vector
        z = z.mean(1)                                   # (B,C)
        z = self.out(z[:, :, None, None])               # (B,C,1,1)

        return vis + z, entropy

class UTransformer(nn.Module):
    """U‑Net backbone with MHSA bottleneck and MHCA‑gated skips (GPU‑only)."""

    def __init__(
        self,
        vocab_size: int,
        in_channels: int = 1,
        num_classes: int = 1,
        base_channels: int = 64,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        _check_cuda()

        c0, c1, c2, c3 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        )

        # Encoder
        self.inc = DoubleConv(in_channels, c0)
        self.patch_embed = nn.Conv2d(c0, c0, kernel_size=4, stride=4, bias=False)
        self.down1 = DownBlock(c0, c1)
        self.down2 = DownBlock(c1, c2)
        self.down3 = DownBlock(c2, c3)

        # 1) Focus‑FiLM generator
        self.film = TwinCLIPFiLM(
            clip_text_encoder = clip_model,    # from open_clip
            enc_ch = (c0, c1, c2, c3),
            dec_ch = (2*c2, 2*c1, 2*c0)
        )

        # Bottleneck
        self.mhsa = MHSA(c3, num_heads=num_heads)

        # Skip gating
        self.mhca3 = MHCA(S_ch=c2, Y_ch=c3, num_heads=num_heads)
        self.mhca2 = MHCA(S_ch=c1, Y_ch=c2, num_heads=num_heads)
        self.mhca1 = MHCA(S_ch=c0, Y_ch=c1, num_heads=num_heads)

        # AFTER  (use the *post‑concat* channel count)
        self.xattn3 = Text2VisXAttn(2 * c2)   # mhca3 outputs 2·c2
        self.xattn2 = Text2VisXAttn(2 * c1)   # mhca2 outputs 2·c1
        self.xattn1 = Text2VisXAttn(2 * c0)   # mhca1 outputs 2·c0

        # storage for diagnostics
        self.attn_entropy = []

        # Decoder
        self.up3 = UpBlock(x_ch=c3, skip_ch=2 * c2, out_ch=c2)
        self.up2 = UpBlock(x_ch=c2, skip_ch=2 * c1, out_ch=c1)
        self.up1 = UpBlock(x_ch=c1, skip_ch=2 * c0, out_ch=c0)

        self.film_gain_log = nn.Parameter(torch.tensor(0.0))    # log‑scale

        # Output
        self.refine = nn.Sequential(
            nn.Conv2d(c0, c0, 3, padding=1, bias=False),
            nn.BatchNorm2d(c0),
            nn.ReLU(inplace=True),
            nn.Conv2d(c0, num_classes, 3, padding=1)     # no BN/ReLU here
        )

        # keep channel counts handy for aux heads
        self.c3_channels = c3
    # -----------------------------------------------------------------------------
    # U‑Transformer ‑‑ new forward that consumes TwinCLIPFiLM
    # -----------------------------------------------------------------------------
    def forward(self,
                x: torch.Tensor,
                clip_tokens: torch.LongTensor          # <- CLIP tokens, not BERT ids
              ) -> torch.Tensor:
        H_in, W_in = x.shape[-2:]

        # ── 1. pad so spatial dims are multiples of 32 ──────────────────────────
        pad_h, pad_w = (-H_in) % 32, (-W_in) % 32
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # 2. FiLM parameters from CLIP text tower ─────────────────────────────
        gE, bE, gD, bD = self.film(clip_tokens)   # lists
        # smooth bounds (keep gradients alive near limits).
        def _squash_list(xs, lo, hi):
            mid  = 0.5 * (hi + lo)
            half = 0.5 * (hi - lo)
            return [mid + half * torch.tanh((x - mid) / half) for x in xs]
        gE = _squash_list(gE, 0.5, 2.0);  bE = [torch.tanh(x) for x in bE]
        gD = _squash_list(gD, 0.5, 2.5);  bD = [torch.tanh(x) for x in bD]

        # 0)  FULL token sequence **with grad** (do NOT wrap in no_grad)
        txt_tokens = self.film.clip.token_embedding(clip_tokens)      # (B,77,512)
        txt_tokens = txt_tokens + self.film.clip.positional_embedding
        txt_tokens = self.film.clip.transformer(
            txt_tokens, attn_mask=self.film.clip.attn_mask
        )

        # ── 3. ENCODER ----------------------------------------------------------
        x0 = self.inc(x)
        x0 = self.patch_embed(x0)
        # x0 = gain * (gE[0] * x0 + bE[0])                # stage‑0
        x0 = film_apply(x0, gE[0], bE[0], self.film_gain_log)

        x1 = self.down1(x0)
        # x1 = gain * (gE[1] * x1 + bE[1])                # stage‑1
        x1 = film_apply(x1, gE[1], bE[1], self.film_gain_log)

        x2 = self.down2(x1)
        # x2 = gain * (gE[2] * x2 + bE[2])                # stage‑2
        x2 = film_apply(x2, gE[2], bE[2], self.film_gain_log)

        x3 = self.down3(x2)
        # x3 = gain * (gE[3] * x3 + bE[3])                # stage‑3
        x3 = film_apply(x3, gE[3], bE[3], self.film_gain_log)

        # ── 4. Bottleneck (MHSA) -------------------------------------------------
        x4 = self.mhsa(x3)

        # --- gate‑3 --------------------------------------------------------------
        s3 = self.mhca3(S=x2, Y=x4)                      # (B,2c2,H/4,W/4)
        s3, ent3 = self.xattn3(s3, txt_tokens)
        self.attn_entropy.append(ent3.item())
        s3 = film_apply(s3, gD[0], bD[0], self.film_gain_log)
        x  = self.up3(x4, s3)
        # --- gate‑2 --------------------------------------------------------------
        s2 = self.mhca2(S=x1, Y=x)
        s2, ent2 = self.xattn2(s2, txt_tokens)
        self.attn_entropy.append(ent2.item())
        s2 = film_apply(s2, gD[1], bD[1], self.film_gain_log)
        x  = self.up2(x, s2)
        # --- gate‑1 --------------------------------------------------------------
        s1 = self.mhca1(S=x0, Y=x)
        s1, ent1 = self.xattn1(s1, txt_tokens)
        self.attn_entropy.append(ent1.item())
        s1 = film_apply(s1, gD[2], bD[2], self.film_gain_log)
        x  = self.up1(x, s1)

        # ── 6. Upsample back to input resolution & final head -------------------
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        if pad_h or pad_w:
            x = x[..., :H_in, :W_in]

        # return torch.sigmoid(self.refine(x))            # refine‑head → 1‑ch prob‑map
        return self.refine(x)                           # return logits

class GRefDataset(Dataset):
    def __init__(self, entries, img_dir, img_size):
        self.samples, self.img_dir, self.img_size = [], img_dir, img_size
        self._wh_ = {}
        for e in entries:
            path = os.path.join(img_dir, coco_name(e["image_id"]))
            if not os.path.isfile(path): continue
            for s in e["sentences"]:
              self.samples.append(dict(
                  img_id  = e["image_id"],
                  path    = path,
                  ann_ids = e["ann_id"],
                  clip_ids= encode_text_clip(s["sent"]),   # tokenise here
                  text    = s["sent"]
              ))
        if not self.samples: raise RuntimeError("No samples")
    def __len__(self): return len(self.samples)
    def _wh(self, iid, path):
        if iid not in self._wh_:
            with Image.open(path) as im: self._wh_[iid]=im.size
        return self._wh_[iid]
    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s["path"]).convert("RGB").resize(
              (IMG_SIZE,IMG_SIZE), Image.BILINEAR)
        img = torch.from_numpy(np.array(img)).permute(2,0,1).float()/255.
        W,H = Image.open(s["path"]).size
        m = ann_ids_to_mask(s["ann_ids"], (W,H))
        m = Image.fromarray(m*255).resize((IMG_SIZE,IMG_SIZE), Image.NEAREST)
        m = torch.from_numpy(np.array(m)//255)[None].float()
        # clip_ids = encode_text_clip(s["text"])          # (77,)
        return img, s["clip_ids"], m, s["text"], s["img_id"]




