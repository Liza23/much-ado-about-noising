"""Visualize slot attention maps from a trained SlotObjectEncoder checkpoint.

For each sampled trajectory, shows:
  - Top row: input image frames (future k frames)
  - Per-slot rows: attention heatmap overlaid on frame (red = high attention)
  - Selector weight bar: which slot "won" for each frame

Usage:
    python examples/viz_slot_attention.py \
        --ckpt /data/user_data/ldahiya/mip_render/lift-mh-image/slot_intent_seed0/models/model_best.pt \
        --hdf5 ~/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/snapshots/e25b108012219f9f59a84ec6538f5ba52c931353/robomimic/lift/mh/image.hdf5 \
        --n-samples 4 \
        --out rollouts/slot_attn_viz.png
"""

import argparse
import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from mip.networks.slot_attention import SlotObjectEncoder


# ── helpers ──────────────────────────────────────────────────────────────────

def load_slot_encoder(ckpt_path: str, num_slots: int = 4, device: str = "cpu") -> SlotObjectEncoder:
    ckpt = torch.load(ckpt_path, map_location=device)
    # The checkpoint stores the full agent state dict.  Extract slot_encoder keys.
    state = ckpt if isinstance(ckpt, dict) else ckpt
    # checkpoint may store slot_encoder as a nested dict or flattened with prefix
    if "slot_encoder" in state:
        enc_state = state["slot_encoder"]
    else:
        prefix = "slot_encoder."
        enc_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not enc_state:
        raise KeyError(
            f"No 'slot_encoder' key found in checkpoint.  "
            f"Available top-level keys: {sorted(state.keys())}."
        )
    # Infer hyper-params from weight shapes
    # NOTE: slots_mu is always (1, 1, slot_dim) — num_slots must be passed explicitly
    slot_dim  = enc_state["slot_attention.slots_mu"].shape[2]
    num_iters = 3  # not stored; use the same default as training
    obj_state_dim = enc_state["obj_regressor.weight"].shape[0]
    slot_input_dim = enc_state["slot_attention.to_k.weight"].shape[1]
    # proj.weight shape: (slot_input_dim, in_channels) — 128 means layer2, 256 means layer3
    proj_in_channels = enc_state["proj.weight"].shape[1]
    use_layer2 = (proj_in_channels == 128)
    use_recon_decoder = any(k.startswith("recon_decoder.") for k in enc_state)
    use_soft_selector = any(k.startswith("selector.") for k in enc_state)
    print(f"  proj in_channels={proj_in_channels} → use_layer2={use_layer2}")
    print(f"  recon_decoder={use_recon_decoder}  soft_selector={use_soft_selector}")

    enc = SlotObjectEncoder(
        num_slots=num_slots,
        slot_dim=slot_dim,
        num_iters=num_iters,
        obj_state_dim=obj_state_dim,
        slot_input_dim=slot_input_dim,
        use_layer2=use_layer2,
        use_recon_decoder=use_recon_decoder,
        use_soft_selector=use_soft_selector,
    ).to(device)
    enc.load_state_dict(enc_state, strict=False)
    enc.eval()
    return enc


def sample_frames(hdf5_path: str, n_samples: int, k: int, image_key: str = "agentview_image"):
    """Return (n_samples, k, H, W, C) uint8 frames sampled from random demo trajectories."""
    with h5py.File(os.path.expanduser(hdf5_path), "r") as f:
        demos = list(f["data"].keys())
        rng = np.random.default_rng(42)
        chosen = rng.choice(demos, size=n_samples, replace=False)
        frames = []
        for demo in chosen:
            imgs = f["data"][demo]["obs"][image_key][:]  # (T, H, W, C)
            T = imgs.shape[0]
            # pick a random start that leaves k future frames
            start = rng.integers(0, max(1, T - k))
            clip = imgs[start : start + k]
            if clip.shape[0] < k:
                clip = np.pad(clip, ((0, k - clip.shape[0]), (0,0), (0,0), (0,0)), mode="edge")
            frames.append(clip)
    return np.stack(frames)  # (n_samples, k, H, W, C)


def run_encoder(enc: SlotObjectEncoder, frames_np: np.ndarray, device: str):
    """frames_np: (B, k, H, W, C) uint8  → run encoder with attention."""
    # normalize to [0,1] and move channels first
    x = torch.tensor(frames_np.astype(np.float32) / 255.0)  # (B, k, H, W, C)
    x = x.permute(0, 1, 4, 2, 3).to(device)                  # (B, k, C, H, W)
    with torch.no_grad():
        intent, obj_pred, attn, (Hp, Wp) = enc(x, return_attn=True)
    # attn: (B*k, K, N)
    B, k = frames_np.shape[:2]
    attn = attn.cpu().numpy().reshape(B, k, enc.slot_attention.num_slots, Hp * Wp)
    return attn, (Hp, Wp)


def run_recon(enc: SlotObjectEncoder, frames_np: np.ndarray, device: str):
    """Run slot encoder + recon decoder; return per-slot masked RGB and composite.

    Returns:
        per_slot: (B, k, K, H, W, 3) uint8 — each slot's masked RGB contribution
        composite: (B, k, H, W, 3) uint8 — sum of all slot contributions
        masks_np:  (B, k, K, H, W) float — per-slot soft mask (sums to 1 over K)
    """
    if enc.recon_decoder is None:
        raise ValueError("This checkpoint was trained without a recon decoder.")
    H_orig, W_orig = frames_np.shape[2], frames_np.shape[3]
    B, k = frames_np.shape[:2]
    K = enc.slot_attention.num_slots

    x = torch.tensor(frames_np.astype(np.float32) / 255.0).permute(0, 1, 4, 2, 3).to(device)
    Bk = B * k

    with torch.no_grad():
        # get per-frame slots
        feat = enc.cnn(x.reshape(Bk, *x.shape[2:]))
        feat = feat.flatten(2).permute(0, 2, 1)
        feat = enc.proj(feat)
        slots = enc.slot_attention(feat)  # (B*k, K, slot_dim)

        # replicate decoder internals to get per-slot rgb + masks
        D = slots.shape[2]
        xs = torch.linspace(-1, 1, W_orig, device=device)
        ys = torch.linspace(-1, 1, H_orig, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        pos = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(Bk * K, -1, -1, -1)
        s = slots.reshape(Bk * K, D, 1, 1).expand(-1, -1, H_orig, W_orig)
        inp = torch.cat([s, pos], dim=1)
        out = enc.recon_decoder.decoder(inp).reshape(Bk, K, 4, H_orig, W_orig)
        rgb   = torch.sigmoid(out[:, :, :3])       # (B*k, K, 3, H, W)
        masks = out[:, :, 3:].softmax(dim=1)        # (B*k, K, 1, H, W)
        per_slot_rgb = (masks * rgb)                # (B*k, K, 3, H, W)
        composite    = per_slot_rgb.sum(dim=1)      # (B*k, 3, H, W)

    def to_uint8(t):
        return (t.clamp(0, 1) * 255).byte().cpu().numpy()

    per_slot_np = to_uint8(per_slot_rgb).reshape(B, k, K, 3, H_orig, W_orig)
    per_slot_np = per_slot_np.transpose(0, 1, 2, 4, 5, 3)  # → (B, k, K, H, W, 3)
    composite_np = to_uint8(composite).reshape(B, k, 3, H_orig, W_orig)
    composite_np = composite_np.transpose(0, 1, 3, 4, 2)   # → (B, k, H, W, 3)
    masks_np = masks.squeeze(2).cpu().numpy().reshape(B, k, K, H_orig, W_orig)
    return per_slot_np, composite_np, masks_np


# ── plotting ─────────────────────────────────────────────────────────────────

# One distinct colormap per slot; extend if K > 4
_SLOT_CMAPS = ["turbo", "cool", "autumn", "winter"]
_SLOT_BORDER = ["#00ccff", "#ff8800", "#22aa44", "#9933aa"]


def make_heatmap_overlay(frame_hwc: np.ndarray, attn_map: np.ndarray, cmap_name: str = "turbo", alpha: float = 0.7) -> np.ndarray:
    """Overlay attention map on frame — high-attention pixels get bright color, low stay original."""
    H, W = frame_hwc.shape[:2]
    heat = torch.tensor(attn_map[None, None]).float()
    heat = F.interpolate(heat, size=(H, W), mode="bilinear", align_corners=False)
    heat = heat.squeeze().numpy()
    # normalize then raise to power < 1 to boost contrast in nearly-uniform maps
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    heat = heat ** 0.4  # exaggerate relative differences

    cmap = matplotlib.colormaps[cmap_name]
    heat_rgba = cmap(heat)  # (H, W, 4)
    heat_rgb = (heat_rgba[:, :, :3] * 255).astype(np.uint8)

    # use attention weight as per-pixel alpha so low-attention areas show original frame
    per_pixel_alpha = (heat * alpha)[..., None]  # (H, W, 1)
    blended = (per_pixel_alpha * heat_rgb + (1 - per_pixel_alpha) * frame_hwc).clip(0, 255).astype(np.uint8)
    return blended


def visualize(frames_np, attn, spatial_shape, out_path: str):
    """
    frames_np: (B, k, H, W, C) uint8
    attn:      (B, k, K, N)

    Layout: rows = [raw | slot_0 | slot_1 | ...] repeated per trajectory.
    Each slot row uses a distinct colormap.
    """
    B, k, H, W, C = frames_np.shape
    Hp, Wp = spatial_shape
    K = attn.shape[2]

    n_rows = 1 + K
    fig, axes = plt.subplots(
        B * n_rows, k,
        figsize=(k * 1.8, B * n_rows * 1.8),
        squeeze=False,
        gridspec_kw={"hspace": 0.04, "wspace": 0.04},
    )

    for b in range(B):
        row_offset = b * n_rows
        for t in range(k):
            frame = frames_np[b, t]

            # Row 0: raw frame
            ax = axes[row_offset][t]
            ax.imshow(frame)
            ax.axis("off")
            if t == 0:
                ax.set_ylabel("input", fontsize=8, labelpad=3)
            if b == B - 1:
                ax.set_xlabel(f"t+{t+1}", fontsize=8)

            # Rows 1..K: per-slot overlay
            for slot_idx in range(K):
                cmap_name = _SLOT_CMAPS[slot_idx % len(_SLOT_CMAPS)]
                border_color = _SLOT_BORDER[slot_idx % len(_SLOT_BORDER)]
                attn_map = attn[b, t, slot_idx].reshape(Hp, Wp)
                overlay = make_heatmap_overlay(frame, attn_map, cmap_name=cmap_name)

                ax = axes[row_offset + 1 + slot_idx][t]
                ax.imshow(overlay)
                ax.axis("off")

                for spine in ax.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(1.2)
                    spine.set_visible(True)

                if t == 0:
                    ax.set_ylabel(f"slot {slot_idx}", fontsize=8, labelpad=3,
                                  color=border_color)

    fig.suptitle("Slot attention maps (lift-mh-image)", fontsize=11, y=1.002)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def visualize_recon(frames_np, per_slot, composite, out_path: str):
    """
    frames_np: (B, k, H, W, C) uint8
    per_slot:  (B, k, K, H, W, 3) uint8
    composite: (B, k, H, W, 3) uint8

    Layout per trajectory: raw | slot_0_recon | slot_1_recon | ... | composite
    """
    B, k = frames_np.shape[:2]
    K = per_slot.shape[2]
    n_rows = 1 + K + 1  # raw + K slot recons + composite

    fig, axes = plt.subplots(
        B * n_rows, k,
        figsize=(k * 1.8, B * n_rows * 1.8),
        squeeze=False,
        gridspec_kw={"hspace": 0.04, "wspace": 0.04},
    )

    for b in range(B):
        row_offset = b * n_rows
        for t in range(k):
            frame = frames_np[b, t]

            # Row 0: raw input
            ax = axes[row_offset][t]
            ax.imshow(frame)
            ax.axis("off")
            if t == 0:
                ax.set_ylabel("input", fontsize=8, labelpad=3)

            # Rows 1..K: per-slot masked RGB
            for slot_idx in range(K):
                border_color = _SLOT_BORDER[slot_idx % len(_SLOT_BORDER)]
                ax = axes[row_offset + 1 + slot_idx][t]
                ax.imshow(per_slot[b, t, slot_idx])
                ax.axis("off")
                for spine in ax.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(1.2)
                    spine.set_visible(True)
                if t == 0:
                    ax.set_ylabel(f"slot {slot_idx}", fontsize=8, labelpad=3,
                                  color=border_color)

            # Last row: composite reconstruction
            ax = axes[row_offset + 1 + K][t]
            ax.imshow(composite[b, t])
            ax.axis("off")
            if t == 0:
                ax.set_ylabel("recon", fontsize=8, labelpad=3)

    fig.suptitle("Slot reconstructions (lift-mh-image)", fontsize=11, y=1.002)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument(
        "--hdf5",
        default="~/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/snapshots/"
                "e25b108012219f9f59a84ec6538f5ba52c931353/robomimic/lift/mh/image.hdf5",
    )
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--num-slots", type=int, default=4)
    parser.add_argument("--k", type=int, default=8, help="number of future frames")
    parser.add_argument("--image-key", default="agentview_image")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="rollouts/slot_attn_viz.png")
    parser.add_argument("--show-recon", action="store_true",
                        help="Also generate per-slot reconstruction visualization")
    args = parser.parse_args()

    print("Loading slot encoder from checkpoint …")
    enc = load_slot_encoder(args.ckpt, num_slots=args.num_slots, device=args.device)

    print(f"Sampling {args.n_samples} trajectories from dataset …")
    frames_np = sample_frames(args.hdf5, args.n_samples, args.k, args.image_key)

    print("Running encoder (return_attn=True) …")
    attn, spatial_shape = run_encoder(enc, frames_np, args.device)
    print(f"  attn shape:   {attn.shape}")
    print(f"  feature grid: {spatial_shape}")

    print("Generating attention visualization …")
    visualize(frames_np, attn, spatial_shape, args.out)

    if args.show_recon:
        print("Running recon decoder …")
        per_slot, composite, _ = run_recon(enc, frames_np, args.device)
        recon_out = args.out.replace(".png", "_recon.png")
        print("Generating reconstruction visualization …")
        visualize_recon(frames_np, per_slot, composite, recon_out)


if __name__ == "__main__":
    main()
