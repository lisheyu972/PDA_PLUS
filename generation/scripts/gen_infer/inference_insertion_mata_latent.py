

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange
from peft import PeftModel
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

GENERATION_ROOT = Path(__file__).resolve().parents[2]

from flux2.sampling import batched_prc_txt, get_schedule
from flux2.util import load_ae, load_flow_model, load_text_encoder

sys.path.insert(0, str(Path(__file__).parent.parent / "gen_train"))
from train_lora_insertion import ConditionTypeEmbedding


FIXED_SIZE = (512, 512)
LATENT_H = 32
LATENT_W = 32
NUM_STEPS = 50


def _patch_grams_latent(
    feat: torch.Tensor,
    mask: torch.Tensor,
    patch_grid_size: int,
    min_patch_pixels: int = 4,
) -> torch.Tensor | None:

    _, channels, height, width = feat.shape
    k = max(1, min(int(patch_grid_size), height, width))
    patch_h, patch_w = height // k, width // k
    if patch_h == 0 or patch_w == 0:
        return None

    feat = feat[:, :, :k * patch_h, :k * patch_w]
    mask = mask[:, :, :k * patch_h, :k * patch_w]
    feat_p = rearrange(
        feat,
        "1 c (kh ph) (kw pw) -> (kh kw) c (ph pw)",
        kh=k,
        kw=k,
        ph=patch_h,
        pw=patch_w,
    )
    mask_p = rearrange(
        mask,
        "1 1 (kh ph) (kw pw) -> (kh kw) 1 (ph pw)",
        kh=k,
        kw=k,
        ph=patch_h,
        pw=patch_w,
    )

    feat_masked = feat_p * mask_p
    denom = mask_p.sum(dim=-1).squeeze(1)
    valid = denom >= min_patch_pixels
    if not valid.any():
        return None

    grams = torch.bmm(feat_masked, feat_p.transpose(1, 2))
    grams = grams / denom.view(-1, 1, 1).clamp_min(1.0)
    return grams[valid]


def multi_scale_spatial_gram_latent(
    feat: torch.Tensor,
    mask: torch.Tensor,
    patch_grid_sizes: list[int],
    min_patch_pixels: int = 4,
) -> torch.Tensor:

    pieces = []
    for k in patch_grid_sizes:
        grams = _patch_grams_latent(
            feat,
            mask,
            patch_grid_size=k,
            min_patch_pixels=min_patch_pixels,
        )
        if grams is not None:
            pieces.append(grams.flatten(1))
    if not pieces:
        channels = feat.shape[1]
        return feat.new_zeros((0, channels * channels))
    return torch.cat(pieces, dim=0)


def _sinkhorn(
    a: torch.Tensor,
    b: torch.Tensor,
    cost: torch.Tensor,
    blur: float,
    n_iter: int = 100,
) -> torch.Tensor:

    eps = max(float(blur), 1e-6)
    log_a = torch.log(a.clamp_min(1e-30))
    log_b = torch.log(b.clamp_min(1e-30))
    kernel = -cost / eps
    f = torch.zeros_like(log_a)
    g = torch.zeros_like(log_b)
    for _ in range(n_iter):
        f = log_a - torch.logsumexp(kernel + g[None, :], dim=1)
        g = log_b - torch.logsumexp(kernel + f[:, None], dim=0)
    transport = (kernel + f[:, None] + g[None, :]).exp()
    return (transport * cost).sum()


def sinkhorn_divergence(
    x: torch.Tensor,
    y: torch.Tensor,
    blur: float = 0.05,
    n_iter: int = 100,
) -> torch.Tensor:

    n_x, n_y = x.shape[0], y.shape[0]
    if n_x == 0 or n_y == 0:
        return x.sum() * 0.0
    a = x.new_full((n_x,), 1.0 / n_x)
    b = y.new_full((n_y,), 1.0 / n_y)
    cost_xy = torch.cdist(x, y, p=2).square()
    cost_xx = torch.cdist(x, x, p=2).square()
    cost_yy = torch.cdist(y, y, p=2).square()
    return (
        _sinkhorn(a, b, cost_xy, blur, n_iter)
        - 0.5 * _sinkhorn(a, a, cost_xx, blur, n_iter)
        - 0.5 * _sinkhorn(b, b, cost_yy, blur, n_iter)
    )


def build_environment_masks(
    obj_mask: np.ndarray,
    local_dilation: int,
    obj_erosion: int,
    sem_mask: np.ndarray | None,
    alpha_sem: float,
) -> dict:

    obj = (obj_mask > 0).astype(bool)
    dilated = binary_dilation(obj, iterations=max(local_dilation, 1))
    local_ring = dilated & ~obj

    if obj_erosion > 0:
        eroded = binary_erosion(obj, iterations=obj_erosion)
        obj_for_gram = eroded if eroded.any() else obj
    else:
        obj_for_gram = obj

    result = {
        "obj_for_gram": obj_for_gram.astype(np.uint8),
        "local": local_ring.astype(np.uint8),
        "semantic": None,
        "alpha_sem": 0.0,
    }
    if sem_mask is not None and alpha_sem > 0:
        semantic = (sem_mask > 0).astype(bool) & ~obj
        if semantic.any():
            result["semantic"] = semantic.astype(np.uint8)
            result["alpha_sem"] = float(alpha_sem)
    return result


def mata_loss_latent(
    latent_tokens: torch.Tensor,
    obj_mask_t: torch.Tensor,
    env_local_t: torch.Tensor,
    env_sem_t: torch.Tensor | None,
    alpha_sem: float,
    patch_grid_sizes: list[int],
    sinkhorn_blur: float,
    sinkhorn_iters: int,
    latent_h: int,
    latent_w: int,
    min_patch_pixels: int = 4,
) -> torch.Tensor:

    if latent_tokens.shape[0] != 1:
        raise ValueError("Latent MATA currently supports batch size 1 only")
    if latent_tokens.shape[1] != latent_h * latent_w:
        raise ValueError(
            f"Expected {latent_h * latent_w} latent tokens, "
            f"got {latent_tokens.shape[1]}"
        )

    latent_map = rearrange(
        latent_tokens[0],
        "(h w) c -> 1 c h w",
        h=latent_h,
        w=latent_w,
    )
    m_obj = (
        F.interpolate(obj_mask_t, size=(latent_h, latent_w), mode="nearest")
        > 0.5
    ).to(latent_map.dtype)
    m_loc = (
        F.interpolate(env_local_t, size=(latent_h, latent_w), mode="nearest")
        > 0.5
    ).to(latent_map.dtype)
    if env_sem_t is not None and alpha_sem > 0:
        m_sem = (
            F.interpolate(env_sem_t, size=(latent_h, latent_w), mode="nearest")
            > 0.5
        ).to(latent_map.dtype)
    else:
        m_sem = None

    gram_obj = multi_scale_spatial_gram_latent(
        latent_map, m_obj, patch_grid_sizes, min_patch_pixels
    )
    gram_loc = multi_scale_spatial_gram_latent(
        latent_map, m_loc, patch_grid_sizes, min_patch_pixels
    )
    if gram_obj.shape[0] == 0 or gram_loc.shape[0] == 0:
        return latent_tokens.sum() * 0.0

    scale = gram_obj.detach().abs().mean().clamp_min(1e-6)
    gram_obj = gram_obj / scale
    gram_loc = gram_loc / scale

    gram_sem = None
    if m_sem is not None:
        gram_sem = multi_scale_spatial_gram_latent(
            latent_map, m_sem, patch_grid_sizes, min_patch_pixels
        )
        gram_sem = gram_sem / scale if gram_sem.shape[0] > 0 else None

    if gram_sem is not None:
        alpha_loc = max(1.0 - alpha_sem, 0.0)
        gram_env = torch.cat([gram_loc, gram_sem], dim=0)
        w_env = torch.cat(
            [
                gram_loc.new_full(
                    (gram_loc.shape[0],), alpha_loc / gram_loc.shape[0]
                ),
                gram_sem.new_full(
                    (gram_sem.shape[0],), alpha_sem / gram_sem.shape[0]
                ),
            ]
        )
        w_env = w_env / w_env.sum().clamp_min(1e-12)
        w_obj = gram_obj.new_full(
            (gram_obj.shape[0],), 1.0 / gram_obj.shape[0]
        )
        cost = torch.cdist(gram_obj, gram_env, p=2).square()
        loss = _sinkhorn(
            w_obj, w_env, cost, sinkhorn_blur, sinkhorn_iters
        )
    else:
        loss = sinkhorn_divergence(
            gram_obj,
            gram_loc,
            blur=sinkhorn_blur,
            n_iter=sinkhorn_iters,
        )

    loss = loss / gram_obj.shape[1]
    if not torch.isfinite(loss):
        return latent_tokens.sum() * 0.0
    return loss


def encode_condition(ae, image: Image.Image, device):
    tensor = torchvision.transforms.ToTensor()(image.convert("RGB"))
    tensor = (2 * tensor - 1).to(device)
    with torch.no_grad():
        latent = ae.encode(tensor[None])[0].to(torch.bfloat16)
    _, latent_h, latent_w = latent.shape
    ids = torch.cartesian_prod(
        torch.zeros(1, dtype=torch.long),
        torch.arange(latent_h),
        torch.arange(latent_w),
        torch.zeros(1, dtype=torch.long),
    ).to(device)
    tokens = rearrange(latent, "c h w -> (h w) c").unsqueeze(0)
    return tokens, ids.unsqueeze(0)


def load_rgba_with_white_background(path: Path) -> Image.Image:
    image = Image.open(path)
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.getchannel("A"))
        return background
    return image.convert("RGB")


def denoise_with_mata_latent(
    model,
    img: torch.Tensor,
    img_ids: torch.Tensor,
    txt: torch.Tensor,
    txt_ids: torch.Tensor,
    timesteps: list[float],
    img_cond_seq: torch.Tensor,
    img_cond_seq_ids: torch.Tensor,
    obj_mask_full: torch.Tensor,
    env_local_full: torch.Tensor,
    env_sem_full: torch.Tensor | None,
    full_obj_mask: torch.Tensor,
    mata_scale: float,
    mata_iters: int,
    mata_start: float,
    mata_end: float,
    mata_stride: int,
    alpha_sem: float,
    patch_grid_sizes: list[int],
    min_patch_pixels: int,
    sinkhorn_blur: float,
    sinkhorn_iters: int,
    latent_h: int,
    latent_w: int,
    rollback_threshold: float = 1.5,
    log_every: int = 0,
) -> torch.Tensor:

    total_steps = len(timesteps) - 1
    in_window_count = 0
    obj_token_mask = (
        F.interpolate(
            full_obj_mask,
            size=(latent_h, latent_w),
            mode="nearest",
        )
        > 0.5
    ).flatten(2).squeeze(1).squeeze(0)

    for step_idx, (t_curr, t_prev) in enumerate(
        zip(timesteps[:-1], timesteps[1:])
    ):
        progress = step_idx / max(total_steps - 1, 1)
        in_window = mata_start <= progress <= mata_end
        active = (
            mata_scale > 0
            and mata_iters > 0
            and in_window
            and in_window_count % max(mata_stride, 1) == 0
            and obj_token_mask.any()
        )
        if in_window:
            in_window_count += 1

        timestep = torch.full(
            (img.shape[0],),
            t_curr,
            dtype=img.dtype,
            device=img.device,
        )
        model_img = torch.cat([img, img_cond_seq], dim=1)
        model_ids = torch.cat([img_ids, img_cond_seq_ids], dim=1)
        with torch.no_grad():
            pred = model(
                x=model_img,
                x_ids=model_ids,
                timesteps=timestep,
                ctx=txt,
                ctx_ids=txt_ids,
                guidance=None,
            )
            pred = pred[:, :img.shape[1]]

        if active:
            last_loss = None
            for _ in range(mata_iters):
                img_for_grad = img.detach().float().requires_grad_(True)
                loss = mata_loss_latent(
                    latent_tokens=img_for_grad,
                    obj_mask_t=obj_mask_full,
                    env_local_t=env_local_full,
                    env_sem_t=env_sem_full,
                    alpha_sem=alpha_sem,
                    patch_grid_sizes=patch_grid_sizes,
                    sinkhorn_blur=sinkhorn_blur,
                    sinkhorn_iters=sinkhorn_iters,
                    latent_h=latent_h,
                    latent_w=latent_w,
                    min_patch_pixels=min_patch_pixels,
                )
                if not torch.isfinite(loss) or loss.detach().item() == 0.0:
                    break

                grad = torch.autograd.grad(loss, img_for_grad)[0]
                grad = grad * obj_token_mask.view(1, -1, 1).to(grad.dtype)
                if not torch.isfinite(grad).all():
                    break

                abs_grad = grad.abs()
                nonzero = abs_grad[abs_grad > 0]
                if nonzero.numel() == 0:
                    break
                clip_value = torch.quantile(nonzero, 0.99).clamp_min(1e-6)
                grad = grad.clamp(min=-clip_value, max=clip_value)
                grad_max = grad.abs().max().clamp_min(1e-6)
                correction = mata_scale * grad / grad_max

                with torch.no_grad():
                    img_old = img.float()
                    img_new = img_old - correction
                    if rollback_threshold > 0:
                        norm_old = img_old[0, obj_token_mask].norm()
                        norm_new = img_new[0, obj_token_mask].norm()
                        ratio = (
                            norm_new / norm_old.clamp_min(1e-6)
                        ).item()
                        lower = 1.0 / rollback_threshold
                        if not lower < ratio < rollback_threshold:
                            if log_every > 0:
                                tqdm.write(
                                    f"  [latent MATA step {step_idx + 1}] "
                                    f"rolled back (norm ratio={ratio:.2f})"
                                )
                            break
                    img = img_new.to(dtype=img.dtype).detach()
                last_loss = loss.detach()

            if log_every > 0 and step_idx % log_every == 0:
                value = (
                    f"loss={last_loss.item():.4f}"
                    if last_loss is not None
                    else "skipped"
                )
                tqdm.write(
                    f"  [latent MATA step {step_idx + 1}/{total_steps}] {value}"
                )

        img = (
            img + (t_prev - t_curr) * pred.to(dtype=img.dtype)
        ).detach()
    return img


def generate_single(
    model,
    ae,
    text_encoder,
    cond_type_emb,
    bg_img: Image.Image,
    crop_img: Image.Image,
    obj_mask_pil: Image.Image,
    sem_mask_pil: Image.Image | None,
    prompt: str,
    device,
    num_steps: int,
    mata_kwargs: dict,
) -> Image.Image:
    bg_tokens, bg_ids = encode_condition(ae, bg_img, device)
    crop_tokens, crop_ids = encode_condition(ae, crop_img, device)
    bg_tokens = cond_type_emb(bg_tokens.float(), type_id=0).to(torch.bfloat16)
    crop_tokens = cond_type_emb(crop_tokens.float(), type_id=1).to(torch.bfloat16)

    text = text_encoder([prompt]).to(torch.bfloat16)
    text, text_ids = batched_prc_txt(text)

    latent_h, latent_w = LATENT_H, LATENT_W
    noise_ids = torch.cartesian_prod(
        torch.zeros(1, dtype=torch.long),
        torch.arange(latent_h),
        torch.arange(latent_w),
        torch.zeros(1, dtype=torch.long),
    ).to(device).unsqueeze(0)
    img = torch.randn(
        1,
        latent_h * latent_w,
        128,
        device=device,
        dtype=torch.bfloat16,
    )
    cond_tokens = torch.cat([bg_tokens, crop_tokens], dim=1)
    cond_ids = torch.cat([bg_ids, crop_ids], dim=1)

    obj_np = np.array(
        obj_mask_pil.resize(FIXED_SIZE, Image.NEAREST).convert("L")
    )
    obj_bin = (obj_np > 127).astype(np.uint8)
    sem_bin = None
    if sem_mask_pil is not None:
        sem_np = np.array(
            sem_mask_pil.resize(FIXED_SIZE, Image.NEAREST).convert("L")
        )
        sem_bin = (sem_np > 127).astype(np.uint8)

    env = build_environment_masks(
        obj_bin,
        local_dilation=mata_kwargs["local_dilation"],
        obj_erosion=mata_kwargs["obj_erosion"],
        sem_mask=sem_bin,
        alpha_sem=mata_kwargs["alpha_sem"],
    )
    obj_t = torch.from_numpy(env["obj_for_gram"]).float().to(device)
    loc_t = torch.from_numpy(env["local"]).float().to(device)
    obj_t = obj_t.view(1, 1, *FIXED_SIZE)
    loc_t = loc_t.view(1, 1, *FIXED_SIZE)
    sem_t = (
        torch.from_numpy(env["semantic"])
        .float()
        .to(device)
        .view(1, 1, *FIXED_SIZE)
        if env["semantic"] is not None
        else None
    )
    full_obj_t = (
        torch.from_numpy(obj_bin).float().to(device).view(1, 1, *FIXED_SIZE)
    )

    timesteps = get_schedule(num_steps, latent_h * latent_w)
    img = denoise_with_mata_latent(
        model=model,
        img=img,
        img_ids=noise_ids,
        txt=text,
        txt_ids=text_ids,
        timesteps=timesteps,
        img_cond_seq=cond_tokens,
        img_cond_seq_ids=cond_ids,
        obj_mask_full=obj_t,
        env_local_full=loc_t,
        env_sem_full=sem_t,
        full_obj_mask=full_obj_t,
        mata_scale=mata_kwargs["scale"],
        mata_iters=mata_kwargs["iters"],
        mata_start=mata_kwargs["start"],
        mata_end=mata_kwargs["end"],
        mata_stride=mata_kwargs["stride"],
        alpha_sem=env["alpha_sem"],
        patch_grid_sizes=mata_kwargs["patch_grid_sizes"],
        min_patch_pixels=mata_kwargs["min_patch_pixels"],
        sinkhorn_blur=mata_kwargs["sinkhorn_blur"],
        sinkhorn_iters=mata_kwargs["sinkhorn_iters"],
        latent_h=latent_h,
        latent_w=latent_w,
        rollback_threshold=mata_kwargs["rollback_threshold"],
        log_every=mata_kwargs["log_every"],
    )

    latent = rearrange(
        img[0], "(h w) c -> c h w", h=latent_h, w=latent_w
    )
    decoded = ae.decode(latent[None])[0].clamp(-1, 1)
    decoded = (decoded + 1) / 2
    return torchvision.transforms.ToPILImage()(decoded.float().cpu())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Insertion inference with latent-space MATA guidance"
    )
    parser.add_argument("--model_name", default="flux.2-klein-base-4b")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument(
        "--output_dir",
        default=str(GENERATION_ROOT / "output" / "inference_insertion_mata_latent"),
    )
    parser.add_argument("--num_steps", type=int, default=NUM_STEPS)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")

    parser.add_argument("--mata_scale", type=float, default=0.005)
    parser.add_argument("--mata_iters", type=int, default=1)
    parser.add_argument("--mata_start", type=float, default=0.7)
    parser.add_argument("--mata_end", type=float, default=0.9)
    parser.add_argument("--mata_stride", type=int, default=1)
    parser.add_argument("--num_scales", type=int, default=3)
    parser.add_argument(
        "--patch_grid_sizes",
        type=str,
        default="1,2,4",
        help="Comma-separated latent grid sizes; k means a k×k partition.",
    )
    parser.add_argument(
        "--min_patch_pixels",
        type=int,
        default=4,
        help="Minimum valid latent positions required for a patch Gram.",
    )
    parser.add_argument("--alpha_sem", type=float, default=0.3)
    parser.add_argument("--sinkhorn_blur", type=float, default=0.05)
    parser.add_argument("--sinkhorn_iters", type=int, default=50)
    parser.add_argument("--local_dilation", type=int, default=24)
    parser.add_argument("--obj_erosion", type=int, default=0)
    parser.add_argument("--rollback_threshold", type=float, default=1.5)
    parser.add_argument("--semantic_mask_dir", type=str, default=None)
    parser.add_argument("--log_every", type=int, default=0)

    parser.add_argument("--bg_folder", default="Background_Erased")
    parser.add_argument("--crop_folder", default="Crops")
    parser.add_argument("--mask_blend", action="store_true")
    parser.add_argument("--mask_blur_radius", type=int, default=5)
    args = parser.parse_args()

    if args.num_scales < 1:
        parser.error("--num_scales must be at least 1")
    if args.min_patch_pixels < 1:
        parser.error("--min_patch_pixels must be at least 1")
    if args.sinkhorn_iters < 1:
        parser.error("--sinkhorn_iters must be at least 1")
    if not 0 <= args.alpha_sem <= 1:
        parser.error("--alpha_sem must be in [0, 1]")
    try:
        grids = [
            int(value)
            for value in args.patch_grid_sizes.split(",")
            if value.strip()
        ]
    except ValueError:
        parser.error(
            "--patch_grid_sizes must be a comma-separated list of integers"
        )
    if any(grid < 1 for grid in grids):
        parser.error("--patch_grid_sizes values must all be at least 1")
    if len(grids) > args.num_scales:
        grids = grids[:args.num_scales]
    elif len(grids) < args.num_scales:
        grids += [grids[-1] if grids else 1] * (
            args.num_scales - len(grids)
        )
    args.patch_grid_sizes = grids
    return args


def _find(folder: Path, name: str):
    for extension in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
        path = folder / f"{name}{extension}"
        if path.exists():
            return path
    return None


def _validate_input_layout(args, data_root: Path):
    required = [
        data_root / "crop_info.csv",
        data_root / args.bg_folder,
        data_root / args.crop_folder,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if not (data_root / "Mask2").is_dir() and not (
        data_root / "Masks2"
    ).is_dir():
        missing.append(f"{data_root}/Mask2 or {data_root}/Masks2")
    if missing:
        raise FileNotFoundError(
            "Missing required inference inputs:\n  " + "\n  ".join(missing)
        )


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    _validate_input_layout(args, data_root)
    device = torch.device(args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    print(f"Using patch grid sizes: {args.patch_grid_sizes}")
    print(
        "Total patches per scale: "
        f"{[k * k for k in args.patch_grid_sizes]}"
    )
    if args.mata_scale <= 0:
        print("Latent MATA disabled; running baseline insertion inference.")
    else:
        print(
            "Latent MATA active: "
            f"scale={args.mata_scale}, min_patch_pixels="
            f"{args.min_patch_pixels}"
        )

    print(f"Loading model: {args.model_name}")
    text_encoder = load_text_encoder(args.model_name, device=device)
    ae = load_ae(args.model_name, device=device)
    model = load_flow_model(args.model_name, device=device)
    model = PeftModel.from_pretrained(model, args.checkpoint_dir)
    model.eval()

    cond_type_emb = ConditionTypeEmbedding(dim=128, num_types=2).to(device)
    embedding_path = os.path.join(args.checkpoint_dir, "cond_type_emb.pt")
    if os.path.exists(embedding_path):
        cond_type_emb.load_state_dict(
            torch.load(embedding_path, map_location=device)
        )
        print(f"Loaded cond_type_emb from {embedding_path}")
    else:
        print(f"WARNING: {embedding_path} missing; using random init.")
    cond_type_emb.eval()

    semantic_dir = (
        Path(args.semantic_mask_dir) if args.semantic_mask_dir else None
    )
    if semantic_dir is not None and not semantic_dir.exists():
        print(
            f"WARNING: semantic mask directory {semantic_dir} is missing; "
            "semantic reference will be disabled."
        )
        semantic_dir = None

    mata_kwargs = {
        "scale": args.mata_scale,
        "iters": args.mata_iters,
        "start": args.mata_start,
        "end": args.mata_end,
        "stride": args.mata_stride,
        "alpha_sem": args.alpha_sem,
        "patch_grid_sizes": args.patch_grid_sizes,
        "min_patch_pixels": args.min_patch_pixels,
        "sinkhorn_blur": args.sinkhorn_blur,
        "sinkhorn_iters": args.sinkhorn_iters,
        "local_dilation": args.local_dilation,
        "obj_erosion": args.obj_erosion,
        "rollback_threshold": args.rollback_threshold,
        "log_every": args.log_every,
    }

    with open(data_root / "crop_info.csv", newline="") as csv_file:
        records = [
            {key.strip(): value.strip() for key, value in row.items()}
            for row in csv.DictReader(csv_file)
        ]
    print(f"Total test samples: {len(records)}")

    output_dir = Path(args.output_dir)
    condition_dir = output_dir / "conditions"
    output_dir.mkdir(parents=True, exist_ok=True)
    condition_dir.mkdir(parents=True, exist_ok=True)

    for record in tqdm(records, desc="Generating"):
        name = record["output_name"]
        category = record["category"].replace("-", " ").lower()
        prompt = f"Place a {category} at the specified position"
        bg_path = _find(data_root / args.bg_folder, name)
        crop_path = _find(data_root / args.crop_folder, name)
        mask_path = _find(data_root / "Mask2", name) or _find(
            data_root / "Masks2", name
        )
        if bg_path is None or crop_path is None or mask_path is None:
            missing = []
            if bg_path is None:
                missing.append(f"{args.bg_folder}/{name}")
            if crop_path is None:
                missing.append(f"{args.crop_folder}/{name}")
            if mask_path is None:
                missing.append(f"Mask2|Masks2/{name}")
            tqdm.write(f"Skipping {name}: missing {', '.join(missing)}")
            continue

        bg_img = Image.open(bg_path).convert("RGB").resize(
            FIXED_SIZE, Image.BILINEAR
        )
        crop_img = load_rgba_with_white_background(crop_path).resize(
            FIXED_SIZE, Image.BILINEAR
        )
        mask_pil = Image.open(mask_path).convert("L")
        semantic_pil = None
        if semantic_dir is not None:
            semantic_path = _find(semantic_dir, name)
            if semantic_path is not None:
                semantic_pil = Image.open(semantic_path).convert("L")

        try:
            result = generate_single(
                model,
                ae,
                text_encoder,
                cond_type_emb,
                bg_img,
                crop_img,
                mask_pil,
                semantic_pil,
                prompt,
                device,
                num_steps=args.num_steps,
                mata_kwargs=mata_kwargs,
            )
        except Exception as error:
            import traceback

            traceback.print_exc()
            tqdm.write(f"[skip {name}] {error}")
            continue

        if args.mask_blend:
            blend_mask = mask_pil.resize(FIXED_SIZE, Image.NEAREST)
            if args.mask_blur_radius > 0:
                from PIL import ImageFilter

                blend_mask = blend_mask.filter(
                    ImageFilter.GaussianBlur(args.mask_blur_radius)
                )
            result = Image.composite(result, bg_img, blend_mask)

        result.save(output_dir / f"{name}.png")
        bg_img.save(condition_dir / f"{name}_background.png")
        crop_img.save(condition_dir / f"{name}_subject.png")
        mask_pil.resize(FIXED_SIZE, Image.NEAREST).save(
            condition_dir / f"{name}_mask.png"
        )
        original_path = _find(data_root / "Original", name)
        if original_path is not None:
            Image.open(original_path).convert("RGB").resize(
                FIXED_SIZE, Image.BILINEAR
            ).save(condition_dir / f"{name}_original.png")

    print(f"\nDone. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
