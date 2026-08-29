

import argparse
import csv
import os
import random
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
from einops import rearrange
from peft import LoraConfig, get_peft_model
from PIL import Image, ImageFilter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

GENERATION_ROOT = Path(__file__).resolve().parents[2]

from flux2.sampling import batched_prc_txt, default_prep
from flux2.util import load_ae, load_flow_model, load_text_encoder





FIXED_SIZE = (512, 512)

LORA_TARGET_MODULES = [

    "img_attn.qkv", "img_attn.proj",
    "txt_attn.qkv", "txt_attn.proj",

    "img_mlp.0", "img_mlp.2",
    "txt_mlp.0", "txt_mlp.2",

    "linear1", "linear2",
]





class ConditionTypeEmbedding(nn.Module):


    def __init__(self, dim: int = 128, num_types: int = 2):
        super().__init__()
        self.emb = nn.Embedding(num_types, dim)

    def forward(self, tokens: torch.Tensor, type_id: int) -> torch.Tensor:

        bias = self.emb(torch.tensor(type_id, device=tokens.device))
        return tokens + bias.unsqueeze(0).unsqueeze(0)






class InsertionDataset(Dataset):


    def __init__(
        self,
        data_root: str,
        split_names: list[str] | None = None,
        augmentation_prob: float = 0.5,
        rotation_prob: float = 0.5,
        flip_prob: float = 0.5,
        color_jitter_prob: float = 0.5,
        background_blur_prob: float = 0.3,
        brightness_range: tuple[float, float] = (0.7, 1.3),
        contrast_range: tuple[float, float] = (0.75, 1.25),
        saturation_range: tuple[float, float] = (0.8, 1.2),
        hue_range: tuple[float, float] = (-0.05, 0.05),
    ):
        self.root = Path(data_root)
        self.augmentation_prob = augmentation_prob
        self.rotation_prob = rotation_prob
        self.flip_prob = flip_prob
        self.color_jitter_prob = color_jitter_prob
        self.background_blur_prob = background_blur_prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range


        csv_path = self.root / "crop_info.csv"
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)

            records = [{k.strip(): v.strip() for k, v in row.items()} for row in reader]


        if split_names is not None:
            name_set = set(split_names)
            records = [r for r in records if r["output_name"] in name_set]


        self.samples = []
        for rec in records:
            name = rec["output_name"]
            category = rec["category"]


            orig_file = self._find_file("Original", name)
            bg_file = self._find_file("Background_Erased", name)
            crop_file = self._find_file_in_folders(["Crops_Blurred", "Crops"], name)

            if orig_file and bg_file and crop_file:
                mask_file = self._find_file("Masks2", name)
                self.samples.append({
                    "original": orig_file,
                    "background": bg_file,
                    "crop": crop_file,
                    "mask": mask_file,
                    "category": category,
                    "name": name,
                })

    def _find_file(self, folder: str, name: str) -> Path | None:

        for ext in [".png", ".jpg"]:
            path = self.root / folder / f"{name}{ext}"
            if path.exists():
                return path
        return None

    def _find_file_in_folders(self, folders: list[str], name: str) -> Path | None:

        for folder in folders:
            path = self._find_file(folder, name)
            if path:
                return path
        return None

    def __len__(self):
        return len(self.samples)

    def _load_rgba_with_white_background(self, path: Path) -> Image.Image:

        img = Image.open(path)
        if img.mode == "RGBA":
            white_bg = Image.new("RGB", img.size, (255, 255, 255))
            white_bg.paste(img, mask=img.split()[3])
            return white_bg
        return img.convert("RGB")

    def _apply_sync_transforms(self, *images):


        if random.random() < self.rotation_prob:
            angle = random.choice([0, 90, 180, 270])
            images = tuple(img.rotate(angle, expand=False) for img in images)


        if random.random() < self.flip_prob:
            if random.random() < 0.5:
                images = tuple(img.transpose(Image.FLIP_LEFT_RIGHT) for img in images)
            if random.random() < 0.5:
                images = tuple(img.transpose(Image.FLIP_TOP_BOTTOM) for img in images)

        return images

    def _apply_color_jitter(self, img: Image.Image, brightness, contrast, saturation, hue):

        import torchvision.transforms.functional as TF
        img = TF.adjust_brightness(img, brightness)
        img = TF.adjust_contrast(img, contrast)
        img = TF.adjust_saturation(img, saturation)
        img = TF.adjust_hue(img, hue)
        return img

    def _apply_random_blur(self, img: Image.Image, kernel_size: int):

        from PIL import ImageFilter
        return img.filter(ImageFilter.GaussianBlur(radius=kernel_size // 2))

    def __getitem__(self, idx):
        sample = self.samples[idx]

        try:

            target = Image.open(sample["original"]).convert("RGB")
            bg = Image.open(sample["background"]).convert("RGB")
            crop = self._load_rgba_with_white_background(sample["crop"])


            if sample["mask"] is not None:
                mask = Image.open(sample["mask"]).convert("L")
            else:
                mask = None


            target = target.resize(FIXED_SIZE, Image.BILINEAR)
            bg = bg.resize(FIXED_SIZE, Image.BILINEAR)
            crop = crop.resize(FIXED_SIZE, Image.BILINEAR)
            if mask is not None:
                mask = mask.resize(FIXED_SIZE, Image.NEAREST)


            if random.random() < self.augmentation_prob:
                if mask is not None:
                    target, bg, crop, mask = self._apply_sync_transforms(target, bg, crop, mask)
                else:
                    target, bg, crop = self._apply_sync_transforms(target, bg, crop)


            if random.random() < self.color_jitter_prob:
                brightness = random.uniform(*self.brightness_range)
                contrast = random.uniform(*self.contrast_range)
                saturation = random.uniform(*self.saturation_range)
                hue = random.uniform(*self.hue_range)

                target = self._apply_color_jitter(target, brightness, contrast, saturation, hue)
                bg = self._apply_color_jitter(bg, brightness, contrast, saturation, hue)


            if random.random() < self.background_blur_prob:
                kernel_size = random.choice([3, 5, 7])
                bg = self._apply_random_blur(bg, kernel_size)
                if random.random() < 0.5:
                    target = self._apply_random_blur(target, kernel_size)

            category = sample["category"].replace("-", " ").lower()
            prompt = f"Place a {category} at the specified position"

            return bg, crop, target, prompt, mask

        except Exception as e:
            print(f"Error loading sample {idx} ({sample['name']}): {e}")

            return self.__getitem__(random.randint(0, len(self) - 1))


def collate_fn(batch):
    bgs, crops, targets, prompts, masks = zip(*batch)
    return list(bgs), list(crops), list(targets), list(prompts), list(masks)






def encode_condition(ae, img: Image.Image, device):

    t = torchvision.transforms.ToTensor()(img.convert("RGB"))
    t = (2 * t - 1).to(device)
    with torch.no_grad():
        latent = ae.encode(t[None])[0].to(torch.bfloat16)
    _, lh, lw = latent.shape
    ids = torch.cartesian_prod(
        torch.zeros(1, dtype=torch.long),
        torch.arange(lh),
        torch.arange(lw),
        torch.zeros(1, dtype=torch.long),
    ).to(device)
    tokens = rearrange(latent, "c h w -> (h w) c").unsqueeze(0)
    return tokens, ids.unsqueeze(0)






def training_step(model, ae, text_encoder, batch, device, cond_type_emb, cfg_guidance=4.0, mask_loss_weight=5.0):
    bgs, crops, targets, prompts, masks = batch
    B = len(targets)

    with torch.no_grad():

        bg_tokens_list, bg_ids_list = [], []
        crop_tokens_list, crop_ids_list = [], []
        for bg, crop in zip(bgs, crops):
            bt, bi = encode_condition(ae, bg, device)
            ct, ci = encode_condition(ae, crop, device)
            bg_tokens_list.append(bt)
            bg_ids_list.append(bi)
            crop_tokens_list.append(ct)
            crop_ids_list.append(ci)

        bg_tokens = torch.cat(bg_tokens_list, dim=0)
        bg_ids = torch.cat(bg_ids_list, dim=0)
        crop_tokens = torch.cat(crop_tokens_list, dim=0)
        crop_ids = torch.cat(crop_ids_list, dim=0)


        target_tensors = default_prep(targets, limit_pixels=512**2)
        if not isinstance(target_tensors, list):
            target_tensors = [target_tensors]
        target_latents = [ae.encode(t[None].to(device))[0].to(torch.bfloat16) for t in target_tensors]


        ctx_cond = text_encoder(list(prompts)).to(torch.bfloat16)
        ctx_cond_proc, ctx_cond_ids = batched_prc_txt(ctx_cond)


        lh = lw = FIXED_SIZE[0] // 16
        mask_weights = None
        if mask_loss_weight > 1.0 and any(m is not None for m in masks):
            mask_weights_list = []
            for m in masks:
                if m is not None:

                    m_resized = m.resize((lw, lh), Image.NEAREST)
                    m_tensor = torch.tensor(
                        [1.0 if p > 127 else 0.0 for p in m_resized.getdata()],
                        device=device, dtype=torch.bfloat16
                    ).reshape(1, lh * lw, 1)

                    w = 1.0 + (mask_loss_weight - 1.0) * m_tensor
                    mask_weights_list.append(w)
                else:
                    mask_weights_list.append(torch.ones(1, lh * lw, 1, device=device, dtype=torch.bfloat16))
            mask_weights = torch.cat(mask_weights_list, dim=0)


    bg_tokens = cond_type_emb(bg_tokens.float(), type_id=0).to(torch.bfloat16)
    crop_tokens = cond_type_emb(crop_tokens.float(), type_id=1).to(torch.bfloat16)


    noise_ids_single = torch.cartesian_prod(
        torch.zeros(1, dtype=torch.long),
        torch.arange(lh),
        torch.arange(lw),
        torch.zeros(1, dtype=torch.long),
    ).to(device)

    x = torch.stack(target_latents).to(torch.bfloat16)
    x_flat = rearrange(x, "b c h w -> b (h w) c")
    x_ids = noise_ids_single.unsqueeze(0).expand(B, -1, -1)

    t = torch.rand(B, device=device, dtype=torch.bfloat16)
    noise = torch.randn_like(x_flat)
    t_exp = t[:, None, None]
    x_noisy = (1 - t_exp) * x_flat + t_exp * noise
    target_velocity = noise - x_flat


    x_input = torch.cat([x_noisy, bg_tokens, crop_tokens], dim=1)
    x_input_ids = torch.cat([x_ids, bg_ids, crop_ids], dim=1)

    raw_model = model.module if hasattr(model, "module") else model
    pred = raw_model(
        x=x_input, x_ids=x_input_ids,
        timesteps=t, ctx=ctx_cond_proc, ctx_ids=ctx_cond_ids, guidance=None,
    )

    pred = pred[:, : x_noisy.shape[1]]


    if mask_weights is not None:

        loss = (mask_weights * (pred.float() - target_velocity.float()) ** 2).mean()
    else:
        loss = nn.functional.mse_loss(pred.float(), target_velocity.float())

    return loss






def load_split_names(data_root: str, val_ratio: float = 0.05, seed: int = 42):
    import random
    csv_path = Path(data_root) / "crop_info.csv"
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)

        names = []
        for row in reader:
            row_clean = {k.strip(): v.strip() for k, v in row.items()}
            names.append(row_clean["output_name"])
    random.seed(seed)
    random.shuffle(names)
    n_val = max(1, int(len(names) * val_ratio))
    return names[n_val:], names[:n_val]






def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="flux.2-klein-base-4b")
    p.add_argument("--data_roots", nargs="+",
                   default=[str(GENERATION_ROOT / "data" / "ISAID_DOTA_processed"),
                            str(GENERATION_ROOT / "data" / "samars")],
                   help="One or more dataset roots, each with Background_Erased/Crops/Original/crop_info.csv")
    p.add_argument("--output_dir", default=str(GENERATION_ROOT / "checkpoints" / "flux2-lora-insertion"))
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_ratio", type=float, default=0.05)
    p.add_argument("--cfg_guidance", type=float, default=4.0)
    p.add_argument("--mask_loss_weight", type=float, default=5.0,
                   help="Weight multiplier for insertion region loss. Set to 1.0 to disable mask weighting.")
    p.add_argument("--wandb_project", type=str, default="flux2-lora-insertion")
    p.add_argument("--wandb_run_name", type=str, default="insertion-dual-cond")
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args()






def main():
    args = parse_args()

    use_ddp = "LOCAL_RANK" in os.environ
    if use_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        device = torch.device("cuda")

    is_main = local_rank == 0

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)

    use_wandb = is_main and not args.no_wandb
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    if is_main:
        print(f"Loading model: {args.model_name}")

    text_encoder = load_text_encoder(args.model_name, device=device)
    ae = load_ae(args.model_name, device=device)
    model = load_flow_model(args.model_name, device=device)

    for p in model.parameters():
        p.requires_grad_(False)

    lora_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha,
        target_modules=LORA_TARGET_MODULES, lora_dropout=0.0, bias="none",
    )
    model = get_peft_model(model, lora_config)
    if is_main:
        model.print_trainable_parameters()

    text_encoder.eval()
    ae.eval()
    model.train()

    cond_type_emb = ConditionTypeEmbedding(dim=128, num_types=2).to(device)
    cond_type_emb.train()

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    trainable_params = (
        list(filter(lambda p: p.requires_grad, model.parameters()))
        + list(cond_type_emb.parameters())
    )
    try:
        from prodigyopt import Prodigy
    except ImportError:
        raise ImportError("Please install prodigyopt: pip install prodigyopt")

    optimizer = Prodigy(
        trainable_params,
        lr=1.0,
        weight_decay=1e-4,
        use_bias_correction=True,
        safeguard_warmup=True,
    )
    scaler = torch.amp.GradScaler("cuda")


    from torch.utils.data import ConcatDataset
    train_datasets = []
    for root in args.data_roots:
        ds = InsertionDataset(
            root, split_names=None,
            augmentation_prob=0.5,
            rotation_prob=0.5,
            flip_prob=0.5,
            color_jitter_prob=0.5,
            background_blur_prob=0.3,
            brightness_range=(0.7, 1.3),
            contrast_range=(0.75, 1.25),
            saturation_range=(0.8, 1.2),
            hue_range=(-0.05, 0.05),
        )
        train_datasets.append(ds)
        if is_main:
            print(f"  {root}: {len(ds)} samples")

    train_dataset = ConcatDataset(train_datasets)

    sampler = DistributedSampler(train_dataset, shuffle=True) if use_ddp else None
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(sampler is None),
        sampler=sampler, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    if is_main:
        print(f"Total training samples: {len(train_dataset)}")
        if use_ddp:
            print(f"Using {dist.get_world_size()} GPUs")


    if is_main:
        vis_dir = os.path.join(args.output_dir, "dataloader_vis")
        os.makedirs(vis_dir, exist_ok=True)
        print(f"Saving dataloader visualization to {vis_dir} ...")
        vis_loader = DataLoader(
            train_dataset, batch_size=1, shuffle=True,
            num_workers=0, collate_fn=collate_fn,
        )
        for i, (bgs, crops, targets, prompts, masks) in enumerate(vis_loader):
            if i >= 8:
                break
            bg, crop, target, prompt = bgs[0], crops[0], targets[0], prompts[0]

            w, h = bg.size
            canvas = Image.new("RGB", (w * 3, h))
            canvas.paste(bg, (0, 0))
            canvas.paste(crop, (w, 0))
            canvas.paste(target, (w * 2, 0))
            canvas.save(os.path.join(vis_dir, f"sample_{i:03d}.png"))
            print(f"  [{i}] prompt: {prompt}")
        print(f"Saved {min(8, len(train_dataset))} visualization samples.")

    global_step = 0
    for epoch in range(args.num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            try:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    loss = training_step(
                        model, ae, text_encoder, batch, device,
                        cond_type_emb, cfg_guidance=args.cfg_guidance,
                        mask_loss_weight=args.mask_loss_weight,
                    )
            except Exception as e:
                import traceback
                if is_main:
                    traceback.print_exc()
                    print(f"[step {global_step}] Skipping: {e}")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1

            if is_main and global_step % args.log_every == 0:
                loss_val = loss.item()
                print(f"[epoch {epoch+1} | step {global_step}] loss={loss_val:.4f}")
                if use_wandb:
                    wandb.log({"train/loss": loss_val, "epoch": epoch + 1}, step=global_step)

            if is_main and global_step % args.save_every == 0:
                raw = model.module if hasattr(model, "module") else model
                ckpt_path = os.path.join(args.output_dir, f"lora_step{global_step}")
                raw.save_pretrained(ckpt_path)
                torch.save(cond_type_emb.state_dict(), os.path.join(ckpt_path, "cond_type_emb.pt"))
                print(f"Saved: {ckpt_path}")

        if is_main:
            raw = model.module if hasattr(model, "module") else model
            ckpt_path = os.path.join(args.output_dir, f"lora_epoch{epoch+1}")
            raw.save_pretrained(ckpt_path)
            torch.save(cond_type_emb.state_dict(), os.path.join(ckpt_path, "cond_type_emb.pt"))
            print(f"Epoch {epoch+1} done. Saved: {ckpt_path}")
            if use_wandb:
                wandb.log({"epoch": epoch + 1}, step=global_step)

    if is_main:
        raw = model.module if hasattr(model, "module") else model
        final_path = os.path.join(args.output_dir, "lora_final")
        raw.save_pretrained(final_path)
        torch.save(cond_type_emb.state_dict(), os.path.join(final_path, "cond_type_emb.pt"))
        print("Training complete.")
        if use_wandb:
            wandb.finish()

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
