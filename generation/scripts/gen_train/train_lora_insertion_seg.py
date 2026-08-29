

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


from train_lora_insertion import *
from train_lora_insertion import InsertionDataset, parse_args






class InsertionDatasetSeg(InsertionDataset):


    def __init__(self, data_root, split_names=None, **kwargs):


        import torch.utils.data as tud
        tud.Dataset.__init__(self)

        import csv
        self.root = Path(data_root)


        self.augmentation_prob = kwargs.get("augmentation_prob", 0.5)
        self.rotation_prob = kwargs.get("rotation_prob", 0.5)
        self.flip_prob = kwargs.get("flip_prob", 0.5)
        self.color_jitter_prob = kwargs.get("color_jitter_prob", 0.5)
        self.background_blur_prob = kwargs.get("background_blur_prob", 0.3)
        self.brightness_range = kwargs.get("brightness_range", (0.7, 1.3))
        self.contrast_range = kwargs.get("contrast_range", (0.75, 1.25))
        self.saturation_range = kwargs.get("saturation_range", (0.8, 1.2))
        self.hue_range = kwargs.get("hue_range", (-0.05, 0.05))

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
            bg_file = self._find_file("Background_Erased_WithSeg", name)
            crop_file = self._find_file("Crops_Blurred", name)

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






def main():
    import os
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from torch.utils.data import ConcatDataset, DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from peft import LoraConfig, get_peft_model
    from torch.nn.parallel import DistributedDataParallel as DDP
    from PIL import Image

    from flux2.util import load_ae, load_flow_model, load_text_encoder
    from train_lora_insertion import (
        LORA_TARGET_MODULES,
        ConditionTypeEmbedding,
        collate_fn,
        training_step,
    )

    args = parse_args()

    if args.output_dir == str(GENERATION_ROOT / "checkpoints" / "flux2-lora-insertion"):
        args.output_dir = str(GENERATION_ROOT / "checkpoints" / "flux2-lora-insertion-seg")

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
        wandb.init(project=args.wandb_project,
                   name=args.wandb_run_name or "insertion-seg",
                   config=vars(args))

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
        optimizer = Prodigy(trainable_params, lr=1.0, weight_decay=1e-4,
                           use_bias_correction=True, safeguard_warmup=True)
    except ImportError:
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr,
                                      weight_decay=1e-4)

    scaler = torch.amp.GradScaler("cuda")


    train_datasets = []
    for root in args.data_roots:
        ds = InsertionDatasetSeg(root, split_names=None)
        train_datasets.append(ds)
        if is_main:
            print(f"  {root}: {len(ds)} samples")

    train_dataset = ConcatDataset(train_datasets)
    sampler = DistributedSampler(train_dataset, shuffle=True) if use_ddp else None
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(sampler is None),
        sampler=sampler, num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True,
    )

    if is_main:
        print(f"Total training samples: {len(train_dataset)}")
        if use_ddp:
            print(f"Using {dist.get_world_size()} GPUs")


    if is_main:
        vis_dir = os.path.join(args.output_dir, "dataloader_vis")
        os.makedirs(vis_dir, exist_ok=True)
        vis_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                                num_workers=0, collate_fn=collate_fn)
        for i, (bgs, crops, targets, prompts, masks) in enumerate(vis_loader):
            if i >= 8:
                break
            bg, crop, target = bgs[0], crops[0], targets[0]
            w, h = bg.size
            canvas = Image.new("RGB", (w * 3, h))
            canvas.paste(bg, (0, 0))
            canvas.paste(crop, (w, 0))
            canvas.paste(target, (w * 2, 0))
            canvas.save(os.path.join(vis_dir, f"sample_{i:03d}.png"))
        print("Saved 8 visualization samples.")


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
                print(f"[epoch {epoch+1} | step {global_step}] loss={loss.item():.4f}")
                if use_wandb:
                    wandb.log({"train/loss": loss.item(), "epoch": epoch + 1},
                              step=global_step)

            if is_main and global_step % args.save_every == 0:
                raw = model.module if hasattr(model, "module") else model
                ckpt = os.path.join(args.output_dir, f"lora_step{global_step}")
                raw.save_pretrained(ckpt)
                torch.save(cond_type_emb.state_dict(),
                           os.path.join(ckpt, "cond_type_emb.pt"))
                print(f"Saved: {ckpt}")

        if is_main:
            raw = model.module if hasattr(model, "module") else model
            ckpt = os.path.join(args.output_dir, f"lora_epoch{epoch+1}")
            raw.save_pretrained(ckpt)
            torch.save(cond_type_emb.state_dict(),
                       os.path.join(ckpt, "cond_type_emb.pt"))
            print(f"Epoch {epoch+1} done. Saved: {ckpt}")
            if use_wandb:
                wandb.log({"epoch": epoch + 1}, step=global_step)

    if is_main:
        raw = model.module if hasattr(model, "module") else model
        final = os.path.join(args.output_dir, "lora_final")
        raw.save_pretrained(final)
        torch.save(cond_type_emb.state_dict(),
                   os.path.join(final, "cond_type_emb.pt"))
        print("Training complete.")
        if use_wandb:
            wandb.finish()

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
