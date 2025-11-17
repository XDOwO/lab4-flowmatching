"""
Resume Training Script for Rectified Flow (Task 3)

Resumes training a rectified flow model from a saved checkpoint.

Usage:
    python task3_rectified_flow/resume_train_rectified.py \
        --resume_from_ckpt <path_to_checkpoint> \
        --gpu <gpu_id>
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from dotmap import DotMap
from pytorch_lightning import seed_everything
from tqdm import tqdm

sys.path.append('.')
from image_common.dataset import tensor_to_pil_image
from image_common.fm import FlowMatching, FMScheduler
from image_common.network import UNet

from task3_rectified_flow.reflow_dataset import (
    ReflowDataset,
    get_reflow_data_iterator,
)

matplotlib.use("Agg")


def get_current_time():
    now = datetime.now().strftime("%m-%d-%H%M%S")
    return now


def main(args):
    """config"""
    # Load config from the checkpoint's directory to ensure consistency
    ckpt_path = Path(args.resume_from_ckpt)
    save_dir = ckpt_path.parent
    config_path = save_dir / "config.json"

    with open(config_path, "r") as f:
        config = DotMap(json.load(f))
    
    # Update config with any new command-line args, like GPU
    config.update(vars(args))
    config.device = f"cuda:{args.gpu}"

    print(f"Resuming training. Checkpoints will be saved in: {save_dir}")

    seed_everything(config.seed)

    """######"""

    # Load reflow dataset
    print(f"Loading reflow dataset from {config.reflow_data_path}")
    reflow_dataset = ReflowDataset(
        config.reflow_data_path,
        use_cfg=config.use_cfg
    )

    train_dl = torch.utils.data.DataLoader(
        reflow_dataset,
        batch_size=config.batch_size,
        num_workers=4,
        shuffle=True,
        drop_last=True,
    )
    train_it = get_reflow_data_iterator(train_dl)

    # Get image resolution and num_classes from dataset metadata
    image_resolution = reflow_dataset.metadata.get("image_resolution", 64)
    num_classes = reflow_dataset.metadata.get("num_classes", None)

    print(f"Image resolution: {image_resolution}")
    if config.use_cfg:
        print(f"Number of classes: {num_classes}")

    # Set up the scheduler (same as base FM)
    fm_scheduler = FMScheduler(sigma_min=config.sigma_min)

    # Initialize network (same architecture as base FM)
    network = UNet(
        image_resolution=image_resolution,
        ch=128,
        ch_mult=[1, 2, 2, 2],
        attn=[1],
        num_res_blocks=4,
        dropout=0.1,
        use_cfg=config.use_cfg,
        cfg_dropout=config.cfg_dropout,
        num_classes=num_classes,
    )

    fm = FlowMatching(network, fm_scheduler)
    fm = fm.to(config.device)

    # Same optimizer and scheduler as base FM
    optimizer = torch.optim.Adam(fm.network.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda t: min((t + 1) / config.warmup_steps, 1.0)
    )

    step = 0
    losses = []
    
    print(f"Resuming training from checkpoint: {args.resume_from_ckpt}")
    ckpt_data = torch.load(args.resume_from_ckpt, map_location=config.device)

    # Check for different checkpoint formats
    if "state_dict" in ckpt_data:
        fm.load_state_dict(ckpt_data["state_dict"])
    else:
        # Handle older checkpoints that only saved the model
        fm.load_state_dict(ckpt_data)
        print("Warning: This appears to be an old checkpoint format. Only model weights are loaded.")

    # Load optimizer state if it exists
    if "optimizer_state_dict" in ckpt_data:
        optimizer.load_state_dict(ckpt_data["optimizer_state_dict"])
    else:
        print("Warning: Optimizer state not found in checkpoint. Optimizer will be re-initialized.")

    # Load step count if it exists
    if "step" in ckpt_data:
        step = ckpt_data["step"]
    else:
        print("Warning: Step count not found in checkpoint. Starting from step 0.")

    # Override start step if provided
    if args.start_step is not None:
        step = args.start_step
        print(f"Overriding start step to {step}.")

    # Manually advance the scheduler to the correct step
    for _ in range(step):
        scheduler.step()
    
    print(f"Resumed from step {step}.")

    print(f"Continuing training up to {config.train_num_steps} steps...")
    print(f"This is {config.reflow_iteration}-rectified flow training")

    with tqdm(initial=step, total=config.train_num_steps) as pbar:
        while step < config.train_num_steps:
            if step % config.log_interval == 0:
                fm.eval()
                # Plot loss curve
                plt.plot(losses)
                plt.savefig(f"{save_dir}/loss.png")
                plt.close()

                # Generate sample images
                shape = (4, 3, fm.image_resolution, fm.image_resolution)
                if config.use_cfg:
                    class_label = torch.tensor([1, 1, 2, 3]).to(config.device)
                    samples = fm.sample(
                        shape,
                        class_label=class_label,
                        guidance_scale=7.5,
                        num_inference_timesteps=20,
                        verbose=False
                    )
                else:
                    samples = fm.sample(shape, return_traj=False, verbose=False)

                pil_images = tensor_to_pil_image(samples)
                for i, img in enumerate(pil_images):
                    img.save(save_dir / f"step={step}-{i}.png")

                # Save checkpoint with optimizer and step
                torch.save({
                    'step': step,
                    'state_dict': fm.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f"{save_dir}/last.ckpt")
                fm.train()

            # Load batch from reflow dataset
            if config.use_cfg:
                x_0, z_1, label = next(train_it)
                x_0, z_1, label = x_0.to(config.device), z_1.to(config.device), label.to(config.device)
            else:
                x_0, z_1 = next(train_it)
                x_0, z_1 = x_0.to(config.device), z_1.to(config.device)
                label = None

            if config.use_cfg:
                loss = fm.get_loss(z_1, class_label=label, x0=x_0)
            else:
                loss = fm.get_loss(z_1, x0=x_0)

            pbar.set_description(f"Loss: {loss.item():.4f}")

            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

            step += 1
            pbar.update(1)

    print(f"Training completed! Final checkpoint saved at {save_dir}/last.ckpt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume Rectified Flow model training")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID to use for resumed training.")
    parser.add_argument("--resume_from_ckpt", type=str, required=True, help="Path to the checkpoint to resume from.")
    parser.add_argument("--start_step", type=int, default=None, help="Optional: Override the starting step number.")
    args = parser.parse_args()
    main(args)
