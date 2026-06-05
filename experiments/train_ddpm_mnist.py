"""Train a DDPM (UNet) on MNIST.

Produces the checkpoint used downstream by ``compute_hessian_mnist.py`` and the
``image_results.ipynb`` visualization. Uses the in-repo DDPM modules under
``models/ddpm/core/ddpm_torch`` (no external dependencies).

Example
-------
    python experiments/train_ddpm_mnist.py \
        --data-root ./data/mnist \
        --chkpt-dir ./checkpoints/mnist \
        --epochs 50 --device cuda:0
"""
import argparse
import math
import os
import sys

import torch
import torchvision
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

# Make the repo root importable so `models...` resolves regardless of CWD.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.ddpm.core.ddpm_torch.utils import seed_all
from models.ddpm.core.ddpm_torch.diffusion import GaussianDiffusion, get_beta_schedule
from models.ddpm.core.ddpm_torch.models.unet import UNet


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total:,}")
    return total


@torch.no_grad()
def sample_and_save(model, diffusion, device, out_dir, epoch, num_samples=64,
                    apply_clamp=True, fixed_noise=None):
    """Sample a grid of images and save it as a PNG for visual monitoring."""
    was_training = model.training
    model.eval()

    samples = diffusion.p_sample(
        denoise_fn=model, shape=(num_samples, 1, 28, 28),
        device=device, noise=fixed_noise, seed=None,
    )
    if apply_clamp:
        samples = torch.clamp(samples, -1.0, 1.0)
    vis = torch.clamp((samples + 1.0) / 2.0, 0.0, 1.0)

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"mnist_samples_epoch{epoch:04d}.png")
    save_image(make_grid(vis, nrow=int(math.sqrt(num_samples))), save_path)
    print(f"[Sample] saved: {save_path}")

    if was_training:
        model.train()


def build_argparser():
    p = argparse.ArgumentParser(description="Train a DDPM UNet on MNIST.")

    # data / io
    p.add_argument("--data-root", default="./data/mnist", type=str)
    p.add_argument("--chkpt-dir", default="./checkpoints/mnist_small_timesteps", type=str)
    p.add_argument("--sample-dir", default="./samples/mnist", type=str)

    # training
    p.add_argument("--epochs", default=50, type=int)
    p.add_argument("--batch-size", default=128, type=int)
    p.add_argument("--lr", default=2e-4, type=float)
    p.add_argument("--beta1", default=0.9, type=float)
    p.add_argument("--beta2", default=0.999, type=float)
    p.add_argument("--seed", default=1234, type=int)
    p.add_argument("--device", default="cuda:0", type=str)
    p.add_argument("--num-workers", default=4, type=int)
    p.add_argument("--resume", action="store_true")

    # diffusion (defaults match the "small timesteps" MNIST setup)
    p.add_argument("--timesteps", default=500, type=int)
    p.add_argument("--beta-schedule",
                   choices=["quad", "linear", "warmup10", "warmup50", "jsd"], default="linear")
    p.add_argument("--beta-start", default=0.001, type=float)
    p.add_argument("--beta-end", default=0.2, type=float)
    p.add_argument("--model-mean-type", choices=["mean", "x_0", "eps"], default="eps")
    p.add_argument("--model-var-type",
                   choices=["learned", "fixed-small", "fixed-large"], default="fixed-large")
    p.add_argument("--loss-type", choices=["kl", "mse"], default="mse")

    # UNet
    p.add_argument("--hid-ch", default=128, type=int)
    p.add_argument("--ch-mults", default="1,2,2", type=str)       # 3 levels
    p.add_argument("--num-res-blocks", default=2, type=int)
    p.add_argument("--attn-levels", default="0,1,1", type=str)    # length must match ch-mults
    p.add_argument("--drop-rate", default=0.0, type=float)
    p.add_argument("--resample-with-conv", action="store_true", default=True)

    # logging / sampling
    p.add_argument("--save-intv", default=10, type=int)           # epochs
    p.add_argument("--sample-intv", default=10, type=int)         # epochs
    p.add_argument("--num-samples", default=64, type=int)
    p.add_argument("--apply-clamp", action="store_true", default=True)
    return p


def main():
    args = build_argparser().parse_args()

    seed_all(args.seed)
    device = torch.device(args.device)

    # MNIST -> [-1, 1]
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),
    ])
    train_set = torchvision.datasets.MNIST(
        root=args.data_root, train=True, download=True, transform=tfm,
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Diffusion
    betas = get_beta_schedule(
        args.beta_schedule, beta_start=args.beta_start,
        beta_end=args.beta_end, timesteps=args.timesteps,
    )
    diffusion = GaussianDiffusion(
        betas=betas, model_mean_type=args.model_mean_type,
        model_var_type=args.model_var_type, loss_type=args.loss_type,
    )

    # Model (UNet)
    ch_multipliers = tuple(int(x) for x in args.ch_mults.split(","))
    apply_attn = tuple(bool(int(x)) for x in args.attn_levels.split(","))
    assert len(apply_attn) == len(ch_multipliers), "attn-levels length must match ch-mults length"

    in_channels = 1
    out_channels = 2 * in_channels if args.model_var_type == "learned" else in_channels
    model = UNet(
        in_channels=in_channels, hid_channels=args.hid_ch, out_channels=out_channels,
        ch_multipliers=ch_multipliers, num_res_blocks=args.num_res_blocks,
        apply_attn=apply_attn, time_embedding_dim=None,
        drop_rate=args.drop_rate, resample_with_conv=args.resample_with_conv,
    ).to(device)
    count_parameters(model)

    optimizer = Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    os.makedirs(args.chkpt_dir, exist_ok=True)
    ckpt_path = os.path.join(args.chkpt_dir, "mnist_unet_ddpm.pt")

    start_epoch, global_step = 1, 0
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        print(f"[Resume] loaded {ckpt_path} (start_epoch={start_epoch})")

    # Fixed noise so epoch-wise sample grids are comparable
    fixed_sample_noise = torch.randn(args.num_samples, 1, 28, 28, device=device)

    def save_checkpoint(epoch):
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        }, ckpt_path)
        print(f"[Checkpoint] saved: {ckpt_path}")

    model.train()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_loss, n_batches = 0.0, 0
        for x, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            x = x.to(device, non_blocking=True)
            t = torch.randint(0, diffusion.timesteps, (x.shape[0],), device=device)
            loss = diffusion.train_losses(model, x_0=x, t=t).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

        print(f"[Epoch End] epoch={epoch} avg_loss={epoch_loss / max(1, n_batches):.6f}")

        if args.sample_intv > 0 and epoch % args.sample_intv == 0:
            sample_and_save(model, diffusion, device, args.sample_dir, epoch,
                            num_samples=args.num_samples, apply_clamp=args.apply_clamp,
                            fixed_noise=fixed_sample_noise)
        if args.save_intv > 0 and epoch % args.save_intv == 0:
            save_checkpoint(epoch)

    save_checkpoint(args.epochs)
    print(f"[Done] final checkpoint saved: {ckpt_path}")


if __name__ == "__main__":
    main()
