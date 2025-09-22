
import torch
import copy
import pickle
import os
import numpy as np
from PIL import Image
import torchvision.transforms as T
from diff_gaussian_rasterization import GaussianRasterizer
from utils.loss_utils import l1_loss, ssim

dtype = torch.float32
device = 'cuda'
torch.set_default_dtype(dtype)
torch.set_default_device(device)

# Load data and settings
with open("../gs-splats/splats_data/grad_check_raster.pkl", "rb") as f:
    args = pickle.load(f)
    raster_settings = args["raster_settings"]

with open("../gs-splats/splats_data/grad_check_data.pkl", "rb") as f:
    args = pickle.load(f)
    means3D = args["means3D"].to(device=device, dtype=dtype)
    means2D = args["means2D"].to(device=device, dtype=dtype)
    sh = args["shs"].to(device=device, dtype=dtype)
    opacities = args["opacities"].to(device=device, dtype=dtype)
    scales = args["scales"].to(device=device, dtype=dtype)
    rotations = args["rotations"].to(device=device, dtype=dtype)
    colors_precomp = args["colors_precomp"].to(device=device, dtype=dtype) if args["colors_precomp"] is not None else None
    cov3D_precomp = args["cov3D_precomp"].to(device=device, dtype=dtype) if args["cov3D_precomp"] is not None else None

# Load target image
target_path = "/home/jiexiao/research/skm-gs/data/playroom/model/train/ours_30000/gt/00028.png"
target_img = Image.open(target_path).convert("RGB")
transform = T.Compose([
    T.ToTensor(),
])
target_img = transform(target_img).to(device=device, dtype=dtype)

# Detached copies
means3D_det = means3D.clone().detach()
means2D_det = means2D.clone().detach()
sh_det = sh.clone().detach()
opacities_det = opacities.clone().detach()
scales_det = scales.clone().detach()
rotations_det = rotations.clone().detach()
colors_det = colors_precomp.clone().detach() if colors_precomp is not None else None
cov3D_det = cov3D_precomp.clone().detach() if cov3D_precomp is not None else None

def compute_loss(means2D_in, opacities_in, scales_in, colors_in):
    """
    Generic rendering and loss computation.
    """
    raster_settings_copy = copy.deepcopy(raster_settings)
    rasterizer = GaussianRasterizer(raster_settings=raster_settings_copy)
    rendered, *_ = rasterizer(
        means3D = means3D_det,
        means2D = means2D_in,
        shs = sh_det,
        colors_precomp = colors_in,
        opacities = opacities_in,
        scales = scales_in,
        rotations = rotations_det,
        cov3D_precomp = cov3D_det,
    )
    l1 = l1_loss(rendered, target_img)
    ssim_loss = ssim(rendered, target_img)
    loss = 0.5 * l1 + 0.5 * (1.0 - ssim_loss)
    return loss

def grad_check(param_tensor, compute_loss_fn, name, eps=3e-3, num_check=100):
    """
    Generic gradient check: 
      - param_tensor: tensor with requires_grad
      - compute_loss_fn: function taking param and returning scalar loss
      - name: string for printing
    """
    # Autograd gradient
    param_autograd = param_tensor.clone().requires_grad_(True)
    loss = compute_loss_fn(param_autograd)
    torch.cuda.synchronize()
    loss.backward()
    grad_autograd = param_autograd.grad.clone().detach()

    # Finite difference gradient
    finite_grad = torch.zeros_like(param_tensor)
    flat_autograd = grad_autograd.view(-1)
    
    # Take random indices for finite difference check
    target_idx = torch.randperm(flat_autograd.numel(), device=device)

    for i in target_idx[:num_check]:
        # positive
        p_pos = param_tensor.clone().detach()
        p_pos.view(-1)[i] += eps
        loss_pos = compute_loss_fn(p_pos)

        # negative
        p_neg = param_tensor.clone().detach()
        p_neg.view(-1)[i] -= eps
        loss_neg = compute_loss_fn(p_neg)

        grad_val = (loss_pos - loss_neg) / (2 * eps)
        finite_grad.view(-1)[i] = grad_val

    # Metrics
    min_aut, max_aut = grad_autograd.min().item(), grad_autograd.max().item()
    min_fd, max_fd = finite_grad.min().item(), finite_grad.max().item()
    sample_aut = flat_autograd[target_idx[:num_check]].cpu().numpy()
    sample_fd = finite_grad.view(-1)[target_idx[:num_check]].cpu().numpy()
    dist = torch.norm(flat_autograd - finite_grad.view(-1), p=2).item()

    print(f"--- Grad Check for {name} ---")
    print(f"Autograd grad (min, max): ({min_aut:.6e}, {max_aut:.6e})")
    print(f"Finite-diff grad (min, max): ({min_fd:.6e}, {max_fd:.6e})")
    print(f"Autograd sample: {sample_aut}")
    print(f"Finite-diff sample: {sample_fd}")
    print(f"L2 distance: {dist:.6e}\n")

def main():
    # Opacities
    grad_check(
        param_tensor=opacities_det,
        compute_loss_fn=lambda x: compute_loss(means2D_det, x, scales_det, colors_det),
        name="opacities",
        eps=1e-2
    )

    # means2D
    grad_check(
        param_tensor=means2D_det,
        compute_loss_fn=lambda x: compute_loss(x, opacities_det, scales_det, colors_det),
        name="means2D",
        eps=1e-2
    )

    # scales (conic2D)
    grad_check(
        param_tensor=scales_det,
        compute_loss_fn=lambda x: compute_loss(means2D_det, opacities_det, x, colors_det),
        name="scales",
        eps=1e-1
    )

    # colors_precomp
    # grad_check(
    #     param_tensor=colors_det,
    #     compute_loss_fn=lambda x: compute_loss(means2D_det, opacities_det, scales_det, x),
    #     name="colors_precomp",
    #     eps=1e-2
    # )

if __name__ == "__main__":
    main()
