import torch
from diff_gaussian_rasterization import _RasterizeGaussians, GaussianRasterizationSettings
from diff_gaussian_rasterization import GaussianRasterizer
import copy
import pickle
import os
import numpy as np
from PIL import Image
import torchvision.transforms as T
from utils.loss_utils import l1_loss, ssim

dtype = torch.float32
device = 'cuda'
torch.set_default_dtype(dtype)
torch.set_default_device(device)

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

target_img = Image.open("/home/jiexiao/research/skm-gs/data/playroom/model/train/ours_30000/gt/00028.png").convert("RGB")
transform = T.Compose([
    T.ToTensor(),  # Converts to shape (C, H, W) and scales to [0, 1]
])
target_img = transform(target_img).to(device=device, dtype=dtype)


# alpha 3d grad check
opacities_autograd = opacities.clone().requires_grad_(True)

means3D_detached = means3D.clone().detach()
means2D_detached = means2D.clone().detach()
sh_detached = sh.clone().detach()
colors_precomp_detached = colors_precomp.clone().detach() if colors_precomp is not None else None
scales_detached = scales.clone().detach()
rotations_detached = rotations.clone().detach()
cov3D_precomp_detached = cov3D_precomp.clone().detach() if cov3D_precomp is not None else None

# compute scalar loss
def compute_loss_opacity(opacities_input):
    raster_settings_copy = copy.deepcopy(raster_settings)
    rasterizer = GaussianRasterizer(raster_settings=raster_settings_copy)

    rendered_img, *_ = rasterizer(
        means3D = means3D_detached,
        means2D = means2D_detached,
        shs = sh_detached,
        colors_precomp = colors_precomp_detached,
        opacities = opacities_input,
        scales = scales_detached,
        rotations = rotations_detached,
        cov3D_precomp = cov3D_precomp_detached,
    )

    l1loss = l1_loss(rendered_img, target_img)
    ssim_loss = ssim(rendered_img, target_img)
    loss = 0.5 * l1loss + 0.5 * (1.0 - ssim_loss)
    return loss


loss = compute_loss_opacity(opacities_autograd)
torch.cuda.synchronize()
loss.backward()
print("Loss:", loss.item())
autograd_opacity = opacities_autograd.grad.clone().detach()

# Find non-zero entries
non_zero_indices = torch.nonzero(autograd_opacity.view(-1), as_tuple=True)[0]
print("Non-zero opacity grad indices:", non_zero_indices[:10])
print("Autograd Gradient (min, max):", autograd_opacity.min().item(), autograd_opacity.max().item())

# Finite Difference Gradient for opacity
eps = 3e-3
finite_diff_grad = torch.zeros_like(opacities)

for i in non_zero_indices[:10]:
    op_pos = opacities.clone().detach()
    op_pos.view(-1)[i] += eps
    loss_pos = compute_loss_opacity(op_pos)

    op_neg = opacities.clone().detach()
    op_neg.view(-1)[i] -= eps
    loss_neg = compute_loss_opacity(op_neg)

    grad_val = (loss_pos - loss_neg) / (2 * eps)
    finite_diff_grad.view(-1)[i] = grad_val

print("Finite Diff Gradient (min, max):", finite_diff_grad.min().item(), finite_diff_grad.max().item())
print("Autograd Gradient (sample):", autograd_opacity.view(-1)[non_zero_indices[:10]])
print("Finite Diff Gradient (sample):", finite_diff_grad.view(-1)[non_zero_indices[:10]])

opacities_autograd.grad = None