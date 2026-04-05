#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import csv
import os
import sys
from argparse import ArgumentParser

import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from lpipsPyTorch.modules.lpips import LPIPS
from scene import Scene
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import ssim


def iter_model_paths(models_path):
    point_cloud_dir = os.path.join(models_path, "point_cloud")
    cfg_args_file = os.path.join(models_path, "cfg_args")

    if os.path.isdir(point_cloud_dir) and os.path.isfile(cfg_args_file):
        return [models_path]

    model_dirs = []
    for entry in os.scandir(models_path):
        if entry.is_dir():
            point_cloud_dir = os.path.join(entry.path, "point_cloud")
            cfg_args_file = os.path.join(entry.path, "cfg_args")
            if os.path.isdir(point_cloud_dir) and os.path.isfile(cfg_args_file):
                model_dirs.append(entry.path)

    model_dirs.sort()
    return model_dirs


def load_model_args(model_path, iteration, quiet):
    parser = ArgumentParser(description="Metrics script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")

    args_list = ["--model_path", model_path]
    if iteration >= 0:
        args_list.extend(["--iteration", str(iteration)])
    if quiet:
        args_list.append("--quiet")

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *args_list]
        args = get_combined_args(parser)
    finally:
        sys.argv = old_argv

    args.eval = True
    args.depths = ""
    args.train_test_exp = False

    return model.extract(args), pipeline.extract(args), args.iteration


def evaluate_model(model_path, iteration, quiet):
    dataset, pipeline, loaded_iteration = load_model_args(model_path, iteration, quiet)

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=loaded_iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        views = scene.getTestCameras()

        if not views:
            raise RuntimeError(f"No test cameras found in {model_path}")

        lpips_metric = LPIPS(net_type="vgg").cuda()

        psnrs = []
        ssims = []
        lpipss = []

        for view in tqdm(views, desc=f"Evaluating {os.path.basename(model_path)}", leave=False):
            rendering_pack = render(
                view,
                gaussians,
                pipeline,
                background,
                use_trained_exp=dataset.train_test_exp,
                separate_sh=False,
            )
            rendering = torch.clamp(rendering_pack["render"], 0.0, 1.0)
            gt = torch.clamp(view.original_image[0:3, :, :], 0.0, 1.0)

            if dataset.train_test_exp:
                rendering = rendering[..., rendering.shape[-1] // 2 :]
                gt = gt[..., gt.shape[-1] // 2 :]

            psnrs.append(psnr(rendering, gt).mean().item())
            ssims.append(ssim(rendering.unsqueeze(0), gt.unsqueeze(0)).item())
            lpipss.append(lpips_metric(rendering.unsqueeze(0), gt.unsqueeze(0)).mean().item())

        return (
            sum(psnrs) / len(psnrs),
            sum(ssims) / len(ssims),
            sum(lpipss) / len(lpipss),
        )


def append_row(csv_path, model_name, psnr_value, ssim_value, lpips_value):
    file_exists = os.path.exists(csv_path)
    needs_header = (not file_exists) or os.path.getsize(csv_path) == 0

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if needs_header:
            writer.writerow(["model_name", "PSNR", "SSIM", "LPIPS"])
        writer.writerow(
            [
                model_name,
                f"{psnr_value:.6f}",
                f"{ssim_value:.6f}",
                f"{lpips_value:.6f}",
            ]
        )


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="metrics_all.py",
        description="Render and evaluate all Gaussian Splatting models in a folder",
    )
    parser.add_argument(
        "--models_path",
        default="output",
        type=str,
        help="Path to directory containing trained models",
    )
    parser.add_argument(
        "--iteration", default=-1, type=int, help="Iteration to load (-1 for latest)"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    parser.add_argument(
        "--csv_path",
        default="all_metrics.csv",
        type=str,
        help="Path to output CSV file",
    )

    args = parser.parse_args()

    torch.cuda.set_device(torch.device("cuda:0"))
    safe_state(args.quiet)

    model_dirs = iter_model_paths(args.models_path)
    if not model_dirs:
        raise RuntimeError(f"No model directories found in {args.models_path}")

    for model_path in model_dirs:
        model_name = os.path.basename(os.path.normpath(model_path))
        print(f"\nEvaluating {model_name}")
        try:
            psnr_value, ssim_value, lpips_value = evaluate_model(
                model_path, args.iteration, args.quiet
            )
            print(
                f"{model_name}: PSNR={psnr_value:.6f}, SSIM={ssim_value:.6f}, LPIPS={lpips_value:.6f}"
            )
            append_row(args.csv_path, model_name, psnr_value, ssim_value, lpips_value)
        except Exception as e:
            print(f"Error processing {model_path}: {e}")

    print(f"Wrote results to {args.csv_path}")
