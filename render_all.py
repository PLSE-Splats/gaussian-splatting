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

import torch
from scene import Scene
import os
import sys
from time import perf_counter
from tqdm import tqdm
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

try:
    from diff_gaussian_rasterization import SparseGaussianAdam

    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


NUM_WARMUP_ITERATIONS = 200
NUM_ITERATIONS = 200


def get_render_scale(model_name):
    """Determine render scale based on model name."""
    model_name_lower = model_name.lower()
    if model_name_lower in ["playroom", "drjohnson"]:
        return 1.0  # Full resolution
    elif model_name_lower in ["truck", "train"]:
        return 0.5  # Half resolution
    else:
        return 0.25  # Quarter resolution


def benchmark_model(model_path, iteration, separate_sh, quiet):
    parser = ArgumentParser(description="Benchmark script parameters")
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

    model_name = os.path.basename(os.path.normpath(model_path))
    render_scale = get_render_scale(model_name)

    print(f"Benchmarking {args.model_path} (render scale: {render_scale})")
    safe_state(args.quiet)

    dataset = model.extract(args)
    pipeline_params = pipeline.extract(args)

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, resolution_scales=[render_scale])

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        views = scene.getTestCameras(scale=render_scale)
        if len(views) == 0:
            raise RuntimeError(f"No test images found in {model_path}")

        # Warmup
        print("Warmup...")
        for _ in tqdm(range(NUM_WARMUP_ITERATIONS), desc="Warming Up", leave=False):
            for view in views:
                _ = render(
                    view,
                    gaussians,
                    pipeline_params,
                    background,
                    use_trained_exp=dataset.train_test_exp,
                    separate_sh=separate_sh,
                )

        # Actual timing
        print("Benchmarking...")
        num_test_images = len(views)
        torch.cuda.synchronize()
        start_time = perf_counter()

        for _ in tqdm(
            range(NUM_ITERATIONS), desc="Benchmarking Performance", leave=False
        ):
            for view in views:
                _ = render(
                    view,
                    gaussians,
                    pipeline_params,
                    background,
                    use_trained_exp=dataset.train_test_exp,
                    separate_sh=separate_sh,
                )

        torch.cuda.synchronize()
        end_time = perf_counter()

        total_time = end_time - start_time
        total_num_images = NUM_ITERATIONS * num_test_images
        avg_fps = total_num_images / total_time
        avg_ms_per_image = (total_time * 1000) / total_num_images

        return avg_fps, avg_ms_per_image


def iter_model_paths(models_path):
    # Check if the path itself is a model (has point_cloud directory and cfg_args file)
    point_cloud_dir = os.path.join(models_path, "point_cloud")
    cfg_args_file = os.path.join(models_path, "cfg_args")

    if os.path.isdir(point_cloud_dir) and os.path.isfile(cfg_args_file):
        return [models_path]

    # Otherwise, iterate through subdirectories
    model_dirs = []
    for entry in os.scandir(models_path):
        if entry.is_dir():
            point_cloud_dir = os.path.join(entry.path, "point_cloud")
            cfg_args_file = os.path.join(entry.path, "cfg_args")
            if os.path.isdir(point_cloud_dir) and os.path.isfile(cfg_args_file):
                model_dirs.append(entry.path)

    model_dirs.sort()
    return model_dirs


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(
        prog="render_all.py",
        description="Runs warmup and measured benchmark passes for all Gaussian Splatting models",
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
        "--csv_path", default="all_fps.csv", type=str, help="Path to output CSV file"
    )

    args = parser.parse_args()

    model_dirs = iter_model_paths(args.models_path)
    if not model_dirs:
        raise RuntimeError(f"No model directories found in {args.models_path}")

    rows = ["model_name,FPS,MS"]

    for model_path in model_dirs:
        try:
            model_name = os.path.basename(os.path.normpath(model_path))
            print(f"\nBenchmark {model_name} ({NUM_ITERATIONS} iterations).")

            fps, ms = benchmark_model(
                model_path, args.iteration, SPARSE_ADAM_AVAILABLE, args.quiet
            )

            fps_round = round(fps)
            ms_round = round(ms, 2)

            print(f"{model_name}:\t{fps_round:,} FPS ({ms_round} ms)\n")
            rows.append(f"{model_name},{fps_round},{ms_round}")

        except Exception as e:
            print(f"Error processing {model_path}: {e}")
            continue

    with open(args.csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")

    print(f"Wrote results to {args.csv_path}")
