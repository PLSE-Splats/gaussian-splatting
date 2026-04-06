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
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from gaussian_renderer import GaussianModel
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import time
import csv
from pathlib import Path
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_and_save_test_set(model_path, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    """Render test set and save GT and rendered images"""
    render_path = os.path.join(model_path, "test", "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, "test", "ours_{}".format(iteration), "gt")
    
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    for idx, view in enumerate(tqdm(views, desc="Rendering test images")):
        rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
        gt = view.original_image[0:3, :, :]
        
        if train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]
        
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


def benchmark_fps(views, gaussians, pipeline, background, train_test_exp, separate_sh, num_loops=200):
    """Run warmup and real benchmark loops"""
    # Warmup loops
    print(f"Running {num_loops} warmup loops...")
    for _ in tqdm(range(num_loops), desc="Warmup"):
        for view in views:
            render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
    
    # Real benchmark loops
    print(f"Running {num_loops} real benchmark loops...")
    start_time = time.perf_counter()
    for _ in tqdm(range(num_loops), desc="Benchmark"):
        for view in views:
            render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    total_renders = num_loops * len(views)
    avg_fps = total_renders / total_time
    
    return avg_fps


def compute_metrics(render_path, gts_path):
    """Compute PSNR, SSIM, and LPIPS metrics"""
    renders_dir = Path(render_path)
    gt_dir = Path(gts_path)
    
    renders = []
    gts = []
    
    for fname in sorted(os.listdir(renders_dir)):
        render_img = torchvision.io.read_image(str(renders_dir / fname)).float() / 255.0
        gt_img = torchvision.io.read_image(str(gt_dir / fname)).float() / 255.0
        
        renders.append(render_img.unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(gt_img.unsqueeze(0)[:, :3, :, :].cuda())
    
    psnrs = []
    ssims = []
    lpipss = []
    
    print("Computing metrics...")
    for idx in tqdm(range(len(renders)), desc="Metrics"):
        psnrs.append(psnr(renders[idx], gts[idx]))
        ssims.append(ssim(renders[idx], gts[idx]))
        lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))
    
    avg_psnr = torch.tensor(psnrs).mean().item()
    avg_ssim = torch.tensor(ssims).mean().item()
    avg_lpips = torch.tensor(lpipss).mean().item()
    
    return avg_psnr, avg_ssim, avg_lpips


def infer_source_path(model_path):
    """Infer source path from model path or cfg_args file"""
    model_path = Path(model_path)
    
    # If models are in 3dgs_models/models/bicycle
    # Source images are in 3dgs_models/source/bicycle
    scene_name = model_path.name
    parent = model_path.parent
    
    # Check if parent directory is named "models"
    # Go up one level and look for "source" sibling directory
    grandparent = parent.parent
    source_dir = grandparent / "source" / scene_name
    if source_dir.exists():
        print(f"Inferred source path from directory structure: {source_dir}")
        return str(source_dir)
    
    
    print(f"Warning: Could not infer source path for model {model_path}")
    return ""


def benchmark_model(model_path, iteration, sh_degree, white_background, num_loops=200):
    """Benchmark a single model"""
    print(f"\n{'='*80}")
    print(f"Benchmarking model: {model_path}")
    print(f"{'='*80}\n")
    
    with torch.no_grad():
        # Infer source path
        source_path = infer_source_path(model_path)
        
        # Create a minimal dataset params object
        class DatasetParams:
            def __init__(self, model_path, sh_degree, white_background, source_path):
                self.model_path = model_path
                self.sh_degree = sh_degree
                self.source_path = source_path
                self.images = "images"
                self.resolution = -1
                self.white_background = white_background
                self.data_device = "cuda"
                self.eval = False
                self.train_test_exp = False
                self.depths = ""
        
        dataset = DatasetParams(model_path, sh_degree, white_background, source_path)
        
        # Load the model
        gaussians = GaussianModel(sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        # Get test cameras
        test_views = scene.getTestCameras()
        
        if len(test_views) == 0:
            print("No test cameras found! Skipping this model.")
            return None
        
        # Setup background
        bg_color = [1, 1, 1] if white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # Create pipeline params
        class PipelineParamsSimple:
            def __init__(self):
                self.convert_SHs_python = False
                self.compute_cov3D_python = False
                self.debug = False
        
        pipeline = PipelineParamsSimple()
        
        # Render and save test set
        render_and_save_test_set(
            model_path, 
            scene.loaded_iter, 
            test_views, 
            gaussians, 
            pipeline, 
            background,
            dataset.train_test_exp,
            SPARSE_ADAM_AVAILABLE
        )
        
        # Benchmark FPS
        avg_fps = benchmark_fps(
            test_views, 
            gaussians, 
            pipeline, 
            background,
            dataset.train_test_exp,
            SPARSE_ADAM_AVAILABLE,
            num_loops
        )
        
        print(f"\nAverage FPS: {avg_fps:.2f}")
        
        # Compute metrics
        render_path = os.path.join(model_path, "test", "ours_{}".format(scene.loaded_iter), "renders")
        gts_path = os.path.join(model_path, "test", "ours_{}".format(scene.loaded_iter), "gt")
        
        avg_psnr, avg_ssim, avg_lpips = compute_metrics(render_path, gts_path)
        
        print(f"Average PSNR: {avg_psnr:.4f}")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"Average LPIPS: {avg_lpips:.4f}")
        
        return {
            'model_name': os.path.basename(model_path),
            'fps': avg_fps,
            'psnr': avg_psnr,
            'ssim': avg_ssim,
            'lpips': avg_lpips
        }


def main():
    parser = ArgumentParser(description="Benchmark all models in a directory")
    parser.add_argument("--models_path", "-m", required=True, type=str, help="Path to directory containing trained models")
    parser.add_argument("--iteration", default=-1, type=int, help="Iteration to load (-1 for latest)")
    parser.add_argument("--sh_degree", default=3, type=int, help="Spherical harmonics degree")
    parser.add_argument("--white_background", action="store_true", help="Use white background")
    parser.add_argument("--num_loops", default=200, type=int, help="Number of loops for warmup and benchmark")
    parser.add_argument("--output_csv", default="benchmark.csv", type=str, help="Output CSV file name")
    args = parser.parse_args()
    
    # Initialize CUDA
    safe_state(False)
    
    # Find all model directories
    models_path = Path(args.models_path)
    if not models_path.exists():
        print(f"Error: Models path {models_path} does not exist!")
        return
    
    # Get all subdirectories that contain trained models
    model_dirs = []
    for item in models_path.iterdir():
        if item.is_dir():
            # Check if it looks like a trained model (has point_cloud directory)
            if (item / "point_cloud").exists():
                model_dirs.append(item)
    
    if len(model_dirs) == 0:
        print(f"No trained models found in {models_path}")
        return
    
    print(f"Found {len(model_dirs)} models to benchmark")
    
    # Benchmark each model
    results = []
    for model_dir in sorted(model_dirs):
        try:
            result = benchmark_model(
                str(model_dir),
                args.iteration,
                args.sh_degree,
                args.white_background,
                args.num_loops
            )
            if result is not None:
                results.append(result)
        except Exception as e:
            print(f"Error benchmarking {model_dir}: {e}")
            import traceback
            traceback.print_exc()
    
    # Write results to CSV
    if len(results) > 0:
        output_path = Path(args.models_path) / args.output_csv
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = ['model_name', 'fps', 'psnr', 'ssim', 'lpips']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        
        print(f"\n{'='*80}")
        print(f"Benchmark complete! Results written to {output_path}")
        print(f"{'='*80}")
        
        # Print summary table
        print("\nSummary:")
        print(f"{'Model':<40} {'FPS':>10} {'PSNR':>10} {'SSIM':>10} {'LPIPS':>10}")
        print("-" * 80)
        for result in results:
            print(f"{result['model_name']:<40} {result['fps']:>10.2f} {result['psnr']:>10.4f} {result['ssim']:>10.4f} {result['lpips']:>10.4f}")
    else:
        print("\nNo models were successfully benchmarked.")


if __name__ == "__main__":
    main()
