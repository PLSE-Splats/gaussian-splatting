"""
Usage:
python render_single_view.py -m /path/to/model  -s /path/to/put/renders --view_index 23 --skip_test
"""
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel


def render_view(model_path, name, iteration, views, gaussians, pipeline, background, view_index):
    render_path = os.path.join(model_path, name, f"ours_{iteration}", "renders")
    gts_path = os.path.join(model_path, name, f"ours_{iteration}", "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    print(f"Rendering view {view_index}")
    view = views[view_index]
    render_result = render(view, gaussians, pipeline, background)["render"]
    gt = view.original_image[0:3, :, :]
    torchvision.utils.save_image(render_result, os.path.join(str(render_path), f"{view_index:05d}.png"))
    torchvision.utils.save_image(gt, os.path.join(str(gts_path), f"{view_index:05d}.png"))


def render_sets(
        dataset: ModelParams,
        iteration: int,
        pipeline: PipelineParams,
        skip_train: bool,
        skip_test: bool,
        view_index: int
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_view(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline,
                        background, view_index)

        if not skip_test:
            render_view(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline,
                        background, view_index)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Single view render script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--view_index", default=0, type=int, help="Index of the view to render")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,
                args.view_index)
