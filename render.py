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
import numpy as np
from scene import Scene
import os
from tqdm import tqdm, trange
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import fix_random
from utils.visualize import visualize_gating_weights, visualize_canonical_gating_map
from scene import GaussianModel

from utils.general_utils import Evaluator, PSEvaluator

import hydra
from omegaconf import OmegaConf
import wandb
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
def predict(config):
    with torch.set_grad_enabled(False):
        gaussians = GaussianModel(config.model.gaussian)
        scene = Scene(config, gaussians, config.exp_dir)
        scene.eval()
        load_ckpt = config.get('load_ckpt', None)
        if load_ckpt is None:
            load_ckpt = os.path.join(scene.save_dir, "ckpt" + str(config.opt.iterations) + ".pth")
        scene.load_checkpoint(load_ckpt)

        bg_color = [1, 1, 1] if config.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_path = os.path.join(config.exp_dir, config.suffix, 'renders')
        makedirs(render_path, exist_ok=True)

        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)
        times = []
        for idx in trange(len(scene.test_dataset), desc="Rendering progress"):
            view = scene.test_dataset[idx]
            iter_start.record()

            render_pkg = render(view, config.opt.iterations, scene, config.pipeline, background,
                                compute_loss=False, return_opacity=False)
            iter_end.record()
            torch.cuda.synchronize()
            elapsed = iter_start.elapsed_time(iter_end)

            rendering = render_pkg["render"]

            wandb_img = [wandb.Image(rendering[None], caption='render_{}'.format(view.image_name)),]
            wandb.log({'test_images': wandb_img})

            torchvision.utils.save_image(rendering, os.path.join(render_path, f"render_{view.image_name}.png"))

            # evaluate
            times.append(elapsed)

        _time = np.mean(times[1:])
        wandb.log({'metrics/time': _time})
        np.savez(os.path.join(config.exp_dir, config.suffix, 'results.npz'),
                 time=_time)
        
        video_path = os.path.join(config.exp_dir, config.suffix, 'render_video.mp4')
        image_files = sorted([f for f in os.listdir(render_path) if f.endswith(".png")])
        if image_files:
            import cv2
            first_frame = cv2.imread(os.path.join(render_path, image_files[0]))
            height, width, _ = first_frame.shape
            fps = 24  # adjust as needed

            out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

            for fname in image_files:
                frame = cv2.imread(os.path.join(render_path, fname))
                out.write(frame)

            out.release()
            print(f"Video saved to {video_path}")
        else:
            print("No rendered images found to create video.")

def test(config):
    with torch.no_grad():
        gaussians = GaussianModel(config.model.gaussian)
        scene = Scene(config, gaussians, config.exp_dir)
        scene.eval()

        bg_color = [1, 1, 1] if config.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        evaluator = PSEvaluator() if config.dataset.name == 'people_snapshot' else Evaluator()

        ckpt_name = "Final"
        ckpt_path = os.path.join(config.exp_dir, "ckpt" + str(config.opt.iterations) + ".pth")
        
        if not os.path.exists(ckpt_path):
            print(f"\n[Error] Final checkpoint not found at: {ckpt_path}")
            return

        scene.load_checkpoint(ckpt_path)
        vis_path = os.path.join(config.exp_dir, config.suffix, 'visualizations', ckpt_name)
        os.makedirs(vis_path, exist_ok=True)

        # Gating weight visualization
        if config.get('visualize_gating', False):
            # Choose the frame to visualize
            target_frames = config.get('gating_frames', [7, 8, 9, 10])

            print(f"\n[Visualization] Extracting gating weights for frames: {target_frames}")
            for frame_idx in target_frames:
                canonical_ply_path = os.path.join(vis_path, f"canonical_gating_frame_{frame_idx}.ply")
                visualize_canonical_gating_map(scene, config, frame_idx, canonical_ply_path)

                obs_ply_path = os.path.join(vis_path, f"obs_gating_frame_{frame_idx}.ply")
                visualize_gating_weights(scene, config, frame_idx, obs_ply_path)

        evaluate_current_model(
            scene=scene,
            config=config,
            background=background,
            evaluator=evaluator,
            ckpt_name=ckpt_name,
        )

def evaluate_current_model(scene, config, background, evaluator, ckpt_name):
    render_path = os.path.join(config.exp_dir, config.suffix, 'renders', ckpt_name)
    os.makedirs(render_path, exist_ok=True)
    
    vis_path = os.path.join(config.exp_dir, config.suffix, 'visualizations', ckpt_name)
    do_color_visualization = config.get('visualize_color', False)
    
    if do_color_visualization:
        os.makedirs(vis_path, exist_ok=True)

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    metrics_acc = {'psnr': [], 'ssim': [], 'lpips': []}
    times = []
    for idx in trange(len(scene.test_dataset), desc=f"Rendering ({ckpt_name})"):
        view = scene.test_dataset[idx]
        
        iter_start.record()

        render_pkg = render(view, config.opt.iterations, scene, config.pipeline, background,
                            compute_loss=False, return_opacity=True)

        iter_end.record()
        torch.cuda.synchronize()
        elapsed = iter_start.elapsed_time(iter_end) 
        times.append(elapsed) 

        rendering = render_pkg["render"]
        opacity = render_pkg["opacity_render"]
        gt = view.original_image[:3, :, :]
        gt_mask = view.original_mask

        current_metrics = {'psnr': 0.0, 'ssim': 0.0, 'lpips': 0.0}
        
        if config.evaluate:
            m = evaluator(rendering, gt, opacity=opacity, gt_mask=gt_mask)
            for k in metrics_acc.keys():
                if k in m:
                    val = m[k].item()
                    metrics_acc[k].append(val)
                    current_metrics[k] = val

        else:
            for k in metrics_acc.keys():
                metrics_acc[k].append(0.0)

        diff = torch.abs(gt - rendering)
        error_mag = torch.mean(diff, dim=0, keepdim=True) * 5.0
        colored_error_map = torch.cat([error_mag, error_mag, torch.ones_like(error_mag)], dim=0).clamp(0, 1)
        comparison_img = torch.cat([gt, rendering, colored_error_map], dim=2)
        ndarr = comparison_img.clamp(0, 1).mul(255).add_(0.5).to('cpu', torch.uint8).numpy().transpose(1, 2, 0)
        
        pil_img = Image.fromarray(ndarr)
        draw = ImageDraw.Draw(pil_img)
        text = (f"PSNR: {current_metrics['psnr']:.2f} | SSIM: {current_metrics['ssim']:.4f}")
        draw.rectangle([10, 10, 300, 40], fill=(0, 0, 0))
        draw.text((20, 15), text, fill=(255, 255, 255))
        pil_img.save(os.path.join(render_path, f"render_{view.image_name}.png"))

    results = {}
    for k, v in metrics_acc.items():
        results[k] = torch.mean(torch.tensor(v)).item() if v else 0.0

    if hasattr(evaluator, 'compute_fid'):
        results['fid'] = evaluator.compute_fid().item()
    else:
        results['fid'] = 0.0

    if len(times) > 1:
        avg_time_ms = np.mean(times[1:])
    else:
        avg_time_ms = np.mean(times) if times else 0.0
        
    avg_fps = 1000.0 / avg_time_ms if avg_time_ms > 0 else 0.0

    print(f"\n----- [{ckpt_name}] Evaluation Summary -----")
    print(f"  PSNR:  {results['psnr']:.4f}")
    print(f"  SSIM:  {results['ssim']:.4f}")
    print(f"  LPIPS: {results['lpips']:.5f}")
    print(f"  FID:   {results['fid']:.4f}") 
    print(f"  Time:  {avg_time_ms:.4f} ms")
    print(f"  FPS:   {avg_fps:.2f}")
    print("--------------------------------------------\n")

    log_dict = {f'metrics_{ckpt_name}/{k}': v for k, v in results.items()}
    log_dict.update({
        f'metrics_{ckpt_name}/time': avg_time_ms,
        f'metrics_{ckpt_name}/fps': avg_fps,
    })
    wandb.log(log_dict)


    np.savez(
        os.path.join(config.exp_dir, config.suffix, f'results_{ckpt_name}.npz'),
        psnr=results['psnr'],
        ssim=results['ssim'],
        lpips=results['lpips'],
        fid=results['fid'],
        time=avg_time_ms,
        fps=avg_fps,
    )

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    OmegaConf.set_struct(config, False)
    config.dataset.preload = False

    config.exp_dir = config.get('exp_dir') or os.path.join('./exp', config.name)
    os.makedirs(config.exp_dir, exist_ok=True)

    # set wandb logger
    if config.mode == 'test':
        config.suffix = config.mode + '-' + config.dataset.test_mode
    elif config.mode == 'predict':
        predict_seq = config.dataset.predict_seq
        if config.dataset.name in ['zjumocap', 'zjumocap_refine', 'people_snapshot']:
            predict_dict = {
                0: 'dance0',
                1: 'dance1',
                2: 'flipping',
                3: 'canonical',
                4: 'dance2',
                5: 'fencing',
                6: 'eating',
                7: 'walking'
            }
        else:
            predict_dict = {
                0: 'dance2',
                1: 'rotation',
            }
        predict_mode = predict_dict[predict_seq]
        config.suffix = config.mode + '-' + predict_mode
    else:
        raise ValueError
    if config.dataset.freeview:
        config.suffix = config.suffix + '-freeview'
    wandb_name = config.name + '-' + config.suffix
    wandb.init(
        mode="disabled" if config.wandb_disable else None,
        name=wandb_name,
        project='project_name',
        entity='Your wandb account',
        dir=config.exp_dir,
        config=OmegaConf.to_container(config, resolve=True),
        settings=wandb.Settings(start_method='fork'),
    )

    fix_random(config.seed)

    if config.mode == 'test':
        test(config)
    elif config.mode == 'predict':
        predict(config)
    else:
        raise ValueError

if __name__ == "__main__":
    main()