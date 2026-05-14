import os
import cv2
import numpy as np
import torch
import open3d as o3d

from plyfile import PlyData, PlyElement

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from gaussian_renderer import render

EXPERT_COLOR_MAP = torch.tensor([
    [255, 0, 0],     [0, 255, 0],     [0, 0, 255],     [255, 255, 0],
    [0, 255, 255],   [255, 0, 255],   [255, 128, 0],   [128, 0, 255],
    [0, 255, 128],   [255, 20, 147],  [0, 191, 255],   [255, 215, 0],
    [124, 252, 0],   [255, 99, 71],   [30, 144, 255],  [173, 255, 47],
    [255, 105, 180], [75, 0, 130],    [0, 128, 0],     [184, 134, 11],
    [0, 0, 128],     [128, 128, 0],   [139, 0, 0],     [0, 100, 0]
], dtype=torch.float32) / 255.0

def to_float(x):
    return x.item() if isinstance(x, torch.Tensor) else x

def save_render_comparison(gt_image, rendered_image, iteration, psnr, ssim, lpips, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    gt = gt_image.permute(1, 2, 0).detach().cpu().numpy()
    render = rendered_image.permute(1, 2, 0).detach().cpu().numpy()
    diff = np.abs(gt - render).mean(axis=-1)  # shape: [H, W]

    diff_colored = plt.get_cmap('plasma')(diff)[:, :, :3]  # RGB only, drop alpha
    diff_colored = (diff_colored * 255).astype(np.uint8)

    composite = np.concatenate([
        (gt * 255).astype(np.uint8),
        (render * 255).astype(np.uint8),
        diff_colored
    ], axis=1)

    H, W, _ = composite.shape
    canvas = np.zeros((H + 30, W, 3), dtype=np.uint8)
    canvas[:H] = composite

    text = f"psnr: {to_float(psnr):.4f}/ssim: {to_float(ssim):.4f}/lpips: {to_float(lpips):.4f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, text, (10, H + 20), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    if isinstance(iteration, str):
        save_path = os.path.join(save_dir, f"_{iteration}.png")
    else:    
        save_path = os.path.join(save_dir, f"iter_{iteration:06d}.png")
    cv2.imwrite(save_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

def save_ply_with_colors(gaussians, path, colors):
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    colors_uint8 = (colors.detach().cpu().numpy() * 255).astype(np.uint8)
 
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
 
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, colors_uint8), axis=1)
    elements[:] = list(map(tuple, attributes))
 
    PlyData([PlyElement.describe(elements, 'vertex')]).write(path)
    print(f"PLY saved to {path}")

def _compute_expert_colors(influence_weights):
    dominant_expert_indices = torch.argmax(influence_weights, dim=1)
    color_map = EXPERT_COLOR_MAP.to(influence_weights.device)
    return color_map[dominant_expert_indices]
 
 
def visualize_gating_weights(scene, config, camera_idx, output_ply_path):
    view = scene.test_dataset[camera_idx]
    scene.eval()
 
    # Run non-rigid forward pass to obtain deformed Gaussians and influence weights.
    deformed_gaussians, _, _ = scene.converter(scene.gaussians, view, 0)
    influence_weights = scene.converter.deformer.non_rigid.influence_weights
 
    if influence_weights is None:
        print("Gating weights not computed. Please check the model.")
        return
 
    point_colors = _compute_expert_colors(influence_weights)
    save_ply_with_colors(deformed_gaussians, output_ply_path, point_colors)
 
 
def visualize_canonical_gating_map(scene, config, camera_idx, output_ply_path):
    view = scene.test_dataset[camera_idx]
    scene.eval()
 
    # Run a render pass to populate influence_weights for this pose.
    render(view, config.opt.iterations, scene, config.pipeline,
           torch.tensor([0.0, 0.0, 0.0], device="cuda"))
 
    influence_weights = scene.converter.deformer.non_rigid.influence_weights
 
    if influence_weights is None:
        print("Gating weights not computed.")
        return
 
    point_colors = _compute_expert_colors(influence_weights)
    save_ply_with_colors(scene.gaussians, output_ply_path, point_colors)
 