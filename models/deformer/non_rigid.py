# Copyright (c) 2026 Hyeri Yang, Junyoung Hong, Shinwoong Kim, Kyungjae Lee
#
# This file is part of the AvatarMoE codebase.
#
# Adapted from 3DGS-Avatar:
# https://github.com/mikeqzy/3dgs-avatar-release
# (CVPR 2024, MIT License)
#
# Modifications include:
# - GMM-based non-rigid deformation field
# - Part-aware Mixture-of-Experts architecture
# - JointGaussian representation

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch3d.transforms as tf

from models.network_utils import (HierarchicalPoseEncoder,
                                  VanillaCondMLP,
                                  HashGrid)
from utils.general_utils import quaternion_multiply, quaternion_to_rotation_matrix

class NonRigidDeform(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, gaussians, iteration, camera, compute_loss=True):
        raise NotImplementedError

class Identity(NonRigidDeform):
    def __init__(self, cfg, metadata):
        super().__init__(cfg)

    def forward(self, gaussians, iteration, camera, compute_loss=True):
        return gaussians, {}

class PositionalEncoder(nn.Module):
    def __init__(self, d_input, n_freqs, log_space=True):
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.d_output = d_input * (1 + 2 * n_freqs)
        self.embed_fns = [lambda x: x]

        if self.log_space:
            freq_bands = 2.**torch.linspace(0., n_freqs - 1, n_freqs)
        else:
            freq_bands = torch.linspace(1., 2.**(n_freqs - 1), n_freqs)

        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))

    def forward(self, x):
        return torch.cat([fn(x) for fn in self.embed_fns], dim=-1)

class JointGaussian(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim * 2, 7) # 3 (scales) + 4 (quaternion)
        )

    def forward(self, joint_position, latent_code_j):
        params = self.decoder(latent_code_j)
        
        raw_scales = params[..., :3]
        max_scale = 0.1
        scales = max_scale * torch.sigmoid(raw_scales) + 1e-4

        quaternion = F.normalize(params[..., 3:], p=2, dim=-1)
        rotation = quaternion_to_rotation_matrix(quaternion)

        D = torch.diag_embed(scales)
        covariance = rotation @ D @ rotation.transpose(-1, -2)
        
        return joint_position, covariance, scales, rotation

class GMMNonRigidField(NonRigidDeform):
    def __init__(self, cfg, metadata):
        super().__init__(cfg)
        self.num_joints = cfg.pose_encoder.num_joints
        self.dim_per_joint = cfg.pose_encoder.dim_per_joint
        self.global_dim = cfg.pose_encoder.global_dim
        self.feature_dim = cfg.get('feature_dim', 0)
        self.aabb = metadata['aabb']
        self.delay = cfg.get('delay', 0)
        self.boundary_threshold = cfg.get('boundary_threshold', 0.1)
        self.canonical_skinning_weights = None
        self.sparse_mlp_info = None
        self.influence_weights = None

        # --- [Ablation Study Flag] ---
        self.use_smpl_gating = cfg.get('use_smpl_weights_for_gating', False)
        # ---------------------------

        self.pose_encoder = HierarchicalPoseEncoder(**cfg.pose_encoder)

        self.joint_gaussians = nn.ModuleList([JointGaussian(self.dim_per_joint) for _ in range(self.num_joints)])

        if not self.use_smpl_gating:
            print("You are using GMM Gating")
            self.mixture_weight_decoder = nn.Sequential(
                nn.Linear(self.global_dim, self.global_dim), nn.ReLU(inplace=True),
                nn.Linear(self.global_dim, self.num_joints)
            )
        else:
            print("Your are using SMPL Gating")

        correction_mlp_out_dim = 3 + 3 + 4 + self.feature_dim

        if cfg.use_hashgrid:
            print("You are using HashGrid")
            self.shared_hash_encoder = HashGrid(cfg.hashgrid)
            correction_mlp_in_dim = self.shared_hash_encoder.n_output_dims + 3

        elif cfg.use_positional_encoder:
            print("You are using Positional Encoder")        
            self.pos_encoder = PositionalEncoder(d_input=3, n_freqs=10)
            correction_mlp_in_dim = self.pos_encoder.d_output

        else:
            print("choose positional encoder or hashgrid")

        print("You are using Vanilla MLP")
        self.correction_mlps = nn.ModuleList([
            VanillaCondMLP(
                dim_in=correction_mlp_in_dim,
                dim_cond=self.dim_per_joint,
                dim_out=correction_mlp_out_dim,
                config=cfg.vanilla_mlp
            ) for _ in range(self.num_joints)
        ])
        for mlp in self.correction_mlps:
            last_layer_name = f"lin{mlp.num_layers - 2}"
            final_layer = getattr(mlp, last_layer_name)
            torch.nn.init.zeros_(final_layer.weight)
            torch.nn.init.zeros_(final_layer.bias[:6])
            with torch.no_grad():
                final_layer.bias[6] = 1.0
                final_layer.bias[7:] = 0.0

    def _compute_intra_expert_coherence_loss(self, sparse_mlp_info):
        all_coherence_losses = []
        num_active_experts = 0

        if sparse_mlp_info is None:
            return torch.tensor(0.0)

        for j in sparse_mlp_info:
            expert_info = sparse_mlp_info[j]
            mlp_outputs = expert_info['outputs'] # (N, D)
            influence_weights = expert_info['weights'] # (N,)

            if mlp_outputs.shape[0] < 2:
                continue
            
            sum_of_weights = torch.sum(influence_weights)
            if sum_of_weights < 1e-6:
                continue

            weighted_deltas = mlp_outputs * influence_weights.unsqueeze(-1)
            mean_deltas = torch.sum(weighted_deltas, dim=0) / sum_of_weights
            
            diff_from_mean_sq = (mlp_outputs - mean_deltas.unsqueeze(0)) ** 2
            weighted_diff_sq = diff_from_mean_sq * influence_weights.unsqueeze(-1)
            weighted_variance = torch.sum(weighted_diff_sq, dim=0) / sum_of_weights

            all_coherence_losses.append(torch.mean(weighted_variance))
            num_active_experts += 1
        
        if num_active_experts == 0:
            device = next(iter(self.parameters())).device
            return torch.tensor(0.0, device=device)
            
        coherence_loss = torch.mean(torch.stack(all_coherence_losses))
        
        return coherence_loss
    
    def _calculate_log_pdf(self, points_norm, means_norm, covariances):
        B, N, D = points_norm.shape
        J = self.num_joints
        diff = points_norm.unsqueeze(2) - means_norm.unsqueeze(1)
        jitter = torch.eye(D, device=points_norm.device).unsqueeze(0).unsqueeze(0) * 1e-6
        inv_covariances = torch.inverse(covariances + jitter)
        log_det_covariances = torch.logdet(covariances)
        diff_for_matmul = diff.unsqueeze(-2)
        inv_cov_for_matmul = inv_covariances.unsqueeze(1)
        temp = diff_for_matmul @ inv_cov_for_matmul
        mahalanobis = (temp @ diff.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        log_2pi = torch.log(torch.tensor(2.0 * np.pi, device=points_norm.device)) * D
        return -0.5 * (mahalanobis + log_det_covariances.unsqueeze(1) + log_2pi)

    def forward(self, gaussians, iteration, camera, compute_loss=True):
        if iteration < self.delay:
            deformed_gaussians = gaussians.clone()
            if self.feature_dim > 0:
                setattr(deformed_gaussians, "non_rigid_feature",
                        torch.zeros(gaussians.get_xyz.shape[0], self.feature_dim).cuda())
            return deformed_gaussians, {}

        self.canonical_skinning_weights = gaussians.skinning_weights
        posed_xyz = gaussians.get_xyz
        n_pts = posed_xyz.shape[0]
        posed_xyz_norm = self.aabb.normalize(posed_xyz, sym=True)

        rots = camera.rots
        Jtrs = camera.Jtrs
        
        if self.cfg.use_betas is True:
            betas = camera.betas.squeeze(0).to(rots.device)
        else:
            betas = None
        all_latent_codes, global_feat = self.pose_encoder(rots, Jtrs, betas)
        per_joint_latents = all_latent_codes.view(1, self.num_joints, self.dim_per_joint)

        all_means, all_covariances, all_rotations = [], [], []

        for j in range(self.num_joints):
            mean, cov, _, rot = self.joint_gaussians[j](Jtrs[:, j, :], per_joint_latents[:, j, :])
            all_means.append(mean)
            all_covariances.append(cov)
            all_rotations.append(rot)

        all_means = torch.cat(all_means, dim=0).unsqueeze(0)
        all_covariances = torch.cat(all_covariances, dim=0).unsqueeze(0)
        all_rotations = torch.cat(all_rotations, dim=0).unsqueeze(0)
        all_means_norm = self.aabb.normalize(all_means.squeeze(0), sym=True).unsqueeze(0)

        if self.use_smpl_gating:
            influence_weights = gaussians.skinning_weights
        else:
            mixture_logits = self.mixture_weight_decoder(global_feat)
            log_mixture_weights = F.log_softmax(mixture_logits, dim=-1)
            log_pdfs = self._calculate_log_pdf(posed_xyz_norm.unsqueeze(0), all_means_norm, all_covariances)
            log_weighted_pdfs = log_mixture_weights.unsqueeze(1) + log_pdfs
            
            log_denominator = torch.logsumexp(log_weighted_pdfs, dim=-1, keepdim=True)
            influence_weights = torch.exp(log_weighted_pdfs - log_denominator).squeeze(0)
        
        self.influence_weights = influence_weights

        if self.cfg.use_hashgrid:
            shared_features_for_mlp  = self.shared_hash_encoder(posed_xyz_norm.contiguous())
        else: 
            shared_features_for_mlp = self.pos_encoder(posed_xyz_norm)

        final_delta_xyz = torch.zeros_like(posed_xyz)
        final_delta_scale = torch.zeros_like(gaussians.get_scaling)
        final_delta_rot_sum = torch.zeros(n_pts, 4, device=posed_xyz.device)
        if self.feature_dim > 0:
            final_feature = torch.zeros(n_pts, self.feature_dim, device=posed_xyz.device)
        
        self.sparse_mlp_info = {}

        for j in range(self.num_joints):
            mean_j = all_means[0, j]
            rotation_j = all_rotations[0, j]
            inv_rotation_j = rotation_j.transpose(-1, -2)

            local_xyz_j = (posed_xyz - mean_j) @ inv_rotation_j
            combined_features = torch.cat([shared_features_for_mlp, local_xyz_j], dim=-1)

            latent_j = per_joint_latents[:, j]
            latent_for_mlp = latent_j.expand(posed_xyz.shape[0], -1)

            mlp_output = self.correction_mlps[j](combined_features, latent_for_mlp)

            delta_j_xyz_local = mlp_output[:, :3]
            delta_j_scale = mlp_output[:, 3:6]
            delta_j_rot = mlp_output[:, 6:10]
            delta_j_xyz_world = delta_j_xyz_local @ rotation_j
            
            weights_j = self.influence_weights[:, j].unsqueeze(-1)

            final_delta_xyz += delta_j_xyz_world.squeeze(0) * weights_j
            final_delta_scale += delta_j_scale * weights_j
            final_delta_rot_sum += delta_j_rot * weights_j
            if self.feature_dim > 0:
                feature_j = mlp_output[:, 10:]
                final_feature += feature_j * weights_j

            self.sparse_mlp_info[j] = {
                'outputs': mlp_output,
                'weights': weights_j.squeeze(-1) # (N, 1) -> (N,)
            }

        final_delta_rot = F.normalize(final_delta_rot_sum, p=2, dim=-1)

        deformed_gaussians = gaussians.clone()
        deformed_gaussians._xyz = posed_xyz + final_delta_xyz    
        deformed_gaussians._scaling = gaussians._scaling + final_delta_scale
        deformed_gaussians._rotation = quaternion_multiply(final_delta_rot, gaussians.get_rotation)

        if self.feature_dim > 0:
            setattr(deformed_gaussians, "non_rigid_feature", final_feature)

        return deformed_gaussians, {}
    
    def regularization(self):
        if self.influence_weights is None:
            return {}
        if hasattr(self, 'sparse_mlp_info') and self.sparse_mlp_info:
            loss_coherence = self._compute_intra_expert_coherence_loss(
                self.sparse_mlp_info
            )
        else:
            loss_coherence = torch.tensor(0.0, device=self.influence_weights.device)
        # ----------------------------------------
        return {
            'loss_coherence': loss_coherence,
        }
    
def get_non_rigid_deform(cfg, metadata):
    name = cfg.name
    model_dict = {
        "identity": Identity,
        "gmm_field": GMMNonRigidField,
    }
    return model_dict[name](cfg, metadata)