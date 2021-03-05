from typing import KeysView
import numpy as np
import torch
import torch.nn.functional as functional
from inplace_abn import active_group, set_active_group
from random import randint
from cirtorch.modules.losses import triplet_loss, contrastive_loss
from cirtorch.utils.misc import Empty
from cirtorch.utils.parallel import PackedSequence

from cirtorch.geometry.conversions import (
    normalize_pixel_coordinates, 
    denormalize_pixel_coordinates,
    convert_points_to_homogeneous,
    convert_points_from_homogeneous)

from cirtorch.geometry.linalg import (
    project_points, relative_transformation, 
    transform_points, 
    inverse_transformation,
    unproject_points)



from cirtorch.geometry.epipolar.essential import essential_from_Rt
from cirtorch.geometry.epipolar.fundamental import (
    fundamental_from_essential, 
    compute_correspond_epilines)


class localFeatureLoss:
    """
        Local Features loss
    """
    def __init__(self, name=None, gamma=1.0, epipolar_margin=0.5, cyclic_margin=0.025, std=1):

        self.name = name
        self.gamma = gamma

        self.std = std

        self.epipolar_margin = epipolar_margin
        self.cyclic_margin = cyclic_margin
    
    def set_weight(self, std, mask=None, regularizer=0.0):
        
        if self.std:
            inverse_std = 1. / torch.clamp( std + regularizer, min=1e-10)
            weight = inverse_std / torch.mean(inverse_std)
            weight = weight.detach()  # Bxn
        else:
            weight = torch.ones_like(std)

        if mask is not None:
            weight *= mask.float()
            weight /= (torch.mean(weight) + 1e-8)
        
        return weight

    def epipolar_term(self, x1, x2, F):
        
        # computer epipolar lines 
        epipolar_line_t = compute_correspond_epilines(points=x1, F_mat=F)
       
        x2 = convert_points_to_homogeneous(x2)
 
        # cost = x2 * l
        epipolar_cost = torch.abs(torch.sum(x2 * epipolar_line_t, dim=2)) 

        return epipolar_cost
    
    def consistency_term(self, x_gt, x, r_dist=40):
        """
            compute the  consistency term
        """
        
        distance = torch.norm(x_gt - x, dim=-1)
        
        distance_t = torch.zeros_like(distance)
        
        distance_t[distance < r_dist] = distance[distance < r_dist]
        
        return distance_t
    
    def __call__(self, kpts, F, P, x1_coord, x2_coord, std_1, std_2, img_size_1, img_size_2):

        shorter_edge, longer_edge = min(img_size_1), max(img_size_1)
        
        # Compute epipolar term
        epipolar_dist= self.epipolar_term(x1=kpts, x2=x2_coord, F=F)
            
        epip_msk = (epipolar_dist < (shorter_edge * self.epipolar_margin))
        epip_weights = self.set_weight(std_2, mask=epip_msk)

        epipolar_loss = torch.mean(epipolar_dist * epip_weights) / longer_edge

        # Compute consistency term
        consistency_dis = self.consistency_term(x_gt=kpts, x=x1_coord)

        cyclic_msk = (epipolar_dist < (shorter_edge * self.cyclic_margin))
        cyclic_weights = self.set_weight(std_1 * std_2, mask=cyclic_msk)

        consistency_loss = torch.mean(consistency_dis * cyclic_weights) / longer_edge

        # Compute total loss
        loss = epipolar_loss + self.gamma * consistency_loss
        std_loss = torch.mean(std_1 * std_2)

        return loss


class localFeatureAlgo:
    """
        Local Feature algorithms

    """
    def __init__(self, loss, min_level, fpn_levels):

        self.loss = loss

        self.min_level = min_level
        self.fpn_levels = fpn_levels

    def _get_level(self, x):
        if isinstance(x, list):
            x = x[self.min_level:self.min_level + self.fpn_levels]
        elif isinstance(x, dict):
            x = x["mod5"]
        else:
            raise NameError("unknown input type")

        return x

    def generate_grid(self, h_min, h_max, w_min, w_max, len_h, len_w, device):

        x, y = torch.meshgrid([
            torch.linspace(w_min, w_max, len_w), 
            torch.linspace(h_min, h_max, len_h)]
            )

        grid = torch.stack((x, y), -1).transpose(0, 1).reshape(-1, 2).float().to(device=device)
        
        return grid
    
    def compute_probability(self, x1_desc, x2_desc, mode="distance"):
        """
            compute probability
            return: probability map [batch_size, m, n]
        """
        # TODO: to config.ini
        assert mode in ['correlation', 'distance']

        if mode == 'correlation':
            corr = x1_desc.bmm(x2_desc.transpose(1, 2)) # dot product
            prob = functional.softmax(corr, dim=-1)  # Bxmxn
        
        else:
            # TODO:  check this part again
            dist = torch.sum(x1_desc**2, dim=-1, keepdim=True) + \
                   torch.sum(x2_desc**2, dim=-1, keepdim=True).transpose(1, 2) - \
                   2 * x1_desc.bmm(x2_desc.transpose(1, 2))  # dot product

            prob = functional.softmax(-dist, dim=-1) 
        
        return prob

    def get_correspondence_locations(self, head, desc, x, with_std=False):
        """
            compute the expected correspondence locations
            return: the normalized expected correspondence locations [batch_size, n_pts, 2]
        """
        B, C, h, w = x.size()

        # Generate grid
        grid_n = self.generate_grid(-1, 1, -1, 1, h, w, device=x.device)
            
        # Generate proposals
        x_desc = x.reshape(B, C, h*w).transpose(1, 2)                             # B(hw)C

        # Run over head and normalize
        x_desc = head.whiten(x_desc)
        x_desc = head.norm(x_desc, dim=2)
            
        # Compute probability map
        prob = self.compute_probability(desc, x_desc)                             # Bxnx(hw)

        grid_n = grid_n.unsqueeze(0).unsqueeze(0)                                   # 1x1x(hw)x2
            
        # Get coordinates
        coord = torch.sum(grid_n * prob.unsqueeze(-1), dim=2)                     # Bxnx2

        # convert to normalized scale [-1, 1]
        var = torch.sum(grid_n**2 * prob.unsqueeze(-1), dim=2) - coord **2         # Bxnx2
        std = torch.sum(torch.sqrt(torch.clamp(var, min=1e-10)), -1)                # Bxn

        if with_std:
            return coord, std
        else:
            return coord

    def get_transformation(self, intrinsics1, intrinsics2, extrinsics1, extrinsics2, aug_trans1=None, aug_trans2=None):
        
        # Unpack
        extrinsics1, extrinsics1_i = extrinsics1.contiguous
        extrinsics2, extrinsics2_i = extrinsics2.contiguous
        
        intrinsics1, intrinsics1_i = intrinsics1.contiguous
        intrinsics2, intrinsics2_i = intrinsics2.contiguous
        
        # Projection x1 -- > x2
        P = relative_transformation(trans_01=extrinsics1, trans_02=extrinsics2)

        # Essentail 
        R_1 = extrinsics1[..., :3, 0:3]  # Nx3x3
        T_1 = extrinsics1[..., :3, 3:4]  # Nx3x1
        
        R_2 = extrinsics2[..., :3, 0:3]  # Nx3x3
        T_2 = extrinsics2[..., :3, 3:4]  # Nx3x1

        E = essential_from_Rt(R1=R_1, t1=T_1, 
                              R2=R_2, t2=T_2)
        
        # Fundamental
        F = fundamental_from_essential(E_mat=E, K1=intrinsics1, K2=intrinsics2)

        return P, E, F

    def normalize_pixels(self, kpts, size):
        """
            from [h, w] --> [-1, 1]
        """

        height, width = size

        kpts_norm = normalize_pixel_coordinates(pixel_coordinates=kpts, height=height, width=width)

        return kpts_norm

    def denormalize_pixels(self, kpts_norm, size):
        """
            from [-1, 1]  -->  [h, w]
        """

        height, width = size

        kpts = denormalize_pixel_coordinates(pixel_coordinates=kpts_norm, height=height, width=width)

        return kpts
    
    def training(self, head, x1, x2, kpts, img_size_1, img_size_2, 
                intrinsics1, intrinsics2, extrinsics1, extrinsics2,
                aug_trans1, aug_trans2):

        # select fpn levels
        x1 = self._get_level(x1)
        x2 = self._get_level(x2)

        try:

            if kpts.all_none:
                raise Empty

            kpts, kpts_i = kpts.contiguous
            kpts_norm = self.normalize_pixels(kpts, img_size_1)

            local_loss = 0.0

            for x1_i, x2_i in zip(x1,x2): 
            
                # Run head
                set_active_group(head, active_group(True))

                # Run Head for image 1 in all levels
                x1_desc = head(x1_i, kpts_norm, img_size_1)
                
                # Get x2 coordinates from x1_desc 
                x2_coord_norm, std_2 = self.get_correspondence_locations(head, desc=x1_desc, x=x2_i, with_std=True)

                # Run Head for image 2 in all levels
                x2_desc = head(x2_i, x2_coord_norm, img_size_2)

                # Get x1 coordinates from x2_desc 
                x1_coord_norm, std_1 = self.get_correspondence_locations(head, desc=x2_desc, x=x1_i, with_std=True)

                # Denormalize for each scale    
                x1_coord = self.denormalize_pixels(x1_coord_norm, img_size_1)
                x2_coord = self.denormalize_pixels(x2_coord_norm, img_size_2)

                # Compute F and P matrices
                P, E, F = self.get_transformation(intrinsics1, intrinsics2, 
                                                extrinsics1, extrinsics2, 
                                                aug_trans1, aug_trans2)
                # Compute Loss
                local_loss += self.loss(kpts, F, P,
                                   x1_coord, x2_coord, 
                                   std_1, std_2, 
                                   img_size_1, img_size_2)
                                   
        except Empty:
            active_group(False)
            local_loss = sum(x_i.sum() for x_i in x1) * 0

        return local_loss

    def inference(self, head, x, kpts, img_size):

        x = self._get_level(x)[0]
        
        try:
            if kpts.all_none:
                raise Empty
            
            kpts, kpts_i = kpts.contiguous 
            
            # Normalize 
            kpts_norm = self.normalize_pixels(kpts, img_size)
            
            local_pred = head(x, kpts_norm, img_size)
            
            local_pred = PackedSequence(local_pred)  
                    
        except Empty:
            local_pred = PackedSequence([None for _ in range(x[0].size(0))])

        return local_pred