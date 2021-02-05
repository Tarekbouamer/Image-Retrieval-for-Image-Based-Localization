import torch
import torch.nn as nn
import torch.nn.functional as F

from .responses import BlobHessian
from ..geometry.subpix.spatial_soft_argmax import ConvSoftArgmax3d
from ..features.orientation import PassLAF
from ..features.laf import (
    denormalize_laf,
    normalize_laf,
    laf_is_inside_image
)
from ..geometry.transform.pyramid import ScalePyramid


def _scale_index_to_scale(max_coords, sigmas, num_levels):
    """
        Auxiliary function for ScaleSpaceDetector. Converts scale level index from ConvSoftArgmax3d
        to the actual scale, using the sigmas from the ScalePyramid output
    """
    # depth (scale) in coord_max is represented as (float) index, not the scale yet.
    # we will interpolate the scale using pytorch.grid_sample function
    # Because grid_sample is for 4d input only, we will create fake 2nd dimension
    # ToDo: replace with 3d input, when grid_sample will start to support it

    # Reshape for grid shape
    B, N, _ = max_coords.shape
    L = sigmas.size(1)
    scale_coords = max_coords[:, :, 0].contiguous().view(-1, 1, 1, 1)

    # Replace the scale_x_y

    out = torch.cat([sigmas[0, 0] * torch.pow(2.0, scale_coords / float(num_levels)).view(B, N, 1),
                     max_coords[:, :, 1:]], dim=2)
    return out


def _create_octave_mask(mask, octave_shape):
    """
        Downsamples a mask based on the given octave shape.
    """
    mask_shape = octave_shape[-2:]
    mask_octave = F.interpolate(mask, mask_shape, mode='bilinear', align_corners=False)
    return mask_octave.unsqueeze(1)


class ScaleSpaceDetector(nn.Module):
    """
        Module for differentiable local feature detection, as close as possible to classical
        local feature detectors like Harris, Hessian-Affine or SIFT (DoG).
        It has 5 modules inside: scale pyramid generator, response ("cornerness") function,
        soft nms function, affine shape estimator and patch orientation estimator.
        Each of those modules could be replaced with learned custom one, as long, as
        they respect output shape.
    """

    def __init__(self, num_features=500, mr_size=6.0, scale_pyr_module=ScalePyramid(3, 1.6, 15),
                 resp_module=BlobHessian(),
                 nms_module=ConvSoftArgmax3d((3, 3, 3),
                                             (1, 1, 1),
                                             (1, 1, 1),
                                             normalized_coordinates=False,
                                             output_value=True),
                 ori_module=PassLAF(),
                 aff_module=PassLAF(),
                 minima_are_also_good=False, scale_space_response=False):

        super(ScaleSpaceDetector, self).__init__()
        self.mr_size = mr_size
        self.num_features = num_features
        self.scale_pyr = scale_pyr_module
        self.resp = resp_module
        self.nms = nms_module
        self.ori = ori_module
        self.aff = aff_module
        self.minima_are_also_good = minima_are_also_good

        # scale_space_response should be True if the response function works on scale space
        # like Difference-of-Gaussians

        self.scale_space_response = scale_space_response

        return

    def detect(self, img, num_feats, mask=None):
        dev = img.device
        dtype = img.dtype
        sp, sigmas, pix_dists = self.scale_pyr(img)

        all_responses = []
        all_lafs = []

        for oct_idx, octave in enumerate(sp):
            sigmas_oct = sigmas[oct_idx]
            pix_dists_oct = pix_dists[oct_idx]

            B, CH, L, H, W = octave.size()

            # Run response function
            if self.scale_space_response:
                oct_resp = self.resp(octave, sigmas_oct.view(-1))
            else:
                oct_resp = self.resp(octave.permute(0, 2, 1, 3, 4).reshape(B * L, CH, H, W),
                                     sigmas_oct.view(-1)).view(B, L, CH, H, W)

                # We want nms for scale responses, so reorder to (B, CH, L, H, W)
                oct_resp = oct_resp.permute(0, 2, 1, 3, 4)

                # 3rd extra level is required for DoG only
                if self.scale_pyr.extra_levels % 2 != 0:  # type: ignore
                    oct_resp = oct_resp[:, :, :-1]

            if mask is not None:
                oct_mask = _create_octave_mask(mask, oct_resp.shape)
                oct_resp = oct_mask * oct_resp

            # Differentiable nms
            coord_max, response_max = self.nms(oct_resp)
            if self.minima_are_also_good:
                coord_min, response_min = self.nms(-oct_resp)
                take_min_mask = (response_min > response_max).to(response_max.dtype)
                response_max = response_min * take_min_mask + (1 - take_min_mask) * response_max
                coord_max = coord_min * take_min_mask.unsqueeze(2) + (1 - take_min_mask.unsqueeze(2)) * coord_max

            # Now, lets crop out some small responses
            responses_flatten = response_max.view(response_max.size(0), -1)  # [B, N]
            max_coords_flatten = coord_max.view(response_max.size(0), 3, -1).permute(0, 2, 1)  # [B, N, 3]

            if responses_flatten.size(1) > num_feats:
                resp_flat_best, idxs = torch.topk(responses_flatten, k=num_feats, dim=1)
                max_coords_best = torch.gather(max_coords_flatten, 1, idxs.unsqueeze(-1).repeat(1, 1, 3))
            else:
                resp_flat_best = responses_flatten
                max_coords_best = max_coords_flatten
            B, N = resp_flat_best.size()

            # Converts scale level index from ConvSoftArgmax3d to the actual scale, using the sigmas
            max_coords_best = _scale_index_to_scale(max_coords_best, sigmas_oct, self.scale_pyr.n_levels)

            # Create local affine frames (LAFs)
            rotmat = torch.eye(2, dtype=dtype, device=dev).view(1, 1, 2, 2)
            current_lafs = torch.cat([self.mr_size * max_coords_best[:, :, 0].view(B, N, 1, 1) * rotmat,
                                      max_coords_best[:, :, 1:3].view(B, N, 2, 1)], dim=3)

            # Zero response lafs, which touch the boundary
            good_mask = laf_is_inside_image(current_lafs, octave[:, 0])
            resp_flat_best = resp_flat_best * good_mask.to(dev, dtype)

            # Normalize LAFs
            current_lafs = normalize_laf(current_lafs, octave[:, 0])  # We don`t need # of scale levels, only shape

            all_responses.append(resp_flat_best)
            all_lafs.append(current_lafs)

        # Sort and keep best n
        responses = torch.cat(all_responses, dim=1)
        lafs = torch.cat(all_lafs, dim=1)
        responses, idxs = torch.topk(responses, k=num_feats, dim=1)
        lafs = torch.gather(lafs, 1, idxs.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 2, 3))

        return responses, denormalize_laf(lafs, img)

    def forward(self, img, mask=None):
        """
            Three stage local feature detection. First the location and scale of interest points are determined by
            detect function. Then affine shape and orientation.

        """
        responses, lafs = self.detect(img, self.num_features, mask)

        lafs = self.aff(lafs, img)

        lafs = self.ori(lafs, img)

        return lafs, responses