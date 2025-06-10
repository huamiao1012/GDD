from mmdet.registry import MODELS
from mmengine.model import BaseModule
from .reins import Reins
from .dino_v2 import DinoVisionTransformer
from .utils import set_requires_grad, set_train
import torch.nn.functional as F

@MODELS.register_module()
class ReinsDinoVisionTransformer(DinoVisionTransformer):
    def __init__(
        self,
        reins_config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reins: Reins = MODELS.build(reins_config)
    def resize_mask(self, mask, patch_size=16):
        # mask: [B, 1, 512, 512]
        downsampled_mask = F.interpolate(mask.unsqueeze(1).float(), size=(512 // patch_size, 512 // patch_size), mode='nearest')
        return downsampled_mask.squeeze(1).flatten(1) 
    
    def forward_features(self, x, masks=None, gt_masks = None):
        B, _, h, w = x.shape
        H, W = h // self.patch_size, w // self.patch_size
        x = self.prepare_tokens_with_masks(x, masks)
        outs = []
        if gt_masks is not None:
            gt_masks = self.resize_mask(gt_masks, self.patch_size)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.reins.forward(
                x,
                idx,
                batch_first=True,
                has_cls_token=True,
                gt_masks = gt_masks
            )
            if idx in self.out_indices:
                outs.append(
                    x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
                )
        return self.reins.return_auto(outs)[0]

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["reins"])
        set_train(self, ["reins"])

    def state_dict(self, destination, prefix, keep_vars):
        state = super().state_dict(destination, prefix, keep_vars)
        keys = [k for k in state.keys() if "rein" not in k]
        for key in keys:
            state.pop(key)
            if key in destination:
                destination.pop(key)
        return state
