from typing import *
from .v2 import MoGeModel
from .vpt import VPTDINOv2Encoder

class MoGeModelVPT(MoGeModel):
    def __init__(self,
        encoder: Dict[str, Any],
        neck: Dict[str, Any],
        points_head: Dict[str, Any] = None,
        mask_head: Dict[str, Any] = None,
        normal_head: Dict[str, Any] = None,
        scale_head: Dict[str, Any] = None,
        remap_output: str = 'linear',
        num_tokens_range: List[int] = [1200, 3600],
        vpt_cfg: Optional[Dict[str, Any]] = None,
        **deprecated_kwargs
    ):
        super().__init__(encoder, neck, points_head, mask_head, normal_head, scale_head,
                         remap_output, num_tokens_range, **deprecated_kwargs)
        if vpt_cfg is not None:
            base_enc = self.encoder
            self.encoder = VPTDINOv2Encoder(base_enc, **vpt_cfg)

