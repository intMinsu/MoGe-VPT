import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
from typing import Union, List, Tuple, Optional
from .utils import wrap_dinov2_attention_with_sdpa, wrap_module_with_gradient_checkpointing

class VPTDINOv2Encoder(nn.Module):
    """
    DINOv2 encoder with shallow Visual Prompt Tuning (VPT):
    - Uses ViT 'register_tokens' as prompt tokens (no pos-enc).
    - Keeps a per-scene ParameterDict of prompts.
    - Everything else can be frozen for light test-time adaptation.
    """

    backbone: nn.Module
    image_mean: torch.Tensor
    image_std: torch.Tensor
    dim_features: int

    def __init__(self,
                 backbone: str,
                 intermediate_layers: Union[int, List[int]],
                 dim_out: int,
                 vpt_num_tokens: int = 8,
                 vpt_init: str = "zeros",   # "zeros" | "randn"
                 freeze_backbone: bool = True,
                 ):
        super().__init__()
        self.intermediate_layers = intermediate_layers
        self.vpt_num_tokens = int(vpt_num_tokens)
        self.freeze_backbone = freeze_backbone

        # Build backbone WITH register tokens for VPT
        hub = importlib.import_module(".dinov2.hub.backbones", __package__)
        self.hub_loader = getattr(hub, backbone)  # e.g. dinov2_vitl14
        self.backbone_name = backbone
        self.backbone = self.hub_loader(pretrained=False, num_register_tokens=self.vpt_num_tokens)

        # dims & layer fusion heads (same as original)
        self.dim_features = self.backbone.blocks[0].attn.qkv.in_features
        self.num_features = intermediate_layers if isinstance(intermediate_layers, int) else len(intermediate_layers)
        self.output_projections = nn.ModuleList([
            nn.Conv2d(self.dim_features, dim_out, kernel_size=1, stride=1, padding=0)
            for _ in range(self.num_features)
        ])

        # input normalization
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("image_std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

        # -------- VPT prompts (per-scene) --------
        # We keep a dict of prompts and a pointer to the currently active scene id.
        self.scene_prompts = nn.ParameterDict()
        self._active_scene_id: Optional[str] = None

        # Create a default/global prompt so the module is usable immediately
        self.add_scene_prompts("default", init=vpt_init)
        self.set_active_scene("default")

        # Init backbone weights (ignore missing/extra keys for register_tokens)
        self.init_weights(strict=False)

        # Optionally freeze everything except the active prompts and the 1x1 projections
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        # you typically keep the 1x1 heads trainable, but you can freeze them too if desired
        # for p in self.output_projections.parameters(): p.requires_grad_(False)

    # ---- Public VPT API ----
    def add_scene_prompts(self, scene_id: str, init: str = "zeros"):
        """Create per-scene prompt tokens (1, N, C) and register them."""
        C = self.dim_features
        N = self.vpt_num_tokens
        if scene_id in self.scene_prompts:
            return
        if init == "zeros":
            w = torch.zeros(1, N, C)
        else:
            w = torch.randn(1, N, C) * 0.02
        self.scene_prompts[scene_id] = nn.Parameter(w)

    def set_active_scene(self, scene_id: str):
        """Select which per-scene prompts will be used on forward()."""
        if scene_id not in self.scene_prompts:
            raise KeyError(f"Unknown scene_id '{scene_id}'. Call add_scene_prompts() first.")
        self._active_scene_id = scene_id
        # plug into backbone so prepare_tokens_with_masks() will use them
        # NOTE: assign as nn.Parameter to keep it in the module graph & optimizer
        self.backbone.register_tokens = self.scene_prompts[scene_id]

    # ---- MoGe utility parity ----
    @property
    def onnx_compatible_mode(self):
        return getattr(self, "_onnx_compatible_mode", False)

    @onnx_compatible_mode.setter
    def onnx_compatible_mode(self, v: bool):
        self._onnx_compatible_mode = v
        self.backbone.onnx_compatible_mode = v

    def init_weights(self, strict: bool = False):
        """Load DINOv2 pretrained weights, ignoring the new register_tokens if needed."""
        sd = self.hub_loader(pretrained=True, num_register_tokens=0).state_dict()
        self.backbone.load_state_dict(sd, strict=strict)

    def enable_gradient_checkpointing(self):
        for i in range(len(self.backbone.blocks)):
            wrap_module_with_gradient_checkpointing(self.backbone.blocks[i])

    def enable_pytorch_native_sdpa(self):
        for i in range(len(self.backbone.blocks)):
            wrap_dinov2_attention_with_sdpa(self.backbone.blocks[i].attn)

    # ---- Forward identical to DINOv2Encoder (now with prompts injected by backbone) ----
    def forward(self, image: torch.Tensor,
                token_rows: Union[int, torch.LongTensor],
                token_cols: Union[int, torch.LongTensor],
                return_class_token: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        # ensure some active prompts exist
        if self._active_scene_id is None:
            self.set_active_scene("default")

        img = F.interpolate(image, (token_rows * 14, token_cols * 14),
                            mode="bilinear", align_corners=False,
                            antialias=not self.onnx_compatible_mode)
        img = (img - self.image_mean) / self.image_std

        # pull selected layers; CLS token comes from the last selected layer
        feats = self.backbone.get_intermediate_layers(img, n=self.intermediate_layers, return_class_token=True)

        x = torch.stack([
            proj(feat.permute(0, 2, 1).unflatten(2, (token_rows, token_cols)).contiguous())
            for proj, (feat, cls_tok) in zip(self.output_projections, feats)
        ], dim=1).sum(dim=1)

        if return_class_token:
            return x, feats[-1][1]
        else:
            return x
