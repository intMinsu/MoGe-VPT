import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
import math
from typing import Union, List, Tuple, Optional, Sequence, Callable

from .utils import wrap_dinov2_attention_with_sdpa, wrap_module_with_gradient_checkpointing

def _flatten_vit_blocks(backbone: nn.Module) -> list[nn.Module]:
    blocks = []
    for chunk in backbone.blocks:
        if isinstance(chunk, nn.ModuleList):  # chunked
            for b in chunk:
                if not isinstance(b, nn.Identity):
                    blocks.append(b)
        else:
            blocks.append(chunk)
    return blocks

def last_n_indices(backbone: nn.Module, n: int) -> list[int]:
    flat = _flatten_vit_blocks(backbone)
    n = min(n, len(flat))
    return list(range(len(flat) - n, len(flat)))

class VPTDINOv2Encoder(nn.Module):
    """
    DINOv2 ViT encoder with **multi-route, per-scene test-time adaptation**:

      • **Shallow VPT** (global prompt): a single prompt tensor of shape (1, P, C)
        is inserted once at the encoder input (after CLS) and flows through all blocks.

      • **Deep VPT** (per-block prompts): for a selected set of transformer blocks,
        we insert block-specific prompt tokens (1, P, C) *only inside that block* via
        forward pre/post hooks:
            pre-hook  : concat [CLS | prompt | tokens]
            post-hook : strip   [CLS | prompt] → restore the original sequence length

        Prompts are stored **per scene** and activated by `set_active_scene(scene_id)`.

      • **KV-Prefix (P-Tuning v2-style)**: for a selected set of blocks, the attention
        module is **wrapped** so that **K/V prefixes** (1, H, Pkv, Dh) are **prepended**
        to the computed K/V for the current scene (H: heads, Dh: head dim).
        This biases attention without changing token counts. Uses xFormers MEA if available,
        otherwise falls back to PyTorch SDPA.

      • **LoRA** (low-rank adapters): specified Linear layers inside selected blocks are
        replaced by a LoRA wrapper:  y = base(x) + scaling * B(A(drop(x))).
        The base weights are frozen; only A/B are trainable.

    Per-scene state
    ---------------
    - `scene_prompts[scene_id]`     : shallow prompt (1, P, C) as Parameter
    - `_deep_prompts[scene_id]`     : ParameterDict{ "blk_idx" -> Parameter(1, P, C) }
    - `_kv_store[scene_id]`         : ModuleDict{ "blk_idx" -> ParameterDict{K,V} }
                                      with K,V ∈ (1, H, Pkv, Dh)

    Configuration entry points
    --------------------------
    - `configure_deep_vpt(blocks | last_n, prompt_len, init)`:
        selects blocks & prompt length; hooks are installed lazily on first use.
    - `configure_kv_prefix(blocks | last_n, Pkv)`:
        wraps attention in those blocks to prepend K/V prefixes.
    - `inject_lora_backbone(blocks | last_n, targets, r, alpha, drop)`:
        replaces target Linear layers with LoRA; returns trainable params.

    Optimizer groups
    ----------------
    Use `trainable_prompt_params(scene_id)` to collect grouped parameters:
      {"shallow": [...], "deep": [...], "kv": [...], "lora": [...]}
    so you can set different learning rates/weight decays per adaptation route.

    Freezing and efficiency
    -----------------------
    Pass `freeze_backbone=True` to freeze all backbone weights by default; only prompts,
    KV prefixes, and/or LoRA trainables receive gradients. You can also enable
    gradient checkpointing (`enable_gradient_checkpointing`) and SDPA
    (`enable_pytorch_native_sdpa`) for memory and speed.

    Shapes / Conventions
    --------------------
    - Input images are internally resized to (token_rows*14, token_cols*14) to match ViT-L/14.
    - `get_intermediate_layers_with_prompts(...)` mirrors DINOv2 behavior but with prompts.
    - After the encoder, patch tokens are projected by 1x1 heads and summed across levels.

    Notes
    -----
    - All per-scene tensors are created on the same device/dtype as the backbone’s parameters.
    - Deep VPT strictly preserves sequence length across blocks by removing the inserted prompts
      in the post-hook. Attention shapes remain unchanged downstream.
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

        # Build backbone
        hub = importlib.import_module(".dinov2.hub.backbones", __package__)
        self.hub_loader = getattr(hub, backbone)  # e.g. dinov2_vitl14
        self.backbone_name = backbone
        self.backbone = self.hub_loader(pretrained=False)

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
        self.prompt_len = self.vpt_num_tokens
        self.prompt_dim = self.backbone.embed_dim
        self.vpt_init = vpt_init

        # ====== New adaptive prompting / LoRA state ======
        # Deep VPT
        self._deep_cfg = None                      # dict: {'blocks': List[int], 'P': int, 'init': str}
        self._deep_prompts = nn.ModuleDict()       # scene_id -> ModuleDict{ str(blk_idx): Parameter(1,P,C) }
        self._deep_hooks: list[torch.utils.hooks.RemovableHandle] = []

        # KV-prefix
        self._kv_cfg = None                        # dict: {'blocks': List[int], 'Pkv': int}
        self._kv_store = nn.ModuleDict()           # scene_id -> ModuleDict{ str(blk_idx): ModuleList([Kp,Vp]) }
        self._kv_wrapped = False                   # have we swapped attentions?

        # LoRA
        self._lora_params: list[nn.Parameter] = [] # trainables added by LoRA


        # Create a default/global prompt so the module is usable immediately
        self.add_scene_prompts("default")
        self.set_active_scene("default")

        # Init backbone weights (ignore missing/extra keys for register_tokens)
        # self.init_weights()

        # Optionally freeze everything except the active prompts and the 1x1 projections
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        # you typically keep the 1x1 heads trainable, but you can freeze them too if desired
        # for p in self.output_projections.parameters(): p.requires_grad_(False)

    # ---- Public VPT API ----
    def _param_device_dtype(self):
        p = next(self.backbone.parameters(), None)
        if p is None:
            # fallback to pos_embed buffer
            p = self.backbone.pos_embed
        return p.device, p.dtype

    def add_scene_prompts(self, scene_id: str):
        """Create per-scene prompt tokens (1, N, C) and register them."""
        C = self.dim_features
        N = self.vpt_num_tokens
        if scene_id in self.scene_prompts:
            return
        device, dtype = self._param_device_dtype()
        if self.vpt_init == "zeros":
            w = torch.zeros(1, N, C, dtype=dtype, device=device)
        elif self.vpt_init == "rand":
            w = torch.randn(1, N, C, dtype=dtype, device=device) * 0.02
        else:
            raise ValueError(f"Not supported initialization for scene prompts {self.vpt_init}.")
        # We pass freeze_backbone=True so all parameters do not have grad by default.
        self.scene_prompts[scene_id] = nn.Parameter(w, requires_grad=True)

    def set_active_scene(self, scene_id: str):
        """Select which per-scene prompts are used on forward()."""
        if scene_id not in self.scene_prompts:
            self.add_scene_prompts(scene_id)
        # allocate deep/KV per-scene if configured
        self._ensure_deep_scene(scene_id)
        self._ensure_kv_scene(scene_id)
        self._active_scene_id = scene_id
        # (re)install deep hooks lazily
        if self._deep_cfg is not None and not self._deep_hooks:
            self._install_deep_vpt_hooks()

    def trainable_prompt_params(self, scene_id: str) -> dict[str, list[nn.Parameter]]:
        """
        Group params by adaptation type for nice optimizer groups.
        Keys: 'shallow', 'deep', 'kv', 'lora'
        """
        groups: dict[str, list[nn.Parameter]] = {"shallow": [], "deep": [], "kv": [], "lora": []}
        if scene_id in self.scene_prompts:
            groups["shallow"].append(self.scene_prompts[scene_id])
        if scene_id in self._deep_prompts:
            groups["deep"] += list(self._deep_prompts[scene_id].parameters())
        if scene_id in self._kv_store:
            groups["kv"]  += list(self._kv_store[scene_id].parameters())
        if self._lora_params:
            groups["lora"] += self._lora_params
        return {k: v for k, v in groups.items() if v}

    # --------- Deep VPT (per-block prompt tokens) ----------
    def configure_deep_vpt(
            self,
            blocks: Sequence[int] | None = None,
            *, last_n: int | None = None,
            prompt_len: int = 8,
            init: str = "zeros"
    ):
        """
        Select blocks for deep-VPT and set prompt length.
        Call once before training. Prompts are allocated per-scene on-demand.
        """
        if blocks is None:
            assert last_n is not None, "Provide either blocks or last_n."
            blocks = last_n_indices(self.backbone, last_n)
        self._deep_cfg = {"blocks": list(map(int, blocks)), "P": int(prompt_len), "init": str(init)}
        # (re)install hooks if already active
        if self._deep_hooks:
            self._remove_deep_vpt_hooks()
            self._install_deep_vpt_hooks()

    def _ensure_deep_scene(self, scene_id: str):
        if self._deep_cfg is None or scene_id in self._deep_prompts:
            return

        P = self._deep_cfg["P"]
        C = self.backbone.embed_dim
        init = self._deep_cfg["init"]

        device = self.backbone.pos_embed.device; dtype = self.backbone.pos_embed.dtype

        per_block = nn.ParameterDict()

        for idx in self._deep_cfg["blocks"]:
            if init == "zeros":
                w = torch.zeros(1, P, C)
            elif init in ("rand", "randn"):
                w = torch.randn(1, P, C) * 0.02
            else:
                raise ValueError("init must be 'zeros' or 'rand'")
            per_block[str(int(idx))] = nn.Parameter(
                w.to(device=device,dtype=dtype)
            )
        self._deep_prompts[scene_id] = per_block

    def _get_deep_prompt(self, blk_idx: int, B: int, dtype, device):
        sid = self._active_scene_id

        if sid is None or sid not in self._deep_prompts:
            return None

        param_dict: nn.ParameterDict = self._deep_prompts[sid]
        k = str(int(blk_idx))

        if k not in param_dict:
            return None

        P = param_dict[k]
        return P.to(dtype=dtype, device=device).expand(B, -1, -1)

    def _install_deep_vpt_hooks(self):
        if self._deep_cfg is None: return
        if self._deep_hooks:
            self._remove_deep_vpt_hooks()
        flat = _flatten_vit_blocks(self.backbone)
        for i in self._deep_cfg["blocks"]:
            blk = flat[i]

            def pre(module, inputs, idx=i, self_ref=self):
                (x,) = inputs
                B, T, C = x.shape
                P = self_ref._get_deep_prompt(idx, B, x.dtype, x.device)
                if P is None or P.shape[1] == 0:
                    return None
                x = torch.cat([x[:, :1, :], P, x[:, 1:, :]], dim=1)  # insert after CLS
                module._vpt_p_len = P.shape[1]
                return (x,)

            def post(module, inputs, output):
                p = getattr(module, "_vpt_p_len", 0)
                if p <= 0: return None
                return torch.cat([output[:, :1, :], output[:, 1 + p:, :]], dim=1)

            self._deep_hooks.append(blk.register_forward_pre_hook(pre, with_kwargs=False))
            self._deep_hooks.append(blk.register_forward_hook(post, with_kwargs=False))

    def _remove_deep_vpt_hooks(self):
        for h in self._deep_hooks:
            h.remove()
        self._deep_hooks.clear()

    # ---- MoGe utility parity ----
    @property
    def onnx_compatible_mode(self):
        return getattr(self, "_onnx_compatible_mode", False)

    @onnx_compatible_mode.setter
    def onnx_compatible_mode(self, v: bool):
        self._onnx_compatible_mode = v
        self.backbone.onnx_compatible_mode = v

    def init_weights(self):
        """Load DINOv2 pretrained weights"""
        sd = self.hub_loader(pretrained=True).state_dict()
        self.backbone.load_state_dict(sd, strict=True)

    def enable_gradient_checkpointing(self):
        for i in range(len(self.backbone.blocks)):
            wrap_module_with_gradient_checkpointing(self.backbone.blocks[i])

    def enable_pytorch_native_sdpa(self):
        for i in range(len(self.backbone.blocks)):
            wrap_dinov2_attention_with_sdpa(self.backbone.blocks[i].attn)

    # ---- Forward ----
    def _prepare_tokens_with_prompts(self, x, masks=None):
        # identical to DinoVisionTransformer.prepare_tokens_with_masks, except we insert prompts after pe
        B, nc, h, w = x.shape
        x = self.backbone.patch_embed(x)

        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.backbone.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.backbone.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.backbone.interpolate_pos_encoding(x, h, w)

        # (optional) preserve register tokens if you use them
        if self.backbone.register_tokens is not None:
            x = torch.cat((x[:, :1],
                           self.backbone.register_tokens.expand(x.shape[0], -1, -1),
                           x[:, 1:]), dim=1)

        # ---- VPT prompts here ----
        if self._active_scene_id is not None and (self._active_scene_id in self.scene_prompts):
            P = self.scene_prompts[self._active_scene_id]  # (1, L, C)
            P = P.expand(B, -1, -1)  # batch repeat
            x = torch.cat((x[:, :1], P, x[:, 1:]), dim=1)  # insert after CLS

        return x # [BS, (CLS | PROMPTS | REGISTERS | PATCHES), embed_dim]

    def get_intermediate_layers_with_prompts(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,  # Layers or n last layers to take
            return_class_token: bool =True
    ):
        # mirror DinoVisionTransformer._get_intermediate_layers_* but with our prep
        x = self._prepare_tokens_with_prompts(x)
        outputs = []
        i = 0
        # handle chunked / non-chunked blocks transparently
        blocks_iter = (b for chunk in self.backbone.blocks for b in
                       (chunk if isinstance(chunk, nn.ModuleList) else [chunk]))
        total = sum(1 for _ in blocks_iter)
        blocks_iter = (b for chunk in self.backbone.blocks for b in
                       (chunk if isinstance(chunk, nn.ModuleList) else [chunk]))

        blocks_to_take = range(total - n, total) if isinstance(n, int) else n
        for b in blocks_iter:
            x = b(x)
            if i in blocks_to_take:
                outputs.append(x)
            i += 1

        outputs = [self.backbone.norm(out) for out in outputs]
        cls_tokens = [out[:, 0] for out in outputs]
        patches = [out[:, 1 + self.backbone.num_register_tokens:] for out in outputs]
        if return_class_token:
            return tuple(zip(patches, cls_tokens))
        return tuple(patches)

    def _strip_to_patches(self, feat: torch.Tensor, token_rows: int, token_cols: int) -> torch.Tensor:
        # feat: (B, T_no_cls, C), where T_no_cls may include prompts/registers
        B, T, C = feat.shape
        P = int(token_rows) * int(token_cols)  # number of patch tokens
        if T < P:
            raise RuntimeError(f"Not enough tokens: T={T} < patches={P}")
        # Always take the last P tokens → these are patch tokens
        return feat[:, -P:, :]

    def forward(self,
                image: torch.Tensor,
                token_rows: Union[int, torch.LongTensor],
                token_cols: Union[int, torch.LongTensor],
                return_class_token: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:

        image_14 = F.interpolate(
            image, (token_rows * 14, token_cols * 14),
            mode="bilinear", align_corners=False, antialias=not self.onnx_compatible_mode
        )
        image_14 = (image_14 - self.image_mean) / self.image_std

        feats = self.get_intermediate_layers_with_prompts(
            image_14, n=self.intermediate_layers, return_class_token=True
        )
        # strip VPT prompt tokens so only H*W patch tokens remain
        feats = [(self._strip_to_patches(feat, token_rows, token_cols), cls) for (feat, cls) in feats]

        # project & unflatten per-layer, then sum
        x = torch.stack([
            proj(feat.permute(0, 2, 1)  # (B, C, T)
                      .unflatten(2, (int(token_rows), int(token_cols)))  # -> (B, C, H, W)
                      .contiguous())
            for proj, (feat, cls_tok) in zip(self.output_projections, feats)
        ], dim=1).sum(dim=1)


        if return_class_token:
            # pass through the last layer's cls token (unchanged)
            return x, feats[-1][1]  # cls token from last layer
        else:
            return x

    class _KVPrefixWrapper(nn.Module):
        """Wrap DINOv2 Attention/MemEffAttention to prepend K/V prefixes."""
        def __init__(self, attn: nn.Module, num_heads: int, head_dim: int,
                     get_scene: Callable[[], Optional[str]], store: nn.ModuleDict, blk_idx: int):
            super().__init__()
            self.attn = attn
            self.H = num_heads
            self.Dh = head_dim
            self.get_scene = get_scene
            self.store = store
            self.blk_idx = int(blk_idx)

        def _get_prefix(self, B: int, dtype, device):
            sid = self.get_scene()

            if sid is None or sid not in self.store: return None, None

            blocks: nn.ModuleDict = self.store[sid]
            k = str(self.blk_idx)

            if k not in blocks:
                return None, None

            kv_dict: nn.ParameterDict = blocks[k]

            Kp = kv_dict["K"].to(dtype=dtype, device=device).expand(B, -1, -1, -1)
            Vp = kv_dict["V"].to(dtype=dtype, device=device).expand(B, -1, -1, -1)

            return Kp, Vp

        def forward(self, x: torch.Tensor, attn_bias=None):
            B, N, C = x.shape
            qkv = self.attn.qkv(x).reshape(B, N, 3, self.H, self.Dh).permute(2,0,3,1,4)
            q, k, v = qkv[0], qkv[1], qkv[2]      # (B,H,N,Dh)
            Kp, Vp = self._get_prefix(B, x.dtype, x.device)
            if Kp is not None:
                k = torch.cat([Kp, k], dim=2)
                v = torch.cat([Vp, v], dim=2)

            # dispatch to xFormers if available else SDPA
            try:
                from xformers.ops import memory_efficient_attention as mea
                out = mea(q, k, v, attn_bias=attn_bias)
            except Exception:
                out = F.scaled_dot_product_attention(q, k, v, attn_bias)

            out = out.permute(0,2,1,3).reshape(B, N, C)
            out = self.attn.proj(out)
            out = self.attn.proj_drop(out)
            return out

    # --- KV-Prefix ---
    def configure_kv_prefix(self, blocks: Sequence[int] | None = None, *, last_n: int | None = None, Pkv: int = 8):
        """
        Replace attention modules on selected blocks with KV-prefix wrappers.
        Call once before training. K/V tensors are allocated per-scene on-demand.
        """
        if blocks is None:
            assert last_n is not None, "Provide either blocks or last_n."
            blocks = last_n_indices(self.backbone, last_n)
        self._kv_cfg = {"blocks": list(map(int, blocks)), "Pkv": int(Pkv)}

        if not self._kv_wrapped:
            flat = _flatten_vit_blocks(self.backbone)
            # infer heads and head_dim
            attn0 = flat[self._kv_cfg["blocks"][0]].attn
            H = attn0.num_heads
            Dh = attn0.qkv.in_features // H
            for i in self._kv_cfg["blocks"]:
                attn = flat[i].attn
                flat[i].attn = self._KVPrefixWrapper(attn, H, Dh,
                                                     get_scene=lambda: self._active_scene_id,
                                                     store=self._kv_store,
                                                     blk_idx=int(i))
            self._kv_wrapped = True

    def _ensure_kv_scene(self, scene_id: str):
        if self._kv_cfg is None or scene_id in self._kv_store:
            return

        Pkv = self._kv_cfg["Pkv"]

        # infer heads/dim from any selected block
        flat = _flatten_vit_blocks(self.backbone)
        attn0 = flat[self._kv_cfg["blocks"][0]].attn.attn if isinstance(flat[self._kv_cfg["blocks"][0]].attn, self._KVPrefixWrapper) else flat[self._kv_cfg["blocks"][0]].attn

        H = attn0.num_heads
        Dh = attn0.qkv.in_features // H
        device = self.backbone.pos_embed.device; dtype = self.backbone.pos_embed.dtype

        scene_moduledict = nn.ModuleDict()

        for i in self._kv_cfg["blocks"]:
            kv_dict = nn.ParameterDict(
                {
                "K": nn.Parameter(torch.zeros(1, H, Pkv, Dh, device=device, dtype=dtype), requires_grad=True),
                "V": nn.Parameter(torch.zeros(1, H, Pkv, Dh, device=device, dtype=dtype), requires_grad=True),
                }
            )
            scene_moduledict[str(int(i))] = kv_dict

        self._kv_store[scene_id] = scene_moduledict

    class _LoRALinear(nn.Module):
        def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0, drop: float = 0.0):
            super().__init__()
            self.base = base
            self.r = r
            self.scaling = alpha / r
            self.A = nn.Linear(base.in_features, r, bias=False)
            self.B = nn.Linear(r, base.out_features, bias=False)
            self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
            nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.B.weight)
            for p in self.base.parameters(): p.requires_grad_(False)
        def forward(self, x):
            return self.base(x) + self.B(self.A(self.drop(x))) * self.scaling

    def inject_lora_backbone(
            self,
            *,
            blocks: Sequence[int] | None = None,
            last_n: int | None = None,
            targets: Sequence[str] = ("attn.qkv","attn.proj","mlp.fc1","mlp.fc2"),
            r: int = 8,
            alpha: float = 16.0,
            drop: float = 0.0
    ) -> list[nn.Parameter]:
        """
        Wrap target Linear layers in selected blocks with LoRA.
        Returns the list of newly trainable parameters.
        """
        if blocks is None:
            assert last_n is not None
            blocks = last_n_indices(self.backbone, last_n)
        flat = _flatten_vit_blocks(self.backbone)
        params: list[nn.Parameter] = []
        for i in blocks:
            blk = flat[i]
            for path in targets:
                obj = blk
                parts = path.split(".")
                for p in parts[:-1]:
                    obj = getattr(obj, p)
                leaf_name = parts[-1]
                layer = getattr(obj, leaf_name, None)
                if not isinstance(layer, nn.Linear):
                    continue
                wrapped = self._LoRALinear(layer, r=r, alpha=alpha, drop=drop)
                setattr(obj, leaf_name, wrapped)
                params += list(wrapped.A.parameters()) + list(wrapped.B.parameters())
        self._lora_params += params
        return params