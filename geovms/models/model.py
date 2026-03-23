# -------------------------------------------------------------------
# Encoder (Swin Transformers) + Decoder (Multi-Head Self Attention)
#
# References:
# Vaswani, A., et al. (2017). Attention Is All You Need. http://arxiv.org/abs/1706.03762
#
# Liu, Z., et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.
# 2021 IEEE/CVF International Conference on Computer Vision (ICCV), 9992–10002.
# https://doi.org/10.1109/ICCV48922.2021.00986
#
# Huggingface Swin-Transformer: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/swin_transformer.py
# Microsoft Swin-Transformer: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
# Pytorch Swin-Transformer: https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py
# -------------------------------------------------------------------

import math
from typing import Callable, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from timm.layers import PatchEmbed, Mlp, DropPath, to_2tuple, to_ntuple, trunc_normal_, \
    use_fused_attn, resize_rel_pos_bias_table, ndgrid
from timm.models._features import feature_take_indices
from timm.models._manipulate import named_apply
from timm.models.vision_transformer import get_init_weights_vit


class Mlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

def window_partition(
        x: torch.Tensor,
        window_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows, window_size: Tuple[int, int], H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    C = windows.shape[-1]
    x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    return x


def get_relative_position_index(win_h: int, win_w: int):
    # get pair-wise relative position index for each token inside the window
    coords = torch.stack(ndgrid(torch.arange(win_h), torch.arange(win_w)))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += win_h - 1  # shift to start from 0
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)  # Wh*Ww, Wh*Ww


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports shifted and non-shifted windows.
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int,
            head_dim: Optional[int] = None,
            window_size = 8,
            qkv_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
    ):
        """
        Args:
            dim: Number of input channels.
            num_heads: Number of attention heads.
            head_dim: Number of channels per head (dim // num_heads if not set)
            window_size: The height and width of the window.
            qkv_bias:  If True, add a learnable bias to query, key, value.
            attn_drop: Dropout ratio of attention weight.
            proj_drop: Dropout ratio of output.
        """
        super().__init__()
        self.dim = dim
        self.window_size = to_2tuple(window_size)  # Wh, Ww
        win_h, win_w = self.window_size
        self.window_area = win_h * win_w
        self.num_heads = num_heads
        head_dim = head_dim or dim // num_heads
        attn_dim = head_dim * num_heads
        self.scale = head_dim ** -0.5
        self.fused_attn = use_fused_attn(experimental=True)  # NOTE not tested for prime-time yet

        # define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * win_h - 1) * (2 * win_w - 1), num_heads))

        # get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(win_h, win_w), persistent=False)

        self.qkv = nn.Linear(dim, attn_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(attn_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def set_window_size(self, window_size: Tuple[int, int]) -> None:
        """Update window size & interpolate position embeddings
        Args:
            window_size (int): New window size
        """
        window_size = to_2tuple(window_size)
        if window_size == self.window_size:
            return
        self.window_size = window_size
        win_h, win_w = self.window_size
        self.window_area = win_h * win_w
        with torch.no_grad():
            new_bias_shape = (2 * win_h - 1) * (2 * win_w - 1), self.num_heads
            self.relative_position_bias_table = nn.Parameter(
                resize_rel_pos_bias_table(
                    self.relative_position_bias_table,
                    new_window_size=self.window_size,
                    new_bias_shape=new_bias_shape,
            ))
            self.register_buffer("relative_position_index", get_relative_position_index(win_h, win_w), persistent=False)

    def _get_rel_pos_bias(self) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        return relative_position_bias.unsqueeze(0)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.fused_attn:
            attn_mask = self._get_rel_pos_bias()
            if mask is not None:
                num_win = mask.shape[0]
                mask = mask.view(1, num_win, 1, N, N).expand(B_ // num_win, -1, self.num_heads, -1, -1)
                attn_mask = attn_mask + mask.reshape(-1, self.num_heads, N, N)
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn + self._get_rel_pos_bias()
            if mask is not None:
                num_win = mask.shape[0]
                attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.
    """

    def __init__(
            self,
            dim: int,
            input_resolution,
            num_heads: int = 4,
            head_dim: Optional[int] = None,
            window_size= 8,
            shift_size: int = 0,
            always_partition: bool = False,
            dynamic_mask: bool = False,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = nn.LayerNorm,
    ):
        """
        Args:
            dim: Number of input channels.
            input_resolution: Input resolution.
            window_size: Window size.
            num_heads: Number of attention heads.
            head_dim: Enforce the number of channels per head
            shift_size: Shift size for SW-MSA.
            always_partition: Always partition into full windows and shift
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            proj_drop: Dropout rate.
            attn_drop: Attention dropout rate.
            drop_path: Stochastic depth rate.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.target_shift_size = to_2tuple(shift_size)  # store for later resize
        self.always_partition = always_partition
        self.dynamic_mask = dynamic_mask
        self.window_size, self.shift_size = self._calc_window_shift(window_size, shift_size)
        self.window_area = self.window_size[0] * self.window_size[1]
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            num_heads=num_heads,
            head_dim=head_dim,
            window_size=self.window_size,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.register_buffer(
            "attn_mask",
            None if self.dynamic_mask else self.get_attn_mask(),
            persistent=False,
        )

    def get_attn_mask(self, x: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        if any(self.shift_size):
            # calculate attention mask for SW-MSA
            if x is not None:
                H, W = x.shape[1], x.shape[2]
                device = x.device
                dtype = x.dtype
            else:
                H, W = self.input_resolution
                device = None
                dtype = None
            H = math.ceil(H / self.window_size[0]) * self.window_size[0]
            W = math.ceil(W / self.window_size[1]) * self.window_size[1]
            img_mask = torch.zeros((1, H, W, 1), dtype=dtype, device=device)  # 1 H W 1
            cnt = 0
            for h in (
                    (0, -self.window_size[0]),
                    (-self.window_size[0], -self.shift_size[0]),
                    (-self.shift_size[0], None),
            ):
                for w in (
                        (0, -self.window_size[1]),
                        (-self.window_size[1], -self.shift_size[1]),
                        (-self.shift_size[1], None),
                ):
                    img_mask[:, h[0]:h[1], w[0]:w[1], :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_area)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        return attn_mask

    def _calc_window_shift(
            self,
            target_window_size: Union[int, Tuple[int, int]],
            target_shift_size: Optional[Union[int, Tuple[int, int]]] = None,
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        target_window_size = to_2tuple(target_window_size)
        if target_shift_size is None:
            # if passed value is None, recalculate from default window_size // 2 if it was previously non-zero
            target_shift_size = self.target_shift_size
            if any(target_shift_size):
                target_shift_size = (target_window_size[0] // 2, target_window_size[1] // 2)
        else:
            target_shift_size = to_2tuple(target_shift_size)

        if self.always_partition:
            return target_window_size, target_shift_size

        window_size = [r if r <= w else w for r, w in zip(self.input_resolution, target_window_size)]
        shift_size = [0 if r <= w else s for r, w, s in zip(self.input_resolution, window_size, target_shift_size)]
        return tuple(window_size), tuple(shift_size)

    def set_input_size(
            self,
            feat_size: Tuple[int, int],
            window_size: Tuple[int, int],
            always_partition: Optional[bool] = None,
    ):
        """
        Args:
            feat_size: New input resolution
            window_size: New window size
            always_partition: Change always_partition attribute if not None
        """
        self.input_resolution = feat_size
        if always_partition is not None:
            self.always_partition = always_partition
        self.window_size, self.shift_size = self._calc_window_shift(window_size)
        self.window_area = self.window_size[0] * self.window_size[1]
        self.attn.set_window_size(self.window_size)
        self.register_buffer(
            "attn_mask",
            None if self.dynamic_mask else self.get_attn_mask(),
            persistent=False,
        )

    def _attn(self, x):
        B, H, W, C = x.shape

        # cyclic shift
        has_shift = any(self.shift_size)
        if has_shift:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        else:
            shifted_x = x

        # pad for resolution not divisible by window size
        pad_h = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_w = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
        _, Hp, Wp, _ = shifted_x.shape

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        if getattr(self, 'dynamic_mask', False):
            attn_mask = self.get_attn_mask(shifted_x)
        else:
            attn_mask = self.attn_mask
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C
        shifted_x = shifted_x[:, :H, :W, :].contiguous()

        # reverse cyclic shift
        if has_shift:
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        else:
            x = shifted_x
        return x

    def forward(self, x):
        B, H, W, C = x.shape
        x = x + self.drop_path1(self._attn(self.norm1(x)))
        x = x.reshape(B, -1, C)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        x = x.reshape(B, H, W, C)
        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer.
    """

    def __init__(
            self,
            dim: int,
            out_dim: Optional[int] = None,
            norm_layer: Callable = nn.LayerNorm,
    ):
        """
        Args:
            dim: Number of input channels.
            out_dim: Number of output channels (or 2 * dim if None)
            norm_layer: Normalization layer.
        """
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim or 2 * dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, self.out_dim, bias=False)

    def forward(self, x):
        B, H, W, C = x.shape

        pad_values = (0, 0, 0, H % 2, 0, W % 2)
        x = nn.functional.pad(x, pad_values)
        _, H, W, _ = x.shape

        x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class SwinTransformerStage(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    """

    def __init__(
            self,
            dim: int,
            out_dim: int,
            input_resolution: Tuple[int, int],
            depth: int,
            downsample: bool = True,
            num_heads: int = 4,
            head_dim: Optional[int] = None,
            window_size = 8,
            always_partition: bool = False,
            dynamic_mask: bool = False,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: Union[List[float], float] = 0.,
            norm_layer: Callable = nn.LayerNorm,
    ):
        """
        Args:
            dim: Number of input channels.
            out_dim: Number of output channels.
            input_resolution: Input resolution.
            depth: Number of blocks.
            downsample: Downsample layer at the end of the layer.
            num_heads: Number of attention heads.
            head_dim: Channels per head (dim // num_heads if not set)
            window_size: Local window size.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            proj_drop: Projection dropout rate.
            attn_drop: Attention dropout rate.
            drop_path: Stochastic depth rate.
            norm_layer: Normalization layer.
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.output_resolution = tuple(i // 2 for i in input_resolution) if downsample else input_resolution
        self.depth = depth
        window_size = to_2tuple(window_size)
        shift_size = tuple([w // 2 for w in window_size])

        # patch merging layer
        if downsample:
            self.downsample = PatchMerging(
                dim=dim,
                out_dim=out_dim,
                norm_layer=norm_layer,
            )
        else:
            assert dim == out_dim
            self.downsample = nn.Identity()

        # build blocks
        self.blocks = nn.Sequential(*[
            SwinTransformerBlock(
                dim=out_dim,
                input_resolution=self.output_resolution,
                num_heads=num_heads,
                head_dim=head_dim,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else shift_size,
                always_partition=always_partition,
                dynamic_mask=dynamic_mask,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
            )
            for i in range(depth)])

    def set_input_size(
            self,
            feat_size: Tuple[int, int],
            window_size: int,
            always_partition: Optional[bool] = None,
    ):
        """ Updates the resolution, window size and so the pair-wise relative positions.

        Args:
            feat_size: New input (feature) resolution
            window_size: New window size
            always_partition: Always partition / shift the window
        """
        self.input_resolution = feat_size
        if isinstance(self.downsample, nn.Identity):
            self.output_resolution = feat_size
        else:
            self.output_resolution = tuple(i // 2 for i in feat_size)
        for block in self.blocks:
            block.set_input_size(
                feat_size=self.output_resolution,
                window_size=window_size,
                always_partition=always_partition,
            )

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x



class SwinTransformerEncoder(nn.Module):
    """ Swin Transformer

    A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    """

    def __init__(
            self,
            img_size = 64,
            patch_size: int = 4,
            in_chans: int = 3,
            embed_dim: int = 96,
            depths: Tuple[int, ...] = (2, 2, 6, 2),
            num_heads: Tuple[int, ...] = (3, 6, 12, 24),
            head_dim: Optional[int] = None,
            window_size = 8,
            always_partition: bool = False,
            strict_img_size: bool = True,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.1,
            embed_layer: Callable = PatchEmbed,
            norm_layer: Union[str, Callable] = nn.LayerNorm,
            weight_init: str = '',
            **kwargs,
    ):
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of input image channels.
            embed_dim: Patch embedding dimension.
            depths: Depth of each Swin Transformer layer.
            num_heads: Number of attention heads in different layers.
            head_dim: Dimension of self-attention heads.
            window_size: Window size.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            drop_rate: Dropout rate.
            attn_drop_rate (float): Attention dropout rate.
            drop_path_rate (float): Stochastic depth rate.
            embed_layer: Patch embedding layer.
            norm_layer (nn.Module): Normalization layer.
        """
        super().__init__()
        self.output_fmt = 'NHWC'

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = self.head_hidden_size = int(embed_dim * 2 ** (self.num_layers - 1))
        self.feature_info = []

        if not isinstance(embed_dim, (tuple, list)):
            embed_dim = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]

        # split image into non-overlapping patches
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim[0],
            norm_layer=norm_layer,
            strict_img_size=strict_img_size,
            output_fmt='NHWC',
        )
        patch_grid = self.patch_embed.grid_size

        # build layers
        head_dim = to_ntuple(self.num_layers)(head_dim)
        if not isinstance(window_size, (list, tuple)):
            window_size = to_ntuple(self.num_layers)(window_size)
        elif len(window_size) == 2:
            window_size = (window_size,) * self.num_layers
        assert len(window_size) == self.num_layers
        mlp_ratio = to_ntuple(self.num_layers)(mlp_ratio)
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        layers = []
        in_dim = embed_dim[0]
        scale = 1
        for i in range(self.num_layers):
            out_dim = embed_dim[i]
            layers += [SwinTransformerStage(
                dim=in_dim,
                out_dim=out_dim,
                input_resolution=(
                    patch_grid[0] // scale,
                    patch_grid[1] // scale
                ),
                depth=depths[i],
                downsample=i > 0,
                num_heads=num_heads[i],
                head_dim=head_dim[i],
                window_size=window_size[i],
                always_partition=always_partition,
                dynamic_mask=not strict_img_size,
                mlp_ratio=mlp_ratio[i],
                qkv_bias=qkv_bias,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            )]
            in_dim = out_dim
            if i > 0:
                scale *= 2
            self.feature_info += [dict(num_chs=out_dim, reduction=patch_size * scale, module=f'layers.{i}')]
        self.layers = nn.Sequential(*layers)

        self.norm = norm_layer(self.num_features)

        if weight_init != 'skip':
            self.init_weights(weight_init)

    @torch.jit.ignore
    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        named_apply(get_init_weights_vit(mode, head_bias=head_bias), self)

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = set()
        for n, _ in self.named_parameters():
            if 'relative_position_bias_table' in n:
                nwd.add(n)
        return nwd

    def set_input_size(
            self,
            img_size: Optional[Tuple[int, int]] = None,
            patch_size: Optional[Tuple[int, int]] = None,
            window_size: Optional[Tuple[int, int]] = None,
            window_ratio: int = 8,
            always_partition: Optional[bool] = None,
    ) -> None:
        """ Updates the image resolution and window size.

        Args:
            img_size: New input resolution, if None current resolution is used
            patch_size (Optional[Tuple[int, int]): New patch size, if None use current patch size
            window_size: New window size, if None based on new_img_size // window_div
            window_ratio: divisor for calculating window size from grid size
            always_partition: always partition into windows and shift (even if window size < feat size)
        """
        if img_size is not None or patch_size is not None:
            self.patch_embed.set_input_size(img_size=img_size, patch_size=patch_size)
            patch_grid = self.patch_embed.grid_size

        if window_size is None:
            window_size = tuple([pg // window_ratio for pg in patch_grid])

        for index, stage in enumerate(self.layers):
            stage_scale = 2 ** max(index - 1, 0)
            stage.set_input_size(
                feat_size=(patch_grid[0] // stage_scale, patch_grid[1] // stage_scale),
                window_size=window_size,
                always_partition=always_partition,
            )

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^patch_embed',  # stem and embed
            blocks=r'^layers\.(\d+)' if coarse else [
                (r'^layers\.(\d+).downsample', (0,)),
                (r'^layers\.(\d+)\.\w+\.(\d+)', None),
                (r'^norm', (99999,)),
            ]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for l in self.layers:
            l.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head.fc

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        self.head.reset(num_classes, pool_type=global_pool)

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            norm: Apply norm layer to compatible intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs
            intermediates_only: Only return intermediate features
        Returns:

        """
        assert output_fmt in ('NCHW',), 'Output shape must be NCHW.'
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.layers), indices)

        # forward pass
        x = self.patch_embed(x)

        num_stages = len(self.layers)
        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            stages = self.layers
        else:
            stages = self.layers[:max_index + 1]
        for i, stage in enumerate(stages):
            x = stage(x)
            if i in take_indices:
                if norm and i == num_stages - 1:
                    x_inter = self.norm(x)  # applying final norm last intermediate
                else:
                    x_inter = x
                x_inter = x_inter.permute(0, 3, 1, 2).contiguous()
                intermediates.append(x_inter)

        if intermediates_only:
            return intermediates

        x = self.norm(x)

        return x, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ):
        """ Prune layers not required for specified intermediates.
        """
        take_indices, max_index = feature_take_indices(len(self.layers), indices)
        self.layers = self.layers[:max_index + 1]  # truncate blocks
        if prune_norm:
            self.norm = nn.Identity()
        if prune_head:
            self.reset_classifier(0, '')
        return take_indices

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.layers(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    # def forward(self, x):
    #     x = self.forward_features(x)
    #     x = self.forward_head(x)
    #     return x

    def forward(self, x):
        return self.forward_intermediates(x)


class MultiheadAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super(MultiheadAttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, channels, height * width).permute(0, 2, 1)  # [B, N(H*W), C]
        attn_output, _ = self.attn(x, x, x)
        attn_output = self.norm(attn_output)
        attn_output = attn_output.permute(0, 2, 1).view(batch_size, channels, height, width)
        return attn_output


class MultiheadAttentionDecoder(nn.Module):
    def __init__(self, features, embed_dim, out_channels, num_heads, act_fn=nn.GELU):
        super(MultiheadAttentionDecoder, self).__init__()
        self.activation_fn = act_fn()

        # Layers for reconstruction.
        channels_reconstruction = [f.shape[1] for f in features]
        channels_reconstruction = channels_reconstruction[::-1]
        channels_reconstruction.append(embed_dim)
        self.attention_blocks = nn.ModuleList([
            MultiheadAttentionBlock(dim=channels_reconstruction[i], num_heads=num_heads[i])
            for i in range(len(features))
        ])
        self.upsample_layers = nn.ModuleList([
            nn.ConvTranspose2d(channels_reconstruction[i], channels_reconstruction[i+1], kernel_size=2, stride=2)
            for i in range(len(features))
        ])
        self.final_upsample = nn.ConvTranspose2d(channels_reconstruction[-1], out_channels, kernel_size=2, stride=2)

    def forward(self, features):
        x = features[-1]  # Start from the deepest feature (smallest spatial resolution)
        n_att_blocks = len(self.attention_blocks)
        for i in range(len(self.upsample_layers)):
            if i < n_att_blocks:
                x = self.attention_blocks[i](x)
            x = self.upsample_layers[i](x)
            x = self.activation_fn(x)
            # skip connection and normalization (LayerNorm by default)
            if i < n_att_blocks-1:
                x += features[-2-i]
                N, C, H, W = x.shape
                x = nn.LayerNorm([C, H, W]).to(x.device)(x)

        x = self.final_upsample(x)
        output = torch.sigmoid(x)
        return output


class Generator(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim,
                 depths, num_heads, window_size, mlp_ratio, out_channels):
        super(Generator, self).__init__()
        self.img_size = to_2tuple(img_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_layers = len(depths)
        self.encoder = SwinTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio
        )

        # initialize the decoder
        _tmp_data = torch.rand(1, in_chans, self.img_size[0], self.img_size[1])
        output, features = self.encoder(_tmp_data)
        self.decoder = MultiheadAttentionDecoder(
            features=features,
            embed_dim=embed_dim,
            out_channels=out_channels,
            num_heads=num_heads,
            act_fn=nn.GELU,
        )

    def forward(self, x):
        x, features = self.encoder(x)
        out = self.decoder(features)
        return out
