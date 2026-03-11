import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
import time
from scipy.io import savemat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat
from functools import partial
from typing import Optional, Callable
import math
import numbers
import sys
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import to_2tuple, trunc_normal_


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()

        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    """
    return pair-wise similarity matrix between two tensors
    :param x1: [B,...,M,D]
    :param x2: [B,...,N,D]
    :return: similarity matrix [B,...,M,N]
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim


class SparseStateSpaceModule(nn.Module):
    def __init__(
            self,
            d_model,
            proposal_hw,
            fold_hw,
            heads,
            d_state=8,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.proposal_hw = proposal_hw
        self.fold_hw = fold_hw
        self.heads = heads
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model) // self.heads
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.x_conv = nn.Conv1d(in_channels=(self.dt_rank + self.d_state * 2),
                                out_channels=(self.dt_rank + self.d_state * 2), kernel_size=7, padding=3,
                                groups=(self.dt_rank + self.d_state * 2))

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        self.f = nn.Conv2d(self.d_inner, self.d_inner * self.heads, kernel_size=1)  # for similarity
        self.proj = nn.Conv2d(self.d_inner * self.heads, self.d_inner, kernel_size=1)  # for projecting channel number
        self.v = nn.Conv2d(self.d_inner, self.d_inner * self.heads, kernel_size=1)  # for value
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        self.centers_proposal = nn.AdaptiveAvgPool2d((self.proposal_hw, self.proposal_hw))


    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x):
        value = self.v(x)
        x = self.f(x)
        x = rearrange(x, "b (e c) w h -> (b e) c w h", e=self.heads)
        value = rearrange(value, "b (e c) w h -> (b e) c w h", e=self.heads)
        if self.fold_hw > 1:
            # split the big feature maps to small local regions to reduce computations.
            b0, c0, w0, h0 = x.shape
            assert w0 % self.fold_hw == 0 and h0 % self.fold_hw == 0, \
                f"Ensure the feature map size ({w0}*{h0}) can be divided by fold {self.fold_hw}*{self.fold_hw}"
            x = rearrange(x, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_hw,
                          f2=self.fold_hw)  # [bs*blocks,c,ks[0],ks[1]]
            value = rearrange(value, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_hw, f2=self.fold_hw)
        b, c, w, h = x.shape
        centers = self.centers_proposal(x)  # [b,c,C_W,C_H], we set M = C_W*C_H and N = w*h
        value_centers = rearrange(self.centers_proposal(value), 'b c w h -> b (w h) c')  # [b,C_W,C_H,c]
        b, c, ww, hh = centers.shape

        sim = torch.sigmoid(
            self.sim_beta +
            self.sim_alpha * pairwise_cos_sim(
                centers.reshape(b, c, -1).permute(0, 2, 1),
                x.reshape(b, c, -1).permute(0, 2, 1)
            )
        )  # [B,M,N]
        # we use mask to sololy assign each point to one center

        sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)
        mask = torch.zeros_like(sim)  # binary #[B,M,N]
        mask.scatter_(1, sim_max_idx, 1.)
        sim = sim * mask
        value2 = rearrange(value, 'b c w h -> b (w h) c')  # [B,N,D]
        # aggregate step, out shape [B,M,D]
        out = ((value2.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)).sum(dim=2) + value_centers) / (sim.sum(dim=-1, keepdim=True) + 1.0)  # [B,M,D]
        B, L, C = out.shape
        K = 1
        xs = rearrange(out, "b l c -> b c l")
        xs = torch.stack([xs], dim=1).view(B, 1, -1, L)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        x_dbl = self.x_conv(x_dbl.squeeze(1)).unsqueeze(1)
        dts, Bs_, Cs_ = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs_ = Bs_.float().view(B, K, -1, L)
        Cs_ = Cs_.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        out_y = self.selective_scan(
            xs, dts, As, Bs_, Cs_, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        out = rearrange(out_y[:,0], "b c l -> b l c")

        out = (out.unsqueeze(dim=2) * sim.unsqueeze(dim=-1)).sum(dim=1)  # [B,N,D]
        out = rearrange(out, "b (w h) c -> b c w h", w=w)

        if self.fold_hw > 1:
            # recover the splited regions back to big feature maps if use the region partition.
            out = rearrange(out, "(b f1 f2) c w h -> b c (f1 w) (f2 h)", f1=self.fold_hw, f2=self.fold_hw)
        out = rearrange(out, "(b e) c w h -> b (e c) w h", e=self.heads)
        out = self.proj(out)
        return out


    def forward(self, x: torch.Tensor, **kwargs):
        x = rearrange(x, 'b c h w -> b h w c')
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y = self.forward_core(x)
        assert y.dtype == torch.float32
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        out = rearrange(out, 'b h w c -> b c h w')
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class FFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias=False):
        super(FFN, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class SparseMambaBlock(nn.Module):
    def __init__(self, use_attn, feature_dim, proposal_hw, fold_hw, heads, ffn_expansion_factor):
        super().__init__()
        self.use_attn = use_attn
        if self.use_attn:   
            self.norm1 = LayerNorm(feature_dim)
            self.SSSM = SparseStateSpaceModule(feature_dim, proposal_hw, fold_hw, heads)
            self.chan_attn = ChannelAttention(feature_dim)
            self.spat_attn = SpatialAttention()
            self.conv_1 = nn.Conv2d(feature_dim, feature_dim, 1, 1, 0)
            self.conv_2 = nn.Conv2d(feature_dim, feature_dim, 1, 1, 0)
        self.norm2 = LayerNorm(feature_dim)
        self.ffn = FFN(feature_dim, ffn_expansion_factor)

    def forward(self, x):
        if self.use_attn: 
            x_norm = self.norm1(x)
            x_ssm = self.SSSM(x_norm)
            x_attn = self.conv_1(self.chan_attn(x_norm) * x_norm) + self.conv_2(self.spat_attn(x_norm) * x_norm)
            x = x_ssm + x_attn + x

        x = x + self.ffn(self.norm2(x))
        return x


class Downsample(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.body = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(4 * in_chan, out_chan, kernel_size=1) 
        )
    
    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_chan, 4 * out_chan, kernel_size=1), 
            nn.PixelShuffle(2)
        )
    
    def forward(self, x):
        return self.body(x)

@ARCH_REGISTRY.register()
class C2SSM(nn.Module):
    def __init__(self,
                 input_dim = 3,
                 output_dim = 3,
                 feature_dim = 32,
                 proposal_hw = [2, 2, 2], 
                 fold_hw = [4, 2, 1], 
                 heads = [1, 2, 4],
                 num_blocks = [2, 4, 4],
                 ffn_expansion_factor = 2.66):
        super().__init__()
        self.input_proj = nn.Conv2d(input_dim, feature_dim, 3, 1, 1)
        self.down = Downsample(feature_dim, feature_dim)
        self.encoder_level1 = nn.Sequential(*[SparseMambaBlock(False, feature_dim, proposal_hw[0], fold_hw[0], heads[0], ffn_expansion_factor) for _ in range(num_blocks[0])])
        self.down1_2 = Downsample(feature_dim, feature_dim * 2)

        self.encoder_level2 = nn.Sequential(*[SparseMambaBlock(False, feature_dim * 2, proposal_hw[1], fold_hw[1], heads[1], ffn_expansion_factor) for _ in range(num_blocks[1])])
        self.down2_3 = Downsample(feature_dim * 2, feature_dim * 4)

        self.encoder_level3 = nn.Sequential(*[SparseMambaBlock(False, feature_dim * 4, proposal_hw[2], fold_hw[2], heads[2], ffn_expansion_factor) for _ in range(num_blocks[2])])
        self.latent = nn.Sequential(*[SparseMambaBlock(False, feature_dim * 4, proposal_hw[2], fold_hw[2], heads[2], ffn_expansion_factor) for _ in range(num_blocks[2])])
        self.decoder_level3 = nn.Sequential(*[SparseMambaBlock(True, feature_dim * 4, proposal_hw[2], fold_hw[2], heads[2], ffn_expansion_factor) for _ in range(num_blocks[2])])
        self.reduce_chan_level3 = nn.Conv2d(int(feature_dim * 8), int(feature_dim * 4), kernel_size=1)

        self.up3_2 = Upsample(feature_dim * 4, feature_dim * 2)
        self.decoder_level2 = nn.Sequential(*[SparseMambaBlock(True, feature_dim * 2, proposal_hw[1], fold_hw[1], heads[1], ffn_expansion_factor) for _ in range(num_blocks[1])])
        self.reduce_chan_level2 = nn.Conv2d(int(feature_dim * 4), int(feature_dim * 2), kernel_size=1)

        self.up2_1 = Upsample(feature_dim * 2, feature_dim)
        self.decoder_level1 = nn.Sequential(*[SparseMambaBlock(True, feature_dim, proposal_hw[0], fold_hw[0], heads[0], ffn_expansion_factor) for _ in range(num_blocks[0])])
        self.reduce_chan_level1 = nn.Conv2d(int(feature_dim * 2), int(feature_dim), kernel_size=1)

        self.up = Upsample(feature_dim, feature_dim)
        self.refine = nn.Sequential(*[SparseMambaBlock(False, feature_dim, proposal_hw[0], fold_hw[0], heads[0], ffn_expansion_factor) for _ in range(num_blocks[1])])
        self.output_proj = nn.Conv2d(feature_dim, output_dim, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        inp_enc_level1 = self.input_proj(x)
        inp_enc_level1 = self.down(inp_enc_level1)

        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        latent = self.latent(out_enc_level3)
        inp_dec_level3 = torch.cat([latent, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(latent)
        inp_dec_level2 = self.up3_2(out_dec_level3)

        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        inp_dec_level1 = self.up2_1(out_dec_level2)

        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1) 
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refine(out_dec_level1)
        out_dec_level1 = self.up(out_dec_level1)

        out_dec_level1 = self.output_proj(out_dec_level1)

        return out_dec_level1 + x

    @torch.no_grad()
    def test(self, x):
        return self.forward(x)

if __name__== '__main__': 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 3, 3840, 2160).to(device)
    model = C2SSM().to(device)
    print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    with torch.no_grad():
        torch.cuda.reset_max_memory_allocated(device)
        start_time = time.time()
        output = model(x)
        end_time = time.time()
        memory_used = torch.cuda.max_memory_allocated(device)
    running_time = end_time - start_time
    print(output.shape)
    print(running_time)
    print(f"Memory used: {memory_used / 1024**3:.3f} GB")

    
