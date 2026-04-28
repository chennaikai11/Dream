from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv


def channel_shuffle_part(x: torch.Tensor, groups: int = 4) -> torch.Tensor:

    N, C, H, W = x.size()
    if groups <= 1:
        return x

    g_eff = math.gcd(C, groups)  # 取 gcd 保证能整除
    if g_eff <= 1:               # gcd=1，说明无法分组，直接整组 shuffle 相当于 Identity
        return x

    x = x.view(N, g_eff, C // g_eff, H, W).transpose(1, 2).contiguous()
    return x.view(N, C, H, W)


class SMConv(nn.Module):

    def __init__(self, channels: int, n_div: int = 2, kernel_size: int = 3, d=1):
        super().__init__()
        assert n_div >= 1
        self.n_div = int(n_div)
        self.kernel_size = int(kernel_size)
        self.d = int(d)
        self.channels = int(channels)

        self.c_conv = self.channels // self.n_div
        self.c_skip = self.channels - self.c_conv
        if self.c_conv > 0:
            self.conv3 = Conv(self.c_conv, self.c_conv, k=self.kernel_size, s=1, p=None, d=self.d)
        else:
            self.conv3 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.c_conv <= 0 or self.conv3 is None:
            return x

        x1, x2 = torch.split(x, [self.c_conv, self.c_skip], dim=1)
        y1 = self.conv3(x1)  # 保形：H×W 不变
        return torch.cat([y1, x2], dim=1)

class MDFC(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1,
                 n_div=2, d=1, shuffle_groups=4):
        super().__init__()
        # ==== 基本参数 ====
        assert out_dim % 2 == 0, "MDFC要求 out_dim 为偶数，以便两支各占一半通道。"
        assert n_div == 2, "按论文实现，n_div 一律取 2。"
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.k = int(kernel_size)
        self.n_div = int(n_div)      # 固定为2
        self.d = int(d)
        self.shuffle_groups = int(shuffle_groups)
        self.do_shuffle = (self.shuffle_groups > 1)

        # ==== 1) 低秩投影到一半通道（压缩+对齐） ====
        self.mid = self.out_dim // 2            # 两支通道数 = C_out / 2
        self.proj = Conv(self.in_dim, self.mid, k=1, s=stride, p=0)  # 1x1, 低秩投影+对齐

        # ==== 2) 两支在同一 z 上操作（避免 split 拷贝） ====
        # 2a) SMConv（半卷积+半直通）——Conv_Partial 内部依据 n_div=2 动态切分
        self.branch_partial = SMConv(channels=self.mid,
                                           n_div=self.n_div,
                                           kernel_size=kernel_size, d=self.d)

        self.branch_dw = Conv(self.mid, self.mid, k=kernel_size, s=1, p=None,
                              d=1, g=self.mid)  # groups=self.mid -> depthwise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 低秩投影 [N, C_in, H, W] -> [N, mid, H', W']
        z = self.proj(x)

        # 两支共享同一 z（避免多余内存与拷贝）
        y_p  = self.branch_partial(z)  # [N, mid, H', W']
        y_dw = self.branch_dw(z)       # [N, mid, H', W']

        out = torch.cat([y_p, y_dw], dim=1)  # [N, 2*mid(=out_dim), H', W']
        if self.do_shuffle:
            out = channel_shuffle_part(out, groups=self.shuffle_groups)
        return out

class Bottleneck_Pb(nn.Module):
    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1,
                 k: tuple[int, int] = (3, 3), e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels


        self.cv1 = MDFC(c1, c_, kernel_size=3)
        self.cv2 = MDFC(c_, c2, kernel_size=3)
        self.add = shortcut and (c1 == c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y

class C3_pb(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):

        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck_Pb(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3kpb(C3_pb):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):

        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_Pb(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C2f_Pb_old(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.n = int(n)
        self.c = int(c2 * e)

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.m = nn.ModuleList(
            Bottleneck_Pb(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(self.n)
        )
        self.cv2 = Conv((2 + self.n) * self.c, c2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2Pb(C2f_Pb_old):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, g: int = 1, shortcut: bool = True
    ):

        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3kpb(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_Pb(self.c, self.c, shortcut, g) for _ in range(n)
        )


class DRA(nn.Module):
    """
    ECA 通道打分 + Beta 残差门控（公式保持不变）:
        gate = 1 + beta * (a - 1)

    逐通道模式下，仅对 beta 做可学习重参数化（不改变公式）：
        beta_i = s * C^{1/2} * tanh(rho_i / τ)
    标量模式：保持原始的 beta 计算不变。

    统计导出：
        pop_epoch_stats(): {gate, beta_raw}
    额外缓存（便于脚本侧轻正则/日志，不影响前向）：
        last_gate_mean, last_gate_var, last_frac_gt1
    """
    def __init__(self, inp: int,
                 kernel_size: int = None,
                 print_mode: str = "epoch",
                 verbose: bool = True,
                 # ECA 自适应核
                 gamma: float = 2.0, b: float = 1.0,
                 # Beta-Gate
                 per_channel_beta: bool = True,
                 use_reparam: bool = True,
                 beta_init: float = 0.0,
                 gate_clip=None, eps: float = 1e-5):
        super().__init__()
        self.inp = int(inp)
        self.print_mode = print_mode
        self.verbose = bool(verbose)
        self.per_channel_beta = bool(per_channel_beta)
        self.gate_clip = gate_clip
        self.eps = float(eps)
        self.use_reparam = use_reparam

        # --- 自适应核大小 ---
        if kernel_size is None:
            self.kernel_size = None
            self.gamma = float(gamma)
            self.b = float(b)
        else:
            k = int(kernel_size)
            if k % 2 == 0: k += 1
            self.kernel_size = max(1, k)
            self.gamma = None
            self.b = None

        # ECA conv1d
        k_eff = self._get_ksize(self.inp)
        padding = (k_eff - 1) // 2
        self.eca_conv = nn.Conv1d(1, 1, kernel_size=k_eff, padding=padding, bias=False)

        if self.per_channel_beta:
            if self.use_reparam:
                self.rho = nn.Parameter(torch.zeros(self.inp))
                with torch.no_grad():
                    self.rho.add_(1e-2 * torch.randn_like(self.rho))  # 极小扰动打破对称

                self.s_raw = nn.Parameter(torch.tensor(1.0)) # s 1.0
                self.register_buffer("tau", torch.tensor(1.0))  # τ 可调：1.0 / √2 / 2.0
            else:
                self.beta_direct = nn.Parameter(torch.full((self.inp,), float(beta_init)))
        else:
            # 标量模式
            self.beta = nn.Parameter(torch.tensor(float(beta_init)))

        # 统计缓存
        self.register_buffer("gate_sum",   torch.tensor(0.0))
        self.register_buffer("gate_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("last_gate_mean", torch.tensor(float('nan')))
        self.register_buffer("last_gate_var",  torch.tensor(float('nan')))
        self.register_buffer("last_frac_gt1",  torch.tensor(float('nan')))

    def _get_ksize(self, C: int):
        if self.kernel_size is not None:
            return self.kernel_size
        k = int(abs((math.log2(max(1, C)) / self.gamma) + self.b))
        if k % 2 == 0: k += 1
        return max(1, k)

    def _beta_effective(self, C: int) -> torch.Tensor:

        if self.use_reparam:
            s = self.s_raw.abs()
            z = self.rho / 1.0
            phi = torch.tanh(z)
            beta_vec = s * math.sqrt(C) * phi  # (C,)
        else:
            beta_vec = self.beta_direct  # (C,)
        return beta_vec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        assert C == self.inp, f"Expected in channels {self.inp} but got {C}"

        k_eff = self._get_ksize(C)
        if k_eff != self.eca_conv.kernel_size[0]:
            pad = (k_eff - 1) // 2
            new_conv = nn.Conv1d(1, 1, kernel_size=k_eff, padding=pad, bias=False).to(self.eca_conv.weight.device)
            self.eca_conv = new_conv

        # ECA 打分 a ∈ (0,1)
        y = F.adaptive_avg_pool2d(x, 1).view(N, C)       # (N,C)
        y = self.eca_conv(y.unsqueeze(1)).squeeze(1)     # (N,C)
        a = torch.sigmoid(y).unsqueeze(-1).unsqueeze(-1) # (N,C,1,1)

        # ---- gate：严格保持 gate = 1 + beta * (a - 1) ----
        if self.per_channel_beta:
            beta_vec = self._beta_effective(C)                   # (C,)
            beta_e   = beta_vec.view(1, C, 1, 1)                 # (1,C,1,1)
            gate     = 1.0 + beta_e * (a - 1.0)                 # 公式不变
        else:
            beta_e = self.beta * torch.ones((1, C, 1, 1), device=x.device, dtype=x.dtype)
            gate   = 1.0 + beta_e * (a - 1.0)                   # 公式不变

        if self.gate_clip is not None:
            lo, hi = self.gate_clip
            gate = torch.clamp(gate, lo, hi)

        # 统计缓存
        with torch.no_grad():
            m = gate.mean().detach()
            v = gate.var(unbiased=False).detach()
            frac = (gate > 1).to(dtype=torch.float32).mean().detach()
            self.last_gate_mean.copy_(m)
            self.last_gate_var.copy_(v)
            self.last_frac_gt1.copy_(frac)
            self.gate_sum += m
            self.gate_count += 1

        return x * gate

    @torch.no_grad()
    def pop_epoch_stats(self) -> dict:
        if int(self.gate_count.item()) > 0:
            gate_avg = (self.gate_sum / float(self.gate_count.item())).item()
        else:
            gate_avg = float('nan')

        if self.per_channel_beta:
            if self.use_reparam:
                r = self.rho.detach().float()
                beta_raw = float(r.mean().item())
            else:
                b = self.beta_direct.detach().float()
                beta_raw = float(b.mean().item())
        else:
            beta_raw = float(self.beta.detach().item())

        self.gate_sum.zero_()
        self.gate_count.zero_()
        return {"gate": gate_avg, "beta_raw": beta_raw}


class C2f_PA_old(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.n = int(n)
        self.c = int(c2 * e)

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.m = nn.ModuleList(
            Bottleneck_Pb(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(self.n)
        )
        self.cv2 = Conv((2 + self.n) * self.c, c2, 1)

        self.attention = DRA(c2, None)
        # self.attention = ECA(c2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.attention(self.cv2(torch.cat(y, 1)))
        # return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.attention(self.cv2(torch.cat(y, 1)))

class C3k2PA(C2f_PA_old):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, g: int = 1, shortcut: bool = True
    ):

        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3kpb(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_Pb(self.c, self.c, shortcut, g) for _ in range(n)
        )

