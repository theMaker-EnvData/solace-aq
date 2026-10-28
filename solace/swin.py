"""Window-attention and downsampled global-attention blocks of the outer model."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _window_partition(x, ws):
    b, h, w, c = x.shape
    x = x.view(b, h // ws, ws, w // ws, ws, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, c), h // ws, w // ws


def _window_reverse(windows, ws, hb, wb, b):
    x = windows.view(b, hb, wb, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, hb * ws, wb * ws, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        b_, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        a = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            a = a + mask.unsqueeze(1)
        return self.proj((a.softmax(-1) @ v).transpose(1, 2).reshape(b_, n, c))


class SwinBlock(nn.Module):
    """Shifted-window self-attention at full resolution (no patch merging)."""

    def __init__(self, dim, window_size, num_heads, shift_size, mlp_ratio=4.0):
        super().__init__()
        self.ws, self.ss = window_size, shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))

    def _mask(self, h, w, device):
        if self.ss <= 0:
            return None
        ws, ss = self.ws, self.ss
        m = torch.zeros((1, h, w, 1), device=device)
        cnt = 0
        for hs in (slice(0, -ws), slice(-ws, -ss), slice(-ss, None)):
            for wsl in (slice(0, -ws), slice(-ws, -ss), slice(-ss, None)):
                m[:, hs, wsl, :] = cnt
                cnt += 1
        mw, _, _ = _window_partition(m, ws)
        mw = mw.view(-1, ws * ws)
        am = mw.unsqueeze(1) - mw.unsqueeze(2)
        return am.masked_fill(am != 0, -100.0).masked_fill(am == 0, 0.0)

    def forward(self, x):
        b, c, h, w = x.shape
        ws = self.ws
        ph, pw = (ws - h % ws) % ws, (ws - w % ws) % ws
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        hp, wp = x.shape[-2:]
        sc = x.permute(0, 2, 3, 1)
        xs = torch.roll(sc, (-self.ss, -self.ss), (1, 2)) if self.ss > 0 else sc
        mask = self._mask(hp, wp, x.device)
        if mask is not None:
            mask = mask.repeat(b, 1, 1)
        win, hb, wb = _window_partition(xs, ws)
        out = self.attn(self.norm1(win.view(-1, ws * ws, c)), mask).view(-1, ws, ws, c)
        xs = _window_reverse(out, ws, hb, wb, b)
        if self.ss > 0:
            xs = torch.roll(xs, (self.ss, self.ss), (1, 2))
        y = sc + xs
        y = y + self.mlp(self.norm2(y))
        y = y.permute(0, 3, 1, 2)
        return y[:, :, :h, :w] if (ph or pw) else y


class GlobalAttentionBlock(nn.Module):
    """Self-attention on an average-pooled copy, upsampled bilinearly and added back."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, downsample_factor=4):
        super().__init__()
        self.ds, self.num_heads = downsample_factor, num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x):
        b, c, h, w = x.shape
        hd, wd = max(1, h // self.ds), max(1, w // self.ds)
        s = F.adaptive_avg_pool2d(x, (hd, wd)).permute(0, 2, 3, 1).reshape(b, hd * wd, c)
        q, k, v = self.qkv(self.norm1(s)).reshape(b, hd * wd, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        o = self.proj(((q @ k.transpose(-2, -1)) * self.scale).softmax(-1).__matmul__(v).transpose(1, 2).reshape(b, hd * wd, c))
        o = F.interpolate(o.reshape(b, hd, wd, c).permute(0, 3, 1, 2), size=(h, w), mode="bilinear", align_corners=False)
        y = (x + o).permute(0, 2, 3, 1)
        y = y + self.mlp(self.norm2(y))
        return y.permute(0, 3, 1, 2)
