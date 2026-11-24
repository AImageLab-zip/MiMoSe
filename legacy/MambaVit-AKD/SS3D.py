import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
import math
# from hilbert import decode, encode
from pyzorder import ZOrderIndexer


class HSCANS(nn.Module):
    def __init__(self, size=24, dim=3, scan_type='scan', ):
        super().__init__()
        size = int(size)
        max_num = size ** dim
        indexes = np.arange(max_num)
        self.dim = dim
        if 'sweep' == scan_type:  # ['sweep', 'scan', 'zorder', 'zigzag', 'hilbert']
            locs_flat = indexes
        elif 'scan' == scan_type:
            if dim == 2:
                indexes = indexes.reshape(size, size)
                for i in np.arange(1, size, step=2):
                    indexes[i, :] = indexes[i, :][::-1]
                locs_flat = indexes.reshape(-1)
            elif dim == 3:
                indexes = indexes.reshape(size, size, size)
                for i in np.arange(1, size, step=2):
                    indexes[:, i, :] = np.flip(indexes[:, i, :], axis=1)  # flipping y
                for j in np.arange(1, size, step=2):
                    indexes[j, :, :] = np.flip(indexes[j, :, :], axis=(0, 1))
                locs_flat = indexes.reshape(-1)
        elif 'zorder' == scan_type:
            zi = ZOrderIndexer((0, size - 1), (0, size - 1))
            locs_flat = []
            for z in indexes:
                r, c = zi.rc(int(z))
                locs_flat.append(c * size + r)
            locs_flat = np.array(locs_flat)
        elif 'zigzag' == scan_type:
            indexes = indexes.reshape(size, size)
            locs_flat = []
            for i in range(2 * size - 1):
                if i % 2 == 0:
                    start_col = max(0, i - size + 1)
                    end_col = min(i, size - 1)
                    for j in range(start_col, end_col + 1):
                        locs_flat.append(indexes[i - j, j])
                else:
                    start_row = max(0, i - size + 1)
                    end_row = min(i, size - 1)
                    for j in range(start_row, end_row + 1):
                        locs_flat.append(indexes[j, i - j])
            locs_flat = np.array(locs_flat)
        elif 'hilbert' == scan_type:
            bit = int(math.log2(size))
            locs = decode(indexes, dim, bit)
            locs_flat = self.flat_locs_hilbert(locs, dim, bit)
        else:
            raise Exception('invalid encoder mode')
        locs_flat_inv = np.argsort(locs_flat)
        index_flat = torch.LongTensor(locs_flat.astype(np.int64)).unsqueeze(0).unsqueeze(1)
        index_flat_inv = torch.LongTensor(locs_flat_inv.astype(np.int64)).unsqueeze(0).unsqueeze(1)
        self.index_flat = nn.Parameter(index_flat, requires_grad=False)
        self.index_flat_inv = nn.Parameter(index_flat_inv, requires_grad=False)

    def flat_locs_hilbert(self, locs, num_dim, num_bit):
        ret = []
        l = 2 ** num_bit
        for i in range(len(locs)):
            loc = locs[i]
            loc_flat = 0
            for j in range(num_dim):
                loc_flat += loc[j] * (l ** j)
            ret.append(loc_flat)
        return np.array(ret).astype(np.uint64)

    def __call__(self, img):
        img_encode = self.encode(img)
        return img_encode

    def encode(self, img):
        img_encode = torch.zeros(img.shape, dtype=img.dtype, device=img.device).scatter_(2, self.index_flat_inv.expand(
            img.shape), img)
        return img_encode

    def decode(self, img):
        img_decode = torch.zeros(img.shape, dtype=img.dtype, device=img.device).scatter_(2, self.index_flat.expand(
            img.shape), img)
        return img_decode


class SS3D(nn.Module):  # for the original Vanilla VSS block, worse as described in VMamba paper
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=1,  # 2
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device='cuda',
            dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model  # channel dim, 524 or 1024, gets expanded
        self.d_state = d_state

        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv3d = nn.Conv3d(
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
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj],
                                                      dim=0))  # (K=8, N, inner) = (K=8, new_c = self.dt_rank + self.d_state * 2, C)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=8, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=8, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

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
    def A_log_init(d_state, d_inner, copies=8, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        # ('A', A.shape)
        A_log = torch.log(A)  # Keep A_log in fp32

        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=8, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        # 0,1, 2, 3, 4
        B, C, H, W, D = x.shape
        L = H * W * D
        K = 8

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L),
                              torch.transpose(x, dim0=2, dim1=4).contiguous().view(B, -1, L),
                              torch.transpose(x, dim0=3, dim1=4).contiguous().view(B, -1, L)], dim=1).view(B, 4, -1, L)
        # hwd, whd, dwh, hdw; reversed
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, c, l)
        # hwd b, 1, c, l >
        # whd b, 1, c, l >
        # dwh b, 1, c, l >
        # hdw b, 1, c, l >
        # hwd reversed l
        # whd reversed l
        # dwh reversed l
        # hdw reversed l

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)

        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)

        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)

        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)

        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)

        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        # hwd b, 1, c, l >
        # whd b, 1, c, l >
        # dwh b, 1, c, l >
        # hdw b, 1, c, l >
        # hwd reversed l
        # whd reversed l
        # dwh reversed l
        # hdw reversed l

        # revert back to all hwd forward l

        # out1 = out_y[:,0,:,:]
        out2 = torch.transpose(out_y[:, 1].view(B, -1, W, H, D), dim0=2, dim1=3).contiguous().view(B, -1, L)
        out3 = torch.transpose(out_y[:, 2].view(B, -1, W, H, D), dim0=2, dim1=4).contiguous().view(B, -1, L)
        out4 = torch.transpose(out_y[:, 3].view(B, -1, W, H, D), dim0=3, dim1=4).contiguous().view(B, -1, L)

        out5 = torch.flip(out_y[:, 0], dims=[-1]).view(B, -1, L)
        out6 = torch.flip(out2, dims=[-1]).view(B, -1, L)
        out7 = torch.flip(out3, dims=[-1]).view(B, -1, L)
        out8 = torch.flip(out4, dims=[-1]).view(B, -1, L)

        return out_y[:, 0], out2, out3, out4, out5, out6, out7, out8

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, D, C = x.shape  # !!!
        # d_model = C

        xz = self.in_proj(x)  # (b, h, w, d, d_model) -> (b, h, w, d, d_inner * 2)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d, d_inner), z for the multiplicative path

        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.act(self.conv3d(x))  # (b, d, h, w)

        y1, y2, y3, y4, y5, y6, y7, y8 = self.forward_core(x)  # 1 1024 1728

        assert y1.dtype == torch.float32

        y = y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8

        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, D, -1)  # bcl > blc > bhwdc
        y = self.out_norm(y)
        y = y * F.silu(z)  # multiplicative path, ignored in v2 because ssm is inherently selective, described in VMamba

        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)

        return out


class SS3D_v5(nn.Module):  # no multiplicative path, the better version described in VMamba
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=1,  # 2
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device='cuda',
            dtype=None,
            einsum=True,
            size=24,
            scan_type='scan',  # size needs to be a power of 2 to use hilbert
            num_direction=8,
            orientation=0,  # 0, 1, 2
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.orientation = orientation
        self.d_model = d_model  # channel dim, 524 or 1024, gets expanded
        self.d_state = d_state

        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.conv3d = nn.Conv3d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        if einsum:
            self.x_proj = (
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs) for i in
            range(num_direction)
            )
            self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj],
                                                          dim=0))  # (K=8, N, inner) = (K=8, new_c = self.dt_rank + self.d_state * 2, C)
            del self.x_proj
        else:
            # print('no einsum for x_proj')
            raise Exception('have to use einsum for now lol')
        # figure out how to do dts without einsum
        self.dt_projs = [
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for i in range(num_direction)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=num_direction, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=num_direction, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        self.scans = HSCANS(size=size, scan_type=scan_type)

        # self.scans.encode = lambda x: x
        # self.scans.decode = lambda x: x

        self.num_direction = num_direction

        if (orientation % 3) == 0:
            self.transp = lambda x: x
        elif (orientation % 3) == 1:
            self.transp = lambda x: torch.transpose(x, dim0=2, dim1=3)  # change to 3 4 if hilbert
        elif (orientation % 3) == 2:
            self.transp = lambda x: torch.transpose(x, dim0=2, dim1=4)  # scan goes across first dim
        self.transp2 = lambda x: x
        if (orientation % 6) > 2:  # 3, 4, 5
            self.transp2 = lambda x: torch.transpose(x, dim0=3, dim1=4)

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
    def A_log_init(d_state, d_inner, copies=8, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        # ('A', A.shape)
        A_log = torch.log(A)  # Keep A_log in fp32

        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=8, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        # 0,1, 2, 3, 4
        B, C, H, W, D = x.shape
        L = H * W * D
        K = self.num_direction
        xs = []

        xs.append(self.scans.encode(self.transp2(self.transp(x)).contiguous().view(B, -1, L)))

        xs.append(self.scans.encode(self.transp2(
            self.transp(torch.rot90(torch.rot90(x, k=1, dims=(3, 4)), k=1, dims=(2, 4)))).contiguous().view(B, -1, L)))
        xs.append(
            self.scans.encode(self.transp2(self.transp(torch.rot90(x, k=2, dims=(2, 4)))).contiguous().view(B, -1, L)))
        xs.append(self.scans.encode(self.transp2(
            self.transp(torch.rot90(torch.rot90(x, k=-1, dims=(2, 4)), k=1, dims=(2, 3)))).contiguous().view(B, -1, L)))

        # xs.append(self.scans.encode(self.transp2(self.transp(torch.rot90(x, k=2, dims=(2,3)))).contiguous().view(B, -1, L)))
        # xs.append(self.scans.encode(self.transp2(self.transp(torch.rot90(x, k=2, dims=(2,4)))).contiguous().view(B, -1, L)))
        # xs.append(self.scans.encode(self.transp2(self.transp(torch.rot90(x, k=2, dims=(3,4)))).contiguous().view(B, -1, L)))

        xs = torch.stack(xs, dim=1).view(B, K // 2, -1, L)
        xs = torch.cat([xs, torch.flip(xs, dims=[-1])], dim=1)  # (b, k, c, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)

        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)

        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        assert out_y.dtype == torch.float

        # out_y = xs.view(B, K, -1, L) # for testing

        inv_y = torch.flip(out_y[:, K // 2:K], dims=[-1]).view(B, K // 2, -1, L)
        ys = []

        # xs.append(self.scans.encode(self.transp2(self.transp(x)).contiguous().view(B, -1, L)))
        ys.append(
            self.transp(self.transp2(self.scans.decode(out_y[:, 0]).view(B, -1, H, W, D))).contiguous().view(B, -1, L))
        ys.append(
            self.transp(self.transp2(self.scans.decode(inv_y[:, 0]).view(B, -1, H, W, D))).contiguous().view(B, -1, L))

        # xs.append(self.scans.encode(self.transp2(self.transp(torch.rot90(torch.rot90(x, k=1, dims=(3,4)), k=1, dims=(2,4)))).contiguous().view(B, -1, L)))
        ys.append(torch.rot90(
            torch.rot90(self.transp(self.transp2(self.scans.decode(out_y[:, 1]).view(B, -1, H, W, D))), k=-1,
                        dims=(2, 4)), k=-1, dims=(3, 4)).contiguous().view(B, -1, L))
        ys.append(torch.rot90(
            torch.rot90(self.transp(self.transp2(self.scans.decode(inv_y[:, 1]).view(B, -1, H, W, D))), k=-1,
                        dims=(2, 4)), k=-1, dims=(3, 4)).contiguous().view(B, -1, L))

        # xs.append(self.scans.encode(self.transp2(self.transp(torch.rot90(x, k=2, dims=(2,4)))).contiguous().view(B, -1, L)))
        ys.append(torch.rot90(self.transp(self.transp2(self.scans.decode(out_y[:, 2]).view(B, -1, H, W, D))), k=2,
                              dims=(2, 4)).contiguous().view(B, -1, L))
        ys.append(torch.rot90(self.transp(self.transp2(self.scans.decode(inv_y[:, 2]).view(B, -1, H, W, D))), k=2,
                              dims=(2, 4)).contiguous().view(B, -1, L))

        # xs.append(self.scans.encode(self.transp2(self.transp(torch.rot90(torch.rot90(x, k=3, dims=(2,4)), k=1, dims=(2,3)))).contiguous().view(B, -1, L)))
        ys.append(torch.rot90(
            torch.rot90(self.transp(self.transp2(self.scans.decode(out_y[:, 3]).view(B, -1, H, W, D))), k=-1,
                        dims=(2, 3)), k=1, dims=(2, 4)).contiguous().view(B, -1, L))
        ys.append(torch.rot90(
            torch.rot90(self.transp(self.transp2(self.scans.decode(inv_y[:, 3]).view(B, -1, H, W, D))), k=-1,
                        dims=(2, 3)), k=1, dims=(2, 4)).contiguous().view(B, -1, L))

        # ys.append(torch.rot90(self.transp(self.transp2(self.scans.decode(out_y[:, 1]).view(B, -1, H, W, D))), k=2, dims=(2,3)).contiguous().view(B, -1, L))
        # ys.append(torch.rot90(self.transp(self.transp2(self.scans.decode(inv_y[:, 1]).view(B, -1, H, W, D))), k=2, dims=(2,3)).contiguous().view(B, -1, L))

        # ys.append(torch.rot90(self.transp(self.transp2(self.scans.decode(out_y[:, 2]).view(B, -1, H, W, D))), k=2, dims=(2,4)).contiguous().view(B, -1, L))
        # ys.append(torch.rot90(self.transp(self.transp2(self.scans.decode(inv_y[:, 2]).view(B, -1, H, W, D))), k=2, dims=(2,4)).contiguous().view(B, -1, L))

        # ys.append(torch.rot90(self.transp(self.transp2(self.scans.decode(out_y[:, 3]).view(B, -1, H, W, D))), k=2, dims=(3,4)).contiguous().view(B, -1, L))
        # ys.append(torch.rot90(self.transp(self.transp2(self.scans.decode(inv_y[:, 3]).view(B, -1, H, W, D))), k=2, dims=(3,4)).contiguous().view(B, -1, L))

        # for y in ys:
        #     print(torch.all(y==x.view(B, -1, L)))
        return sum(ys)

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, D, C = x.shape  # !!!

        x = self.in_proj(x)

        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.act(self.conv3d(x))  # (b, d, h, w)
        y = self.forward_core(x)  # 1 1024 1728

        assert y.dtype == torch.float32

        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, D, -1)  # bcl > blc > bhwdc

        y = self.out_norm(y)

        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)

        return out


class SS3D_v6(nn.Module):  # no multiplicative path, the better version described in VMamba
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=1,  # 2
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device='cuda',
            dtype=None,
            einsum=True,
            size=24,
            scan_type='scan',  # size needs to be a power of 2 to use hilbert
            num_direction=6,
            orientation=0,  # 0, 1, 2, 4, 5, 6, 7
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.orientation = orientation
        self.d_model = d_model  # channel dim, 524 or 1024, gets expanded
        self.d_state = d_state

        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.conv3d = nn.Conv3d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        if einsum:
            self.x_proj = (
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs) for i in
            range(num_direction)
            )
            self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj],
                                                          dim=0))  # (K=8, N, inner) = (K=8, new_c = self.dt_rank + self.d_state * 2, C)
            del self.x_proj
        else:
            # print('no einsum for x_proj')
            raise Exception('have to use einsum for now lol')
        # figure out how to do dts without einsum
        self.dt_projs = [
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for i in range(num_direction)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=num_direction, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=num_direction, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        # self.scans = HSCANS(size=size, scan_type=scan_type)

        # self.scans.encode = lambda x: x
        # self.scans.decode = lambda x: x

        self.num_direction = num_direction

        if (orientation % 8) == 0:
            self.rot = lambda x: x
            self.unrot = lambda x: x
        elif (orientation % 8) == 1:
            self.rot = lambda x: torch.rot90(x, 1, (2, 3))
            self.unrot = lambda x: torch.rot90(x, -1, (2, 3))
        elif (orientation % 8) == 2:
            self.rot = lambda x: torch.rot90(x, 1, (3, 4))
            self.unrot = lambda x: torch.rot90(x, -1, (3, 4))
        elif (orientation % 8) == 3:
            self.rot = lambda x: torch.rot90(x, -1, (2, 4))
            self.unrot = lambda x: torch.rot90(x, 1, (2, 4))
        elif (orientation % 8) == 4:
            self.rot = lambda x: torch.transpose(
                torch.transpose(torch.rot90(torch.rot90(x, 2, (2, 3)), 1, (2, 4)), 2, 4), 2, 3)
            self.unrot = lambda x: torch.rot90(torch.rot90(torch.transpose(torch.transpose(x, 3, 4), 2, 3), -1, (2, 4)),
                                               2, (2, 3))
        elif (orientation % 8) == 5:
            self.rot = lambda x: torch.rot90(x, 2, (2, 4))
            self.unrot = lambda x: torch.rot90(x, 2, (2, 4))
        elif (orientation % 8) == 6:
            self.rot = lambda x: torch.transpose(torch.transpose(torch.rot90(x, 2, (2, 3)), 3, 4), 2, 3)
            self.unrot = lambda x: torch.rot90(torch.transpose(torch.transpose(x, 2, 3), 3, 4), 2, (2, 3))
        elif (orientation % 8) == 7:
            self.rot = lambda x: torch.transpose(torch.transpose(torch.rot90(x, 2, (3, 4)), 2, 3), 3, 4)
            self.unrot = lambda x: torch.rot90(torch.transpose(torch.transpose(x, 3, 4), 2, 3), 2, (3, 4))

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
    def A_log_init(d_state, d_inner, copies=8, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        # ('A', A.shape)
        A_log = torch.log(A)  # Keep A_log in fp32

        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=8, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        # 0,1, 2, 3, 4
        B, C, H, W, D = x.shape
        L = H * W * D
        K = self.num_direction
        xs = []

        xs.append(self.rot(x).contiguous().view(B, -1, L))
        xs.append(torch.transpose(self.rot(x), 2, 4).contiguous().view(B, -1, L))
        xs.append(torch.transpose(self.rot(x), 3, 4).contiguous().view(B, -1, L))

        xs = torch.stack(xs, dim=1).view(B, K // 2, -1, L)
        xs = torch.cat([xs, torch.flip(xs, dims=[-1])], dim=1)  # (b, k, c, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)

        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)

        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        assert out_y.dtype == torch.float

        # out_y = xs.view(B, K, -1, L) # for testing

        inv_y = torch.flip(out_y[:, K // 2:K], dims=[-1]).view(B, K // 2, -1, L)
        ys = []

        # xs.append(self.rot(x).contiguous().view(B, -1, L))
        # xs.append(torch.transpose(self.rot(x), 2, 4).contiguous().view(B, -1, L)))
        # xs.append(torch.transpose(self.rot(x), 3, 4).contiguous().view(B, -1, L)))

        ys.append(self.unrot(out_y[:, 0].view(B, -1, H, W, D)).contiguous().view(B, -1, L))
        ys.append(self.unrot(inv_y[:, 0].view(B, -1, H, W, D)).contiguous().view(B, -1, L))

        ys.append(self.unrot(torch.transpose(out_y[:, 1].view(B, -1, H, W, D), 2, 4)).contiguous().view(B, -1, L))
        ys.append(self.unrot(torch.transpose(inv_y[:, 1].view(B, -1, H, W, D), 2, 4)).contiguous().view(B, -1, L))

        ys.append(self.unrot(torch.transpose(out_y[:, 2].view(B, -1, H, W, D), 3, 4)).contiguous().view(B, -1, L))
        ys.append(self.unrot(torch.transpose(inv_y[:, 2].view(B, -1, H, W, D), 3, 4)).contiguous().view(B, -1, L))

        # for y in ys:
        #     print(torch.all(y==x.view(B, -1, L)))
        return sum(ys)

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, D, C = x.shape  # !!!

        x = self.in_proj(x)

        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.act(self.conv3d(x))  # (b, d, h, w)
        y = self.forward_core(x)  # 1 1024 1728

        assert y.dtype == torch.float32

        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, D, -1)  # bcl > blc > bhwdc

        y = self.out_norm(y)

        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)

        return out