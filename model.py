import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.utils.checkpoint import checkpoint
import torch.special as S
from scipy.stats import lognorm
from fft_conv_pytorch import fft_conv
import matplotlib.pyplot as plt
import time

class JNetBlock0(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels  = in_channels ,
                              out_channels = out_channels,
                              kernel_size  = 7           ,
                              padding      = 'same'      ,
                              padding_mode = 'replicate' ,)
        
    def forward(self, x):
        x = self.conv(x)
        return x


class JNetBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout):
        super().__init__()
        self.bn1      = nn.BatchNorm3d(num_features = in_channels)
        self.relu1    = nn.ReLU(inplace=True)
        self.conv1    = nn.Conv3d(in_channels  = in_channels    ,
                                  out_channels = hidden_channels,
                                  kernel_size  = 3              ,
                                  padding      = 'same'         ,
                                  padding_mode = 'replicate'    ,)
        
        self.bn2      = nn.BatchNorm3d(num_features = hidden_channels)
        self.relu2    = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p = dropout)
        self.conv2    = nn.Conv3d(in_channels  = hidden_channels,
                                  out_channels = in_channels    ,
                                  kernel_size  = 3              ,
                                  padding      = 'same'         ,
                                  padding_mode = 'replicate'    ,)
        
    def forward(self, x):
        d = self.bn1(x)
        d = self.relu1(d)
        d = self.conv1(d)
        d = self.bn2(d)
        d = self.relu2(d)
        d = self.dropout1(d)
        d = self.conv2(d)
        x = x + d
        return x


class JNetBlockN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels  = in_channels ,
                              out_channels = out_channels,
                              kernel_size  = 3           ,
                              padding      = 'same'      ,
                              padding_mode = 'replicate' ,)
        self.sigm  = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.sigm(x)
        return x


class JNetPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool = nn.MaxPool3d(kernel_size = 2)
        self.conv    = nn.Conv3d(in_channels  = in_channels    ,
                                 out_channels = out_channels   ,
                                 kernel_size  = 1              ,
                                 padding      = 'same'         ,
                                 padding_mode = 'replicate'    ,)
        self.relu    = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv(x)
        x = self.relu(x)
        return x


class JNetUnpooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor = 2            ,
                                    mode         = 'trilinear'  ,)
        self.conv     = nn.Conv3d(in_channels    = in_channels  ,
                                  out_channels   = out_channels ,
                                  kernel_size    = 1            ,
                                  padding        = 'same'       ,
                                  padding_mode   = 'replicate'  ,)
        self.relu     = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.relu(x)
        return x


class JNetUpsample(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor = scale_factor ,
                                    mode         = 'trilinear'  ,)
                                    
    def forward(self, x):
        return self.upsample(x)


class SuperResolutionBlock(nn.Module):
    def __init__(self, scale_factor, in_channels, nblocks, dropout):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor = scale_factor ,
                                    mode         = 'trilinear'  ,)
        self.post     = nn.ModuleList([JNetBlock(in_channels     = in_channels , 
                                                 hidden_channels = in_channels ,
                                                 dropout         = dropout     ,
                                                 ) for _ in range(nblocks)])
    def forward(self, x):
        x = self.upsample(x)
        for f in self.post:
            x = f(x)
        return x


class CrossAttentionBlock(nn.Module):
    """
    ### Transformer Layer
    """

    def __init__(self, channels: int, n_heads: int, d_cond: int):
        """
        :param d_model: is the input embedding size
        :param n_heads: is the number of attention heads
        :param d_head: is the size of a attention head
        :param d_cond: is the size of the conditional embeddings
        """
        super().__init__()
        self.attn = CrossAttention(d_model = channels,
                                   d_cond  = d_cond,
                                   n_heads = n_heads,
                                   d_head  = channels // n_heads,)
        self.norm = nn.LayerNorm(normalized_shape = channels,)

    def forward(self, x: torch.Tensor):
        """
        :param x: are the input embeddings of shape `[batch_size, height * width, d_model]`
        :param cond: is the conditional embeddings of shape `[batch_size,  n_cond, d_cond]`
        """
        b, c, d, h, w = x.shape
        x = x.permute(0, 2, 3, 4, 1).view(b, d * h * w, c)
        x = self.attn(self.norm(x)) + x
        x = x.view(b, d, h, w, c).permute(0, 4, 1, 2, 3)
        return x


class CrossAttention(nn.Module):
    """
    ### Cross Attention Layer
    This falls-back to self-attention when conditional embeddings are not specified.
    """

    def __init__(self, d_model: int, d_cond: int, n_heads: int, d_head: int, is_inplace: bool = True):
        """
        :param d_model: is the input embedding size
        :param n_heads: is the number of attention heads
        :param d_head: is the size of a attention head
        :param d_cond: is the size of the conditional embeddings
        :param is_inplace: specifies whether to perform the attention softmax computation inplace to
            save memory
        """
        super().__init__()

        self.is_inplace = is_inplace
        self.n_heads    = n_heads
        self.d_head     = d_head

        # Attention scaling factor
        self.scale = d_head ** -0.5
        # Query, key and value mappings
        d_attn = d_head * n_heads
        self.to_q = nn.Linear(in_features  = d_model ,
                              out_features = d_attn  ,
                              bias         = False   ,)
        
        self.to_k = nn.Linear(in_features  = d_cond  ,
                              out_features = d_attn  ,
                              bias         = False   ,)
        
        self.to_v = nn.Linear(in_features  = d_cond  ,
                              out_features = d_attn  ,
                              bias         = False   ,)
        # Final linear layer
        self.to_out = nn.Sequential(
                                        nn.Linear(in_features  = d_attn ,
                                                  out_features = d_model,),
                                    )

    def forward(self, x: torch.Tensor, cond=None):
        """
        :param x: are the input embeddings of shape `[batch_size, height * width, d_model]`
        :param cond: is the conditional embeddings of shape `[batch_size, n_cond, d_cond]`
        """

        has_cond = cond is not None
        if not has_cond:
            cond = x
        q = self.to_q(x)
        k = self.to_k(cond)
        v = self.to_v(cond)

        return self.normal_attention(q, k, v)
    
    def normal_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        #### Normal Attention
        :param q: `[batch_size, seq, d_attn]`
        :param k: `[batch_size, seq, d_attn]`
        :param v: `[batch_size, seq, d_attn]`
        """

        q = q.view(*q.shape[:2], self.n_heads, -1)
        k = k.view(*k.shape[:2], self.n_heads, -1)
        v = v.view(*v.shape[:2], self.n_heads, -1)
        attn = torch.einsum('bihd,bjhd->bhij', q, k) * self.scale

        if self.is_inplace:
            half = attn.shape[0] // 2
            attn[half:] = attn[half:].softmax(dim=-1)
            attn[:half] = attn[:half].softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)

        out = torch.einsum('bhij,bjhd->bihd', attn, v)
        # Reshape to `[batch_size, height * width * depth, n_heads * d_head]`
        out = out.reshape(*out.shape[:2], -1)
        # Map to `[batch_size, height * width, d_model]` with a linear layer
        return self.to_out(out)


class VectorQuantizer(nn.Module):
    def __init__(self, threshold, device):
        super().__init__()
        self.t = threshold
        self.device = device
    def forward(self, x):
        x_quantized = (x >= self.t).to(self.device).float()
        x_quantized = x + (x_quantized - x).detach()
        quantize_loss = F.mse_loss(x_quantized.detach(), x)
        return x_quantized, quantize_loss


class JNetLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels_list,
                 attn_list, nblocks, dropout):
        super().__init__()
        is_attn = attn_list.pop(0)
        hidden_channels = hidden_channels_list.pop(0)
        self.hidden_channels = hidden_channels
        self.pool = JNetPooling(in_channels  = in_channels    ,
                                out_channels = hidden_channels,)
        self.conv = nn.Conv3d(in_channels    = hidden_channels,
                              out_channels   = hidden_channels,
                              kernel_size    = 1              ,
                              padding        = 'same'         ,
                              padding_mode   = 'replicate'    ,)
        self.prev = nn.ModuleList([JNetBlock(in_channels     = hidden_channels,
                                             hidden_channels = hidden_channels,
                                             dropout         = dropout        ,
                                             ) for _ in range(nblocks)])
        self.mid = JNetLayer(in_channels           = hidden_channels      ,
                             hidden_channels_list  = hidden_channels_list ,
                             attn_list             = attn_list            ,
                             nblocks               = nblocks              ,
                             dropout               = dropout              ,
                             ) if hidden_channels_list else nn.Identity()
        self.attn = CrossAttentionBlock(channels = hidden_channels ,
                                         n_heads  = 8               ,
                                         d_cond   = hidden_channels ,)\
                                             if is_attn else nn.Identity()
        self.post = nn.ModuleList([JNetBlock(in_channels     = hidden_channels,
                                             hidden_channels = hidden_channels,
                                             dropout         = dropout        ,
                                             ) for _ in range(nblocks)])
        self.unpool = JNetUnpooling(in_channels  = hidden_channels,
                                    out_channels = in_channels    ,)
    
    def forward(self, x):
        d = self.pool(x)
        d = self.conv(d) # checkpoint
        for f in self.prev:
            d = f(d)
        d = self.mid(d)
        d = self.attn(d)
        for f in self.post:
            d = f(d)
        d = self.unpool(d) # checkpoint
        x = x + d
        return x
    

class Emission(nn.Module):
    def __init__(self):
        super().__init__()

    def sample(self, x, params):
        b = x.shape[0]
        pz0  = dist.LogNormal(loc   = torch.tensor(float(params["mu_z"])).view( b,1,1,1,1).expand(*x.shape),
                              scale = torch.tensor(float(params["sig_z"])).view(b,1,1,1,1).expand(*x.shape),)
        x    = x * pz0.sample().to(x.device)
        x    = torch.clip(x, min=0., max=1.)
        return x


class NeuralImplicitPSF(nn.Module):
    def __init__(self, config, psf):
        super().__init__()
        mid = config["mid"]
        self.layers = nn.Sequential(
            nn.BatchNorm1d(2)  ,
            nn.Linear(2, mid)  ,
            nn.Sigmoid()       ,
            nn.BatchNorm1d(mid),
            nn.Linear(mid, 1)  ,
            nn.Sigmoid()
            )
        self.config = config
        self._gen_coord(psf)
        self._gen_label(psf)

    def _init_weights(self):
        for m in self.parameters():
            torch.nn.init.normal_(m)

    def forward(self, coords):
        return self.layers(coords)

    def trainer(self):
        self.to(self.config["device"])
        self._train()

    def _gen_coord(self, psf):
        self.psf_shape = psf.shape
        z, r = self.psf_shape
        zs = torch.linspace(-1, 1, steps=z).abs()
        rs = torch.linspace(-1, 1, steps=r)
        grid_z, grid_r = torch.meshgrid(zs, rs, indexing='ij')
        self.coord = torch.stack((
            grid_z.flatten(), grid_r.flatten()), -1).to(self.config["device"])

    def _gen_label(self, psf):
        self.label = psf.flatten()[:, None]

    def _train(self):
        label = self.label.to(self.config["device"])
        coord = self.coord.to(self.config["device"])
        loss_fn = eval(self.config["loss_fn"])
        loss = torch.tensor(1.)
        while loss.item() >= self.config["nipsf_loss_target"]:
            self._init_weights()
            count = 0
            label = label.to(self.config["device"])
            coord = coord.to(self.config["device"])
            optim = torch.optim.Rprop(self.parameters(), lr=self.config["lr"])
            averaged_model = torch.optim.swa_utils.AveragedModel(
                self,
                multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
            sched = torch.optim.lr_scheduler.ExponentialLR(
                optim,
                gamma=0.999,)
            while loss.item() >= self.config["nipsf_loss_target"]:
                optim.zero_grad()
                loss = loss_fn(self(coord), label)
                loss.backward()
                optim.step()
                averaged_model.update_parameters(self)
                sched.step()
                count += 1
                if count == self.config["num_iter_psf_pretrain"]:
                    print("Neural Implicit PSF is not good enough. " \
                          "Initializing NeuriPSF and trying again...")
                    break
        print(f"NeuriPSF train done with loss target"\
              f" {self.config['nipsf_loss_target']}.")

    def array(self):
        return self(self.coord).view(self.psf_shape)
        

class Blur(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.device      = params["device"]
        self.use_fftconv = params["use_fftconv"]
        if params["blur_mode"] == "gaussian":
            psf_model     = GaussianModel(params)
        elif params["blur_mode"] == "gibsonlanni":
            psf_model     = GibsonLanniModel(params)
        else:
            raise(NotImplementedError(
                f'blur_mode {params["blur_mode"]} is not implemented. Try "gaussian" or "gibsonlanni".'))
        self.init_psf_rz = torch.tensor(psf_model.PSF_rz, requires_grad=False).float().to(self.device)
        self.neuripsf = NeuralImplicitPSF(params, self.init_psf_rz)
        # self.neuripsf.trainer(self.init_psf_rz) <- moved to train_runner.py
        self.psf_rz_s0 = self.init_psf_rz.shape[0]
        self.size_z = params["size_z"]
        xy = torch.meshgrid(torch.arange(params["size_y"]),
             torch.arange(params["size_x"]),
             indexing='ij')
        r = torch.tensor(psf_model.r)
        x0 = (params["size_x"] - 1) / 2
        y0 = (params["size_y"] - 1) / 2
        r_pixel = torch.sqrt((xy[1] - x0) ** 2 + (xy[0] - y0) ** 2) * params["res_lateral"]
        rs0, = r.shape
        self.rps0, self.rps1 = r_pixel.shape
        r_e = r[:, None, None].expand(rs0, self.rps0, self.rps1)
        r_pixel_e = r_pixel[None].expand(rs0, self.rps0, self.rps1)
        r_index = torch.argmin(torch.abs(r_e- r_pixel_e), dim=0)
        r_index_fe = r_index.flatten().expand(self.psf_rz_s0, -1)
        self.r_index_fe = r_index_fe.to(self.device)
        self.z_pad   = int((params["size_z"] - params["res_axial"] // params["res_lateral"] + 1) // 2)
        self.x_pad   = (params["size_x"]) // 2
        self.y_pad   = (params["size_y"]) // 2
        self.stride  = (params["scale"], 1, 1)

    def forward(self, x):
        psf_rz    = self.neuripsf.array()
        l2_psf_rz = torch.mean((psf_rz - self.init_psf_rz) ** 2)
        psf = torch.gather(psf_rz, 1, self.r_index_fe)
        psf = psf.reshape(self.psf_rz_s0, self.rps0, self.rps1)
        psf = F.interpolate(
            input = psf[None, None, :]                  ,
            size  = (self.size_z, self.rps0, self.rps1) ,
            mode  = "nearest"                           ,
            )[0, 0]
        psf = psf / torch.sum(psf)#;print("sum: ", torch.sum(psf));print("max: ", torch.max(psf))#/ torch.sum(psf)
        if self.use_fftconv:
            _x   = fft_conv(signal  = x                                    ,
                            kernel  = psf                                  ,
                            stride  = self.stride                          ,
                            padding = (self.z_pad, self.x_pad, self.y_pad,),
                            )
        else:
            _x   = F.conv3d(input   = x                                    ,
                            weight  = psf                                  ,
                            stride  = self.stride                          ,
                            padding = (self.z_pad, self.x_pad, self.y_pad,),
                            )
        return {"out"     : _x       ,
                "psf_loss": l2_psf_rz}

    def show_psf_3d(self):
        # with torch.no_grad():
        psf_rz = self.neuripsf.array()
        psf = torch.gather(psf_rz, 1, self.r_index_fe)
        psf = psf / torch.sum(psf)
        psf = psf.reshape(self.psf_rz_s0, self.rps0, self.rps1)
        return psf


class GaussianModel():
    def __init__(self, params):
        oversampling = 1    # Defines the upsampling ratio on the image space grid for computations
        size_x = params["size_x"]
        size_y = params["size_y"]
        size_z = params["size_z"] // params["scale"]
        bet_xy = params["bet_xy"]
        bet_z = params["bet_z" ]
        x0 = (size_x - 1) / 2
        y0 = (size_y - 1) / 2
        z0 = (size_z - 1) / 2
        res_lateral = params["res_lateral"]#0.05  # microns # # # # param # # # #
        max_radius = round(np.sqrt((size_x - x0) * (size_x - x0) + (size_y - y0) * (size_y - y0)))
        self.r = res_lateral * np.arange(0, oversampling * max_radius) / oversampling
        xy = np.meshgrid(np.arange(size_z), np.arange(max_radius), indexing="ij")
        distance = np.sqrt((xy[1] / bet_xy) ** 2 + ((xy[0] - z0) / bet_z) ** 2)
        self.PSF_rz = np.exp(- distance ** 2)

    def __call__(self):
        return self.PSF_rz


class GibsonLanniModel():
    def __init__(self, params):
        size_x = params["size_x"]#256 # # # # param # # # #
        size_y = params["size_y"]#256 # # # # param # # # #
        size_z = params["size_z"] // params["scale"]#128 # # # # param # # # #

        # Precision control
        num_basis    = 100  # Number of rescaled Bessels that approximate the phase function
        num_samples  = 1000 # Number of pupil samples along radial direction
        oversampling = 1    # Defines the upsampling ratio on the image space grid for computations

        # Microscope parameters
        NA          = params["NA"]        #1.1   # # # # param # # # #
        wavelength  = params["wavelength"]#0.910 # microns # # # # param # # # #
        M           = params["M"]         #25    # magnification # # # # param # # # #
        ns          = params["ns"] #1.33  # specimen refractive index (RI)
        ng0         = params["ng0"]#1.5   # coverslip RI design value
        ng          = params["ng"] #1.5   # coverslip RI experimental value
        ni0         = params["ni0"]#1.5   # immersion medium RI design value
        ni          = params["ni"] #1.5   # immersion medium RI experimental value
        ti0         = params["ti0"]#150   # microns, working distance (immersion medium thickness) design value
        tg0         = params["tg0"]#170   # microns, coverslip thickness design value
        tg          = params["tg"] #170   # microns, coverslip thickness experimental value
        res_lateral = params["res_lateral"]#0.05  # microns # # # # param # # # #
        res_axial   = params["res_axial"]#0.5   # microns # # # # param # # # #
        pZ          = params["pZ"]       # 2 microns, particle distance from coverslip

        # Scaling factors for the Fourier-Bessel series expansion
        min_wavelength = 0.436 # microns
        scaling_factor = NA * (3 * np.arange(1, num_basis + 1) - 2) * min_wavelength / wavelength
        x0 = (size_x - 1) / 2
        y0 = (size_y - 1) / 2
        max_radius = round(np.sqrt((size_x - x0) * (size_x - x0) + (size_y - y0) * (size_y - y0)))
        r = res_lateral * np.arange(0, oversampling * max_radius) / oversampling
        self.r = r
        a = min([NA, ns, ni, ni0, ng, ng0]) / NA
        rho = np.linspace(0, a, num_samples)

        z = res_axial * np.arange(-size_z / 2, size_z /2) + res_axial / 2

        OPDs = pZ * np.sqrt(ns * ns - NA * NA * rho * rho) # OPD in the sample
        OPDi = (z.reshape(-1,1) + ti0) * np.sqrt(ni * ni - NA * NA * rho * rho) - ti0 * np.sqrt(ni0 * ni0 - NA * NA * rho * rho) # OPD in the immersion medium
        OPDg = tg * np.sqrt(ng * ng - NA * NA * rho * rho) - tg0 * np.sqrt(ng0 * ng0 - NA * NA * rho * rho) # OPD in the coverslip
        W    = 2 * np.pi / wavelength * (OPDs + OPDi + OPDg)
        phase = np.cos(W) + 1j * np.sin(W)
        J = scipy.special.jv(0, scaling_factor.reshape(-1, 1) * rho)
        C, residuals, _, _ = np.linalg.lstsq(J.T, phase.T, rcond=-1)
        b = 2 * np.pi * r.reshape(-1, 1) * NA / wavelength
        J0 = lambda x: scipy.special.j0(x)
        J1 = lambda x: scipy.special.j1(x)
        denom = scaling_factor * scaling_factor - b * b
        R = scaling_factor * J1(scaling_factor * a) * J0(b * a) * a - b * J0(scaling_factor * a) * J1(b * a) * a
        R /= denom
        PSF_rz = (np.abs(R.dot(C))**2).T
        self.PSF_rz = PSF_rz / np.max(PSF_rz)

    def __call__(self):
        return self.PSF_rz


class Noise(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.sig_eps = params["sig_eps"]
        self.a       = params["poisson_weight"]

    def forward(self, x):
        x = x + torch.randn_like(x) * (x * self.a + self.sig_eps)
        return x


class PreProcess(nn.Module):
    def __init__(self, min, max, params):
        super().__init__()
        self.min = min
        self.max = max
        self.background = torch.tensor(params["background"])
        self.gamma = dist.Gamma(torch.tensor([1.0]),
                                torch.tensor(params["background"]))
    
    def forward(self, x):
        x = torch.clip(x, min=self.min, max=self.max)
        #x = (x - self.min) / (self.max - self.min)
        return x
    
    def sample(self, x):
        x = (x - self.min) / (self.max - self.min)
#        x = x + self.background
#        max_value = torch.quantile(x.flatten(), self.max)
        x = torch.clip(x, min=self.min, max=self.max) #max_value.item())
        return x
    

class Hill(nn.Module):
    def __init__(self, n, ka, params):
        super().__init__()
        self.n_init  = n
        self.ka_init = ka
        self.n  = n
        self.k  = ka
        self.x1 = torch.ones(1).to(device=params["device"]) * 0.5
        self.x2 = torch.ones(1).to(device=params["device"]) * 0.8

    def forward(self, x):
        n  = F.sigmoid(self.n)
        ka = torch.clip(self.ka, min=0.)
        x = self.hill(x, n, ka)
        return x
    
    def hill_with_best_value(self, x):
        v = self.solve_hill(
            self.find_y(x, self.x1),
            self.find_y(x, self.x2),
            self.hill_ideal(self.x1),
            self.hill_ideal(self.x2),)
        x = self.hill(x, v['n'], v["ka"])
        return x
    
    def sample(self, x):
        x = self.hill(x, self.n_init, self.ka_init)
        return x

    def hill(self, x, n, ka):
        return (ka ** n + 1) * x ** n / (ka ** n + x ** n)
    
    def hill_ideal(self, x):
        return x ** 0.5 / (1 + x ** 0.5)
    
    def find_y(self, x, x1):
        return torch.quantile(x.flatten(), x1)
    
    def solve_hill(self, x1, x2, y1, y2):
        a = torch.log(x2) * (torch.log(1 - y1) - torch.log(y1))
        b = torch.log(x1) * (torch.log(1 - y2) - torch.log(y2))
        c = torch.log(y2) +  torch.log(1 - y1)
        d = torch.log(y1) +  torch.log(1 - y2)
        self.k = torch.exp((a - b) / (c - d))
        self.n = (torch.log(1 - y1) - torch.log(y1)) \
               / (torch.log(self.k) - torch.log(x1))
        return {"n"  : self.n,
                "ka" : self.k }
        
    def inverse_hill(self, x):
        x = x.clip(min = 0., max = 1.)
        return ((self.k - x + 1) / (self.k * x)) ** (- 1 / self.n)


class ImagingProcess(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.device     = params["device"]
        self.mu_z       = params["mu_z"]
        self.sig_z      = params["sig_z"]
        self.log_ez0    = nn.Parameter(
            (torch.tensor(params["mu_z"] + 0.5 \
                        * params["sig_z"] ** 2)).to(self.device),
            requires_grad=True)
        self.emission   = Emission()
        self.blur       = Blur(params = params)
        self.noise      = Noise(params)
        self.preprocess = PreProcess(min=0., max=1., params=params)
        self.hill       = Hill(n=0.5, ka=1., params=params)

    def forward(self, x):
        out = self.blur(x) 
        x = out["out"]
        x = self.preprocess(x)
#        x = self.hill.sample(x)
        out = {"out"      : x              ,
               "psf_loss" : out["psf_loss"] }
        return out


class JNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        t1 = time.time()
        print('initializing JNet model...')
        scale_factor         = (params["scale"], 1, 1)
        hidden_channels_list =  params["hidden_channels_list"].copy()
        attn_list            =  params["attn_list"].copy()
        hidden_channels      = hidden_channels_list.pop(0)
        attn_list.pop(0)
        self.prev0 = JNetBlock0(in_channels  = 1              ,
                                out_channels = hidden_channels,)
        self.prev  = nn.ModuleList(
            [JNetBlock(
                in_channels     = hidden_channels,
                hidden_channels = hidden_channels,
                dropout         = params["dropout"],
                ) for _ in range(params["nblocks"])
            ])
        self.mid   = JNetLayer(
            in_channels           = hidden_channels      ,
            hidden_channels_list  = hidden_channels_list ,
            attn_list             = attn_list            ,
            nblocks               = params["nblocks"]    ,
            dropout               = params["dropout"]    ,
            ) if hidden_channels_list else nn.Identity()
        self.postx = nn.ModuleList(
            [JNetBlock(in_channels     = hidden_channels,
                       hidden_channels = hidden_channels,
                       dropout         = params["dropout"],
                        ) for _ in range(params["nblocks"])
            ])
        self.postx.append(JNetBlockN(
            in_channels  = hidden_channels ,
            out_channels = 1               ,
            ))
        self.postz = nn.ModuleList(
            [JNetBlock(in_channels     = hidden_channels,
                       hidden_channels = hidden_channels,
                       dropout         = params["dropout"],
                        ) for _ in range(params["nblocks"])
            ])
        self.postz.append(JNetBlockN(
            in_channels  = hidden_channels ,
            out_channels = 1               ,
            ))
        self.image = ImagingProcess(params = params)
        self.upsample    = JNetUpsample(scale_factor = scale_factor)
        self.activation  = params["activation"]
        self.superres    = params["superres"]
        self.reconstruct = params["reconstruct"]
        self.apply_vq    = params["apply_vq"]
        self.vq = VectorQuantizer(threshold=params["threshold"],
                                  device=params["device"])
        t2 = time.time()
        print(f'JNet init done ({t2-t1:.2f} s)')
        self.use_x_quantized = params["use_x_quantized"]
        self.tau = 1.
        a = params["poisson_weight"]
        a_inv_sig = np.log(a / (1 - a))
        self.a     = nn.Parameter(torch.tensor(a_inv_sig)
                                  .to(device=params["device"]))
        s = params["sig_eps"]
        s_inv_sig = np.log(s / (1 - s))
        self.sigma = nn.Parameter(torch.tensor(s_inv_sig)
                                  .to(device=params["device"]))

    def forward(self, x):
        if self.superres:
            x = self.upsample(x)
        x = self.prev0(x)
        for f in self.prev:
            x = f(x)
        _x = self.mid(x)
        x = _x
        z = _x
        for f in self.postx:
            x = f(x)
        for f in self.postz:
            z = f(z)
        if self.apply_vq:
            if self.use_x_quantized:
                x, qloss = self.vq(x)
            else:
                _, qloss = self.vq(x)
        if self.reconstruct:
            lu  = x * z 
            out = self.image(lu)
            r   = out["out"]
            psf_loss = out["psf_loss"]
        else:
            r = x
        
        out = {"enhanced_image"  : x         ,
               "reconstruction"  : r         ,
               "mid"             : _x        ,
               "estim_luminance" : z         ,
               "poisson_weight"  : F.sigmoid(self.a)    ,
               "gaussian_sigma"  : F.sigmoid(self.sigma),
               }
        vqd = {"quantized_loss" : qloss} if self.apply_vq\
            else {"quantized_loss" : None}
        out = dict(**out, **vqd)
        psl = {"psf_loss": psf_loss} if self.reconstruct\
            else {"psf_loss": None}
        out = dict(**out, **psl)
        return out