import torch.nn as nn
import torch as th
import numpy as np
import math
import torch
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

from scipy import integrate
from scipy.stats import norm
from torch.nn.init import xavier_normal_, constant_, xavier_uniform_


class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias 


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, hidden_size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(hidden_size, hidden_size * 4)
        self.w_2 = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_1.weight)
        nn.init.constant_(self.w_1.bias, 0)
        nn.init.xavier_normal_(self.w_2.weight)
        nn.init.constant_(self.w_2.bias, 0)

    def forward(self, hidden):
        hidden = self.w_1(hidden)
        activation = 0.5 * hidden * (1 + torch.tanh(math.sqrt(2 / math.pi) * (hidden + 0.044715 * torch.pow(hidden, 3))))
        return self.w_2(self.dropout(activation))


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, hidden_size, dropout):
        super().__init__()
        assert hidden_size % heads == 0
        self.size_head = hidden_size // heads
        self.num_heads = heads
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.w_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_layer.weight)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        q, k, v = [l(x).view(batch_size, -1, self.num_heads, self.size_head).transpose(1, 2) for l, x in zip(self.linear_layers, (q, k, v))]
        corr = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        
        if mask is not None:
            mask = mask.unsqueeze(1).repeat([1, corr.shape[1], 1]).unsqueeze(-1).repeat([1,1,1,corr.shape[-1]])
            corr = corr.masked_fill(mask == 0, -1e9)
        prob_attn = F.softmax(corr, dim=-1)
        if self.dropout is not None:
            prob_attn = self.dropout(prob_attn)
        hidden = torch.matmul(prob_attn, v)
        hidden = self.w_layer(hidden.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.size_head))
        
        
        
        return hidden



class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, attn_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(heads=attn_heads, hidden_size=hidden_size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        self.input_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.output_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, hidden, c, mask):
        hidden = self.input_sublayer(hidden, lambda _hidden: self.attention.forward(_hidden, _hidden, _hidden, mask=mask))
        hidden = self.output_sublayer(hidden, self.feed_forward)
        return self.dropout(hidden)

class Transformer_rep(nn.Module):
    def __init__(self, args):
        super(Transformer_rep, self).__init__()
        self.hidden_size = args.hidden_size
        self.heads = 4
        self.dropout = args.dropout
        self.n_blocks = args.num_blocks
        self.last = args.last
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.hidden_size, self.heads, self.dropout) for _ in range(self.n_blocks)]
        )

    def forward(self, hidden, c, mask):
        i = 0
        for transformer in self.transformer_blocks:
            i += 1
            hidden = transformer.forward(hidden, c, mask)
            if i == (self.n_blocks - self.last):
                encode = hidden
        return hidden, encode

class FM_xstart(nn.Module):
    def __init__(self, hidden_size, args):
        super(FM_xstart, self).__init__()
        self.hidden_size = hidden_size
        self.linear_item = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_xt = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_t = nn.Linear(self.hidden_size, self.hidden_size)
        time_embed_dim = self.hidden_size * 4
        self.time_embed = nn.Sequential(nn.Linear(self.hidden_size, time_embed_dim), SiLU(), nn.Linear(time_embed_dim, self.hidden_size))
        self.fuse_linear = nn.Linear(self.hidden_size*3, self.hidden_size)
        self.att = Transformer_rep(args)

        self.lambda_uncertainty = args.lambda_uncertainty
        self.dropout = nn.Dropout(args.dropout)
        self.norm_fm_rep = LayerNorm(self.hidden_size)

        self.item_num = args.item_num
        self.out_dims = [512, 2048]
        self.act_func = 'tanh'

        out_dims_temp = [self.hidden_size] + self.out_dims + [self.item_num]
        decoder_modules = []
        for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:]):
            decoder_modules.append(nn.Linear(d_in, d_out))
            if self.act_func == 'relu':
                decoder_modules.append(nn.ReLU())
            elif self.act_func == 'sigmoid':
                decoder_modules.append(nn.Sigmoid())
            elif self.act_func == 'tanh':
                decoder_modules.append(nn.Tanh())
            elif self.act_func == 'leaky_relu':
                decoder_modules.append(nn.LeakyReLU())
            else:
                raise ValueError
        decoder_modules.pop()
        self.decoder = nn.Sequential(*decoder_modules)
        
        self.xavier_normal_initialization(self.decoder)


    def xavier_normal_initialization(self, module):
        r""" using `xavier_normal_`_ in PyTorch to initialize the parameters in
        nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
        using constant 0 to initialize.
        .. _`xavier_normal_`:
            https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_normal_#torch.nn.init.xavier_normal_
        Examples:
            >>> self.apply(xavier_normal_initialization)
        """
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)


    def timestep_embedding(self, timesteps, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = th.exp(-math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
        if dim % 2:
            embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

        
    def forward(self, rep_item, x_t, t, mask_seq):
        emb_t = self.time_embed(self.timestep_embedding(t, self.hidden_size))
        
        lambda_uncertainty = th.normal(mean=th.full(rep_item.shape, self.lambda_uncertainty), std=th.full(rep_item.shape, self.lambda_uncertainty)).to(x_t.device)

        rep_item_New = rep_item + (lambda_uncertainty * (x_t + emb_t).unsqueeze(1))

        condition_cross = rep_item

        rep_fm, encode = self.att(rep_item_New, condition_cross, mask_seq)

        rep_fm = self.norm_fm_rep(self.dropout(rep_fm))

        out = rep_fm[:, -1, :]

        encoded = encode[:, -1, :]

        decode = self.decoder(encoded)
        
        # out = out + self.lambda_uncertainty * x_t
        
        return out, decode

class FMRec(nn.Module):
    def __init__(self, args,):
        super(FMRec, self).__init__()
        self.hidden_size = args.hidden_size
        self.xstart_model = FM_xstart(self.hidden_size, args)
        self.eps = args.eps
        self.sample_N = args.sample_N
        self.eps_reverse = args.eps_reverse
        self.m_logNorm = args.m_logNorm
        self.s_logNorm = args.s_logNorm
        self.s_modsamp = args.s_modsamp
        self.sampling_method = args.sampling_method

    def from_flattened_numpy(self, x, shape):
        """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
        return torch.from_numpy(x.reshape(shape))
    
    def to_flattened_numpy(self, x):
        """Flatten a torch tensor `x` and convert it to numpy."""
        return x.detach().cpu().numpy().reshape((-1,))
    
    def rk45_sampler(self, item_rep, mask_seq, z0):

        with torch.no_grad():
            rtol=atol=tol=1e-5
            method='RK45'
            eps= self.eps 

            device = next(self.xstart_model.parameters()).device
            shape = item_rep[:,-1,:].shape

            x = z0.to(device)

            extra = (1 / self.eps) - 1

            def ode_func(t, x):
                
                x = self.from_flattened_numpy(x, shape).to(device).type(torch.float32)

                vec_t = torch.ones(shape[0], device=x.device) * t
                
                output, _ = self.xstart_model(item_rep, x, vec_t*extra, mask_seq)

                drift = output - z0

                return self.to_flattened_numpy(drift)
            # Black-box ODE solver for the probability flow ODE
            solution = integrate.solve_ivp(ode_func, (eps, self.T), self.to_flattened_numpy(x),
                                            rtol=rtol, atol=atol, method=method)
            
            nfe = solution.nfev

            x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)
            
            return x, nfe
        
    def euler_sampler(self, item_rep, mask_seq, z0):

        with torch.no_grad():

            device = next(self.xstart_model.parameters()).device
            shape = item_rep[:,-1,:].shape

            x = z0.to(device)

            ### Uniform
            dt = 1./self.sample_N
            eps = self.eps_reverse          # default: 1e-3

            extra = (1 / self.eps_reverse) - 1

            for i in range(self.sample_N):
                
                num_t = i /self.sample_N * (self.T - eps) + eps
                t = torch.ones(shape[0], device=device) * num_t
                pred, _  = self.xstart_model(item_rep, x, t*extra, mask_seq)

                # ## V_pred
                # V = pred
                
                ## Rectified Flow
                V = pred - z0       

                # #### Cosine
                # half_pi = math.pi / 2       
                # V = (half_pi * self.Sin_fn(t)).unsqueeze(1) * pred - ( (half_pi * self.Cos_fn(t)).unsqueeze(1) * z0 )

                x = x.detach().clone() + V * dt
            
            nfe = self.sample_N
        return x, nfe

    def reverse_p_sample_rf(self, item_rep, z0, mask_seq):

        # X_pred, nfe = self.rk45_sampler(item_rep, mask_seq, z0)

        X_pred, nfe = self.euler_sampler(item_rep, mask_seq, z0)

        # device = next(self.xstart_model.parameters()).device
        # shape = item_rep[:,-1,:].shape
        # t = torch.ones(shape[0], device=device)
        # X_pred , _= self.xstart_model(item_rep, z0, t , mask_seq)
        
        return X_pred 

    def Sin_fn(self, t):

        half_pi = math.pi / 2

        return torch.sin(half_pi * t)
    
    def Cos_fn(self, t):

        half_pi = math.pi / 2

        return torch.cos(half_pi * t)


    def a_t_fn(self, t):
        return t

    def b_t_fn(self, t):
        return 1 - t

    def a_t_derivative(self, t):
        return -torch.ones_like(t)

    def b_t_derivative(self, t):
        return torch.ones_like(t)
    
    @property
    def T(self):
      return 1.

        
    def q_sample_rf(self, x_start, t, z0, mask=None):

        assert z0.shape == x_start.shape

        # Rectified Flow
        a_t = self.a_t_fn(t)       ### a_t = t
        b_t = self.b_t_fn(t)       ### b_t = 1 - t
        
        # # Cosine
        # a_t = self.Cos_fn(t)
        # b_t = self.Sin_fn(t)

        x_t = a_t * x_start + b_t* z0

        if mask == None:
            return x_t
        
        else:
            mask = th.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape)  ## mask: [0,0,0,1,1,1,1,1]
            return th.where(mask==0, x_start, x_t)  ## replace the output_target_seq embedding (x_0) as x_t


    # Logit-Normal Sampling function for PyTorch
    def logit_normal_sampling_torch(self, m, s, batch_size):

        u_samples = torch.normal(mean=m, std=s, size=(batch_size,))

        t_samples = 1 / (1 + torch.exp(-u_samples))
        return t_samples
    

    def Mode_sample_timestep(self, batch_size, s, device):

        u = torch.rand(batch_size, device=device)
        
        correction_term = s * (torch.cos((torch.pi / 2) * u)**2 - 1 + u)
        
        t = 1 - u - correction_term
        
        return t

    def CosMap_sample_timesteps(self, batch_size, device):

        u = torch.rand(batch_size, device=device)
        
        t = 1 - 1 / (torch.tan((torch.pi / 2) * u) + 1)
        
        return t

    def forward(self, item_rep, item_tag, mask_seq):        
        noise = th.randn_like(item_tag)
        z0 = noise
        batch_size = item_tag.shape[0]

        # t, weights = self.schedule_sampler.sample(item_rep.shape[0], item_tag.device) ## t is sampled from schedule_sampler

        # Mode Sampling with Heavy Tails
        if self.sampling_method == 'mode':
            t_rf = self.Mode_sample_timestep(batch_size, self.s_modsamp, item_tag.device) * (self.T - self.eps) + self.eps
        
        # Uniform Sampling
        elif self.sampling_method == 'uniform':
            t_rf = torch.rand(item_tag.shape[0], device=item_tag.device) * (self.T - self.eps) + self.eps
        
        # Logit-Normal Sampling
        elif self.sampling_method == 'logit_normal':
            t_rf = self.logit_normal_sampling_torch(self.m_logNorm, self.s_logNorm, batch_size) * (self.T - self.eps) + self.eps
            t_rf = t_rf.to(item_tag.device)
        
        # CosMap_sample_timesteps
        elif self.sampling_method == 'cosmap':
            t_rf = self.CosMap_sample_timesteps(batch_size, item_tag.device) * (self.T - self.eps) + self.eps
        
        # Error
        else:
            raise ValueError(f"Unsupported sampling method: {self.sampling_method}")



        t_rf_expand = t_rf.view(-1, 1).repeat(1, item_tag.shape[1])
        
        # t = self.scale_t(t)
        # x_t = self.q_sample(item_tag, t, noise=noise)

        x_t = self.q_sample_rf(item_tag, t_rf_expand, z0=z0)
        
        # eps, item_rep_out = self.xstart_model(item_rep, x_t, self._scale_timesteps(t), mask_seq)  ## eps predict
        # x_0 = self._predict_xstart_from_eps(x_t, t, eps)

        extra = (1 / self.eps) - 1

        #####X0_pred
        x_0, decode_out = self.xstart_model(item_rep, x_t, t_rf*extra, mask_seq)  ##output predict
        
        return x_0, decode_out, t_rf_expand, t_rf, z0


        # #V_pred
        # v, decode_out = self.xstart_model(item_rep, x_t, t_rf*extra, mask_seq)  ##output predict

        # return v, decode_out, t_rf_expand, t_rf, z0
