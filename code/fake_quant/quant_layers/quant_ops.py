import torch, torch.nn as nn, torch.nn.functional as F, math, fast_hadamard_transform
import hadamard_utils
from einops import rearrange
from fast_hadamard_transform import hadamard_transform
from .quantizer import Quantizer


class H(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_hadamard_transform = False
        self.online_full_had = False 
        self.online_partial_had = False 
        self.had_K = None
        self.K = 1
        self.down_dim = 1
    
    def free_temporary(self):
        if hasattr(self,"use_temporary_parameter"):
            self.use_temporary_parameter = False
        if hasattr(self,"temp_weight"):
            self.temp_weight = None
        if hasattr(self,"temp_bias"):
            self.temp_bias = None


    def may_hadamard_transform(self, out):
        i_dtype = out.dtype
        if self.use_hadamard_transform:  # (b,l,h,c)
            out = hadamard_transform(
                out.float(), scale=1 / (math.sqrt(out.size(-1)))
            ).to(i_dtype)
        if self.online_full_had: 
            if self.down_dim != 1:
                shape = out.shape # (bs,seq_len,dim)
                out = out.reshape(shape[0],shape[1],-1,self.down_dim).transpose(2,3)
                out = hadamard_utils.matmul_hadU_cuda(out.contiguous(), self.had_K, self.K).transpose(2,3).reshape(shape)
            else:
                out = hadamard_utils.matmul_hadU_cuda(out, self.had_K, self.K).to(i_dtype)
        elif self.online_partial_had:
            init_shape = out.shape  # (b.h,l,c)
            if (
                self.K == 1
            ):  
                # out = fast_hadamard_transform.hadamard_transform(out.reshape(-1, init_shape[-1]//self.had_dim, self.had_dim).transpose(1, 2),# (b,l,c) -> (bl,h,c) -> (bl,c,h)
                #                                                scale=1/math.sqrt(init_shape[-1]//self.had_dim)).transpose(1, 2)
                out = fast_hadamard_transform.hadamard_transform(
                    out.permute(0, 2, 3, 1), scale=1 / math.sqrt(out.shape[1])
                ).permute(
                    0, 3, 1, 2
                )  # (b h l d) -> (b l d h) -> (b h l c)
            else:
                # out = (self.had_K.to(out.dtype) @ out.reshape(-1, init_shape[-1]//self.had_dim, self.had_dim)) / math.sqrt(init_shape[-1]//self.had_dim).to(i_dtype)
                out = ((out.permute(0,2,3,1) @ self.had_K.to(dtype=out.dtype,device=out.device)).permute(0,3,1,2)/(math.sqrt(self.K))).to(i_dtype)
                # out = (self.had_K.to(out.dtype) @ out.permute(0, 2, 3, 1)).permute(
                #     0, 3, 1, 2
                # ) / math.sqrt(self.K).to(i_dtype)
        return out


class QuantLinear(nn.Linear):
    weight: torch.Tensor
    bias: torch.Tensor

    def __init__(
        self,
        org_module: nn.Linear,
        weight_quant_params=dict(bits=8, sym=True, dynamic_method="pertoken"),
        act_quant_params=dict(bits=16, sym=True, dynamic_method="pertoken"),
    ):
        nn.Module.__init__(self)
        self.register_buffer("weight", org_module.weight.data)
        if org_module.bias is not None:
            self.register_buffer("bias", org_module.bias.data)
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features

        self.use_temporary_parameter = False
        self.temp_weight = self.temp_bias = None

        self.weight_quantizer = Quantizer(**weight_quant_params)
        self.act_quantizer = Quantizer(**act_quant_params)
        self.use_act_quant = False
        self.use_weight_quant = False
        self.cur_sample = 0

    def forward(self, x):
        if self.use_temporary_parameter:
            weight = self.temp_weight
            if self.temp_bias is not None:
                bias = self.temp_bias
            else:
                bias = self.bias
        else:
            weight = self.weight
            bias = self.bias

        if self.use_weight_quant:
            weight = self.weight_quantizer(weight)
        x = x.to(weight.device)
        out = F.linear(x, weight, bias)
        if self.use_act_quant:
            out = self.act_quantizer(out)

        return out


class QuantMatmul(nn.Module):
    def __init__(
        self,
        act_quant_parmas=dict(bits=8, sym=True, dynamic_method="pertoken"),
        matmul_func=torch.matmul,
        is_pvmat =False,
        is_qkmat = False,
    ):
        super().__init__()
        self.matmul_func = matmul_func
        self.is_pvmat = is_pvmat
        self.is_qkmat = is_qkmat
        self.act_quantizer = Quantizer(**act_quant_parmas)
        self.use_act_quant = False

    def forward(self, x1, x2):
        if self.is_qkmat:
            out = torch.matmul(x1.float(), x2.float())
        else:
            out = torch.matmul(x1,x2)
        # out = self.may_hadamard_transform(out)
        if self.is_pvmat:
            b,h,l,c = out.shape
            out = out.transpose(1, 2).contiguous().reshape(b,l, h*c)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantROPE(nn.Module):
    def __init__(
        self,
        rope_quant_parmas=dict(bits=8, sym=True, dynamic_method="pertensor"),
        act_quant_params=dict(bits=8, sym=True, dynamic_method="pertensor"),
    ):
        super().__init__()
        self.rope_quantizer = Quantizer(**rope_quant_parmas)
        self.act_quantizer = Quantizer(**act_quant_params)

        self.use_temporary_parameter = False
        self.pre_rope_Q = None  # TODO decide weather use transpose
        self.post_rope_Q = None

        self.use_act_quant = False
        self.use_weight_quant = False

    def forward(self, x, Wrope):
        if self.use_temporary_parameter:
            if self.pre_rope_Q is not None:
                Wrope = (self.pre_rope_Q.T.to(x.dtype)) @ Wrope.to(x.dtype)
            if self.post_rope_Q is not None:
                Wrope = Wrope.to(x.dtype) @ self.post_rope_Q.to(x.dtype)
        if self.use_weight_quant:
            Wrope = self.rope_quantizer(Wrope)

        out = torch.matmul(x, Wrope)
        # out = self.may_hadamard_transform(out)
        if self.use_act_quant:
            b,l,h,d = out.shape
            out = out.reshape(b,l,-1)
            out = self.act_quantizer(out)
            out = out.reshape(b,l,h,d)

        return out


class QuantRMSNorm(nn.Module):
    weight: torch.Tensor

    def __init__(
        self,
        ori_norm,
        act_quant_params=dict(bits=8, symmetric=True, dynamic_method="perchannel"),
    ):
        super().__init__()
        self.eps = ori_norm.variance_epsilon
        self.act_quantizer = Quantizer(**act_quant_params)
        self.register_buffer("weight", ori_norm.weight.data)
        self.bias = None

        self.use_temporary_parameter = False
        self.temp_weight = self.temp_bias = None

        self.use_act_quant = False
        self.fuse_weight = False 

    def forward(self, hidden_states):
        i_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        if self.fuse_weight:
            out = hidden_states.to(i_dtype)
        else:
            if self.use_temporary_parameter and self.temp_weight is not None:
                weight = self.temp_weight
                bias = self.temp_bias
            else:
                weight = self.weight
                bias = self.bias if hasattr(self, "bias") else None
            out = (
                (weight * hidden_states.to(i_dtype) + bias.to(i_dtype))
                if bias is not None
                else (weight * hidden_states.to(i_dtype))
            )
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantAdd(nn.Module):
    def __init__(
        self, act_quant_parmas=dict(bits=8, sym=True, dynamic_method="pertoken")
    ):
        super().__init__()
        self.act_quantizer = Quantizer(**act_quant_parmas)
        self.use_act_quant = False

    def forward(self, x1, x2):
        out = x1.to(x2.device) + x2
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantSoftmax(nn.Module):
    def __init__(self, act_quant_params: dict = dict(), dim=-1):
        super().__init__()
        self.act_quantizer = Quantizer(**act_quant_params)
        self.dim = dim

        self.use_act_quant = False

    def forward(self, attn_weights, attn_mask=None):
        i_dtype = attn_weights.dtype
        if attn_mask is not None:
            attn_weights = (attn_weights + attn_mask.to(attn_weights.device))
        out = F.softmax(attn_weights, dim=self.dim, dtype=torch.float32)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out.to(i_dtype)




class QuantMul(nn.Module):
    def __init__(
        self, act_quant_param=dict(bits=8, sym=True, dynamic_method="pertoken")
    ):
        super().__init__()
        self.act_quantizer = Quantizer(**act_quant_param)
        self.use_act_quant = False

    def forward(self, x1, x2):
        out = x1 * x2

        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantSiLU(nn.Module):
    def __init__(
        self, act_quant_params=dict(bits=8, sym=True, dynamic_method="pertoken")
    ):
        super().__init__()
        self.act_func = F.silu
        self.act_quantizer = Quantizer(**act_quant_params)
        self.use_act_quant = False
        self.smooth = None

    def forward(self, x):
        if self.smooth is None:
            out = F.silu(x)
        else:
            out = x * F.sigmoid(x / self.smooth.to(x.device))
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantEmbedding(nn.Embedding):
    def __init__(self, ori: nn.Embedding, act_quant_params=dict(bits=16)):
        super().__init__(
            num_embeddings=ori.num_embeddings,
            embedding_dim=ori.embedding_dim,
            padding_idx=ori.padding_idx,
            max_norm=ori.max_norm,
            norm_type=ori.norm_type,
            scale_grad_by_freq=ori.scale_grad_by_freq,
            sparse=ori.sparse,
            _weight=ori.weight,
            _freeze=True,
            device=ori.weight.device,
            dtype=ori.weight.dtype,
        )
        self.use_temporary_parameter = False
        self.Q = None
        del self.weight
        self.register_buffer("weight", ori.weight.data)
        self.act_quantizer = Quantizer(**act_quant_params)
        self.use_act_quant = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = F.embedding(
            input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        if self.use_temporary_parameter:
            out = torch.matmul(out.float(), self.Q.float().to(out.device)).to(out.dtype)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out
