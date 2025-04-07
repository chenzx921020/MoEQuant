import torch,torch.nn as nn,torch.nn.functional as F,math
import hadamard_utils
import fast_hadamard_transform

from typing import Union,Literal

class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,s,zp,qmin,qmax):
        x_int = ((x/s).round() + zp).clip(qmin,qmax)
        return (x_int - zp) * s
    
    def backward(ctx,grad_output):
        return grad_output.clone(),None,None,None,None

def round_ste(x:torch.Tensor):
    return (x.round() - x).detach() + x

class Quantizer(nn.Module):
    def __init__(self,bits:int=16,sym=True,groupsize=-1,clip_ratio=1.0,
                 dynamic=True,dynamic_method:Union[Literal["pertoken"],Literal["pertensor"],Literal["perchannel"]]="pertoken",
                 static_calib_method="minmax",pot=False):
        super(Quantizer,self).__init__()
        self.bits = bits
        self.sym = sym
        self.group_size = groupsize
        self.clip_ratio = clip_ratio # not used
        self.qmax,self.qmin = (2**(bits-1)) -1 if sym else 2**(bits)-1 , -(2**(bits-1)) if sym else 0
        self.dynamic = dynamic
        self.dynamic_method = dynamic_method
        if self.bits == 16:
            self.dynamic_method == "pertensor"
        # 2. static config
        self.static_calib_method = static_calib_method
        self.is_observing = False
        self.scale = None
    
    def fake_quant(x):
        pass
    
    def forward(self,x):
        if self.bits > 16:
            return x
        x_dtype = x.dtype
        # 1.Dynamic quantization
        if self.dynamic:
            if self.bits >= 16:
                return x
            s,zp = self.dynamic_quant_params(x)
            return STE.apply(x,s,zp,self.qmin,self.qmax).to(x_dtype)
        # 2. Static Quantization
        else:
            s,zp = self.static_quant_params(x)
            if self.is_observing:
                return x
            return STE.apply(x,s,zp,self.qmin,self.qmax).to(x_dtype)
    
    def dynamic_quant_params(self,x):
        ori_shape = list(x.shape)
        if self.dynamic_method == "pertensor":
            xmax,xmin = torch.max(x),torch.min(x)
        elif self.dynamic_method == "pertoken" or self.group_size != -1:
            if self.group_size != -1:
                reshaped_x = x.reshape(*ori_shape[:-1],ori_shape[-1] // self.group_size, self.group_size)
                xmax,xmin = torch.amax(reshaped_x,dim=-1,keepdim=True),torch.amin(reshaped_x,dim=-1,keepdim=True)
            else:
                xmax,xmin = torch.amax(x,dim=tuple(range(min(2,x.dim()-1),x.dim())),keepdim=True),torch.amin(x,dim=tuple(range(min(2,x.dim()-1),x.dim())),keepdim=True) 
        elif self.dynamic_method == "perchannel":
            xmax,xmin = torch.amax(x,dim=list(range(0,x.dim()-1)),keepdim=True),torch.amin(x,dim=(0,1),keepdim=False)
        else:
            raise NotImplemented
        xmax,xmin = xmax*self.clip_ratio, xmin * self.clip_ratio
        if self.sym:
            scale = torch.maximum((xmin.clip(max=0))/self.qmin ,xmax.clip(min=1e-6)/(self.qmax))
            # scale = torch.max(xmax.abs(),xmin.abs()) / self.qmax
            zp = torch.zeros_like(scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            scale = (xmax - xmin) / self.qmax
            zp = torch.round(-xmin / self.scale)

        if self.group_size != -1:
            scale = scale.repeat(1, 1, 1, self.group_size).reshape(ori_shape)
            zp = zp.repeat(1, 1, 1, self.group_size).reshape(ori_shape)
        return scale, zp 
    
    def percent(self,x,percent:float):
        if self.dynamic_method == "pertensor":
            return torch.quantile(x.abs(),percent)
        elif self.dynamic_method == "perchannel":
            return torch.quantile(x.abs().reshape(-1,x.shape[-1]),percent,dim=0)
        else:
            raise NotImplemented
    
    def static_quant_params(self,x):
        if not self.is_observing:
            assert self.scale is not None,"must be set before static quantization"
            return self.scale,self.zp
        assert self.sym == True,"only support"
        if self.static_calib_method == "minmax":
            if self.scale is None:
                self.scale,self.zp  = self.dynamic_quant_params(x)
            else:
                scale,self.zp = self.dynamic_quant_params(x)
                self.scale = torch.max(self.scale,scale)
        elif "percent" in self.static_calib_method:
            percent = float(self.static_calib_method[7:])
            if self.scale is None:
                self.scale = self.percnet(x,percent)
                self.zp = torch.zeros_like(self.scale)
            else:
                scale = self.percent(x,percent)
                self.scale = scale * 0.01 + self.scale * 0.99
        return self.scale,self.zp
            
    def enable_dynamic(self,value):
        self.dynamic = value

    def enable_observer(self,value=True):
        self.dynamic = False
        self.is_observing = value
