import torch
import typing
import transformers
import utils
import os,sys
import logging

from quant_layers.quant_layer import QuantDecoderLayer,QuantRMSNorm,QuantLinear,QuantEmbedding,Quantizer
import deepseek_moe_16b_chat
import mixtral_model

OPT_MODEL = transformers.models.opt.modeling_opt.OPTForCausalLM
OPT_LAYER = transformers.models.opt.modeling_opt.OPTDecoderLayer
LLAMA_MODEL = transformers.models.llama.modeling_llama.LlamaForCausalLM
LLAMA_LAYER = transformers.models.llama.modeling_llama.LlamaDecoderLayer
QWEN_MODEL = transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeForCausalLM
QWEN_LAYER = transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeDecoderLayer
DEEPSEEK_MODEL = deepseek_moe_16b_chat.modeling_deepseek.DeepseekForCausalLM
DEEPSEEK_LAYER = deepseek_moe_16b_chat.modeling_deepseek.DeepseekDecoderLayer
MIXTRAL_MODEL = mixtral_model.modeling_mixtral.MixtralForCausalLM
MIXTRAL_LAYER = mixtral_model.modeling_mixtral.MixtralDecoderLayer

def model_type_extractor(model):
    if isinstance(model, LLAMA_MODEL):
        return LLAMA_MODEL
    elif isinstance(model, OPT_MODEL):
        return OPT_MODEL
    elif isinstance(model, QWEN_MODEL):
        return QWEN_MODEL
    elif isinstance(model, DEEPSEEK_MODEL):
        return DEEPSEEK_MODEL
    elif isinstance(model, MIXTRAL_MODEL):
        return MIXTRAL_MODEL
    else:
        raise ValueError(f'Unknown model type {model}')

def skip(*args, **kwargs):
    # This is a helper function to save time during the initialization! 
    pass

def get_rope_function_name(model):
    if isinstance(model, LLAMA_MODEL) or isinstance(model, QWEN_MODEL):
        return "apply_rotary_pos_emb"
    raise NotImplementedError


def get_layers(model):
    if isinstance(model, OPT_MODEL):
        return model.model.decoder.layers
    if isinstance(model, LLAMA_MODEL) or isinstance(model, QWEN_MODEL):
        return model.model.layers
    raise NotImplementedError


def get_llama(model_name, hf_token):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = transformers.LlamaForCausalLM.from_pretrained(model_name, torch_dtype='auto',
                                                          use_auth_token=hf_token,
                                                          low_cpu_mem_usage=True)
    model.seqlen = 2048
    logging.info('---> Loading {} Model with seq_len: {}'.format(model_name, model.seqlen))
    return model

def get_qwen(model_name, hf_token, args):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = transformers.Qwen2MoeForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16,
                                                          use_auth_token=hf_token,attn_implementation = "eager",device_map='cuda')
                                                        #   low_cpu_mem_usage=True,attn_implementation = "eager",device_map='auto')
    model.seqlen = 2048
    logging.info('---> Loading {} Model with seq_len: {}'.format(model_name, model.seqlen))
    if args.online_hadamard:
        for name,module in model.named_modules():
            if 'down_proj' in name and isinstance(module, torch.nn.Linear):
                if 'mlp.experts' in name:
                    new_module = torch.nn.Linear(module.in_features+256, module.out_features,dtype=module.weight.dtype) # (1048+256)/52 = 128
                elif 'mlp.shared_expert' in name:
                    new_module = torch.nn.Linear(module.in_features+1024, module.out_features,dtype=module.weight.dtype)
                with torch.no_grad():
                    new_module.weight[:, :module.in_features] = module.weight.data
                    if module.bias is not None:
                        new_module.bias[:module.out_features].copy_(module.bias)
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                if parent_name:  # 如果模块不是顶层模块
                    parent = dict(model.named_modules())[parent_name]
                    setattr(parent, name.split('.')[-1], new_module)
                else:  # 如果模块是顶层模块
                    setattr(model, name, new_module)
                    
        model.config.intermediate_size += 1024
        model.config.moe_intermediate_size +=256
    model.seqlen=2048
    layers = model.model.layers
    args.embed_quant_params = dict(bits=args.a_bits,sym=not args.a_asym,dynamic=True,dynamic_method="perchannel")
    # 设置每一层的量化参数
    args.weight_quant_params = dict(bits=args.w_bits,sym=not args.w_asym,groupsize=args.w_groupsize,dynamic=True,dynamic_method="pertoken")
    # 学术量化的激活量化参数
    args.norm_quant_params = dict(bits=args.a_bits,sym=not args.a_asym,groupsize=args.a_groupsize,clip_ratio=args.a_clip_ratio,dynamic=True,dynamic_method=args.a_dynamic_method) # qkv 的输入量化, up gate的输入量化
    args.ropek_quant_params = dict(bits=args.a_bits,sym=not args.k_asym,groupsize=args.k_groupsize,clip_ratio=args.k_clip_ratio,dynamic=True,dynamic_method=args.a_dynamic_method) # ropek的输出量化
    args.v_proj_quant_params = dict(bits=args.a_bits,sym=not args.v_asym,groupsize=args.v_groupsize,clip_ratio=args.v_clip_ratio,dynamic=True,dynamic_method=args.a_dynamic_method) # v的输出量化
    args.pv_matmul_quant_params = dict(bits=args.a_bits,sym=not args.a_asym,groupsize=args.a_groupsize,clip_ratio=args.k_clip_ratio,dynamic=True,dynamic_method=args.a_dynamic_method)# o_proj的输入量化
    args.mul_quant_params = dict(bits=args.a_bits,sym=not args.a_asym,groupsize=args.a_groupsize,clip_ratio=args.k_clip_ratio,dynamic=True,dynamic_method=args.a_dynamic_method) #down的输入量化
    # TODO 默认用a_bits,但是仍然可以进行指定
    # fully_quant相关的激活的量化参数
    # moe block
    args.expert_gate_quant_params = dict(bits=args.a_bits,sym=not args.a_asym,groupsize=args.a_groupsize,clip_ratio=args.a_clip_ratio,dynamic=True,dynamic_method=args.a_dynamic_method)
    args.shared_gate_quant_params = dict(bits=args.a_bits,sym=not args.a_asym,groupsize=args.a_groupsize,clip_ratio=args.a_clip_ratio,dynamic=True,dynamic_method=args.a_dynamic_method)
    # others
    args.q_proj_quant_params = dict(bits=args.a_bits,sym=not args.a_asym,groupsize=args.a_groupsize,clip_ratio=args.a_clip_ratio,dynamic=True,dynamic_method=args.a_dynamic_method) # q的输出量化
    args.ropeq_quant_params = dict(bits=args.a_bits,sym=not args.a_asym,groupsize=args.a_groupsize,clip_ratio=args.a_clip_ratio,dynamic=True,dynamic_method=args.a_dynamic_method) # ropeq的输出量化
    args.k_proj_quant_params = dict(bits=args.a_bits,sym=not args.k_asym,groupsize=args.a_groupsize,clip_ratio=args.a_clip_ratio,dynamic=True,dynamic_method=args.a_dynamic_method) # k的输出量化
    args.qk_matmul_quant_params = dict(bits=args.a_bits,sym=not args.a_asym,groupsize=args.a_groupsize,clip_ratio=args.a_clip_ratio,dynamic=True,dynamic_method=args.a_dynamic_method)   # qkmat的输出量化
    args.softmax_quant_params = dict(bits=args.a_bits,sym=not args.a_asym,groupsize=args.a_groupsize,clip_ratio=args.a_clip_ratio,dynamic=True,dynamic_method=args.a_dynamic_method) # softmax的输出量化
    args.o_proj_quant_params = dict(bits=args.a_bits,sym=not args.a_asym,groupsize=args.a_groupsize,clip_ratio=args.a_clip_ratio,dynamic=True,dynamic_method="perchannel") 
    args.resadd1_quant_params = dict(bits=args.a_bits,sym=not args.a_asym,groupsize=args.a_groupsize,clip_ratio=args.a_clip_ratio,dynamic=True,dynamic_method="perchannel")
    args.up_proj_quant_params = dict(bits=args.a_bits,sym=not args.a_asym,groupsize=args.a_groupsize,clip_ratio=args.k_clip_ratio,dynamic=True,dynamic_method=args.a_dynamic_method) # up_proj的输出量化
    args.gate_proj_quant_params = dict(bits=args.a_bits,sym=not args.a_asym,groupsize=args.a_groupsize,clip_ratio=args.k_clip_ratio,dynamic=True,dynamic_method=args.a_dynamic_method) # gate
    args.silu_quant_params = dict(bits=args.a_bits,sym=not args.a_asym,groupsize=args.a_groupsize,clip_ratio=args.k_clip_ratio,dynamic=True,dynamic_method=args.a_dynamic_method) # silu输入mul的输入
    args.down_proj_quant_params = dict(bits=args.a_bits,sym=not args.a_asym,groupsize=args.a_groupsize,clip_ratio=args.k_clip_ratio,dynamic=True,dynamic_method="perchannel") # down_proj的输出量化
    args.resadd2_quant_params = dict(bits=args.a_bits,sym=not args.a_asym,groupsize=args.a_groupsize,clip_ratio=args.k_clip_ratio,dynamic=True,dynamic_method="perchannel")
    for i in range(len(layers)):
        layers[i] = QuantDecoderLayer(model.config,layers[i],args)
    model.model.embed_tokens = QuantEmbedding(model.model.embed_tokens,args.embed_quant_params) # TODO add embed_tokens's quantizaton-config
    model.model.norm = QuantRMSNorm(model.model.norm,dict(bits=32))
    model.lm_head = QuantLinear(model.lm_head,dict(bits=32))
    return model


def get_opt(model_name):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = transformers.OPTForCausalLM.from_pretrained(model_name, torch_dtype='auto',
                                                        low_cpu_mem_usage=True)
    model.seqlen = model.config.max_position_embeddings
    logging.info('---> Loading {} Model with seq_len: {}'.format(model_name, model.seqlen))
    return model


def get_model(
    model_name, hf_token=None,args=None
):
    if 'llama' in model_name:
        return get_llama(model_name, hf_token)
    elif 'opt' in model_name:
        return get_opt(model_name)
    elif 'qwen' in model_name.lower():
        return get_qwen(model_name, hf_token, args)
    else:
        raise ValueError(f'Unknown model {model_name}')


def get_model_type(model):
    if isinstance(model, OPT_MODEL):
        model_type = OPT_MODEL
    elif isinstance(model, LLAMA_MODEL):
        model_type = LLAMA_MODEL
    elif isinstance(model, QWEN_MODEL):
        model_type = QWEN_MODEL
    elif isinstance(model, DEEPSEEK_MODEL):
        model_type = DEEPSEEK_MODEL
    elif isinstance(model, MIXTRAL_MODEL):
        model_type = MIXTRAL_MODEL
    else:
        raise ValueError(f'Unknown model type {model}')
    return model_type

def get_embeddings(model, model_type):# -> list[torch.nn.Module]:
    if model_type == LLAMA_MODEL or model_type== QWEN_MODEL or model_type == DEEPSEEK_MODEL or model_type == MIXTRAL_MODEL:
        return [model.model.embed_tokens]
    elif model_type == OPT_MODEL:
        return [model.model.decoder.embed_tokens, model.model.decoder.embed_positions]
    else:
        raise ValueError(f'Unknown model type {model_type}')


def get_transformer_layers(model, model_type):
    if model_type == LLAMA_MODEL or model_type== QWEN_MODEL or model_type == DEEPSEEK_MODEL or model_type == MIXTRAL_MODEL:
        return [layer for layer in model.model.layers]
    elif model_type == OPT_MODEL:
        return [layer for layer in model.model.decoder.layers]
    else:
        raise ValueError(f'Unknown model type {model_type}')


def get_lm_head(model, model_type):
    if model_type == LLAMA_MODEL or model_type== QWEN_MODEL or model_type == DEEPSEEK_MODEL or model_type == MIXTRAL_MODEL:
        return model.lm_head
    elif model_type == OPT_MODEL:
        return model.lm_head
    else:
        raise ValueError(f'Unknown model type {model_type}')

def get_pre_head_layernorm(model, model_type):
    if model_type == LLAMA_MODEL:
        pre_head_layernorm = model.model.norm
        assert isinstance(pre_head_layernorm,
                          transformers.models.llama.modeling_llama.LlamaRMSNorm)
    elif model_type ==  QWEN_MODEL:
        pre_head_layernorm = model.model.norm
        assert isinstance(pre_head_layernorm,QuantRMSNorm)
                          #transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeRMSNorm)
    elif model_type == DEEPSEEK_MODEL:
        pre_head_layernorm = model.model.norm
        assert isinstance(pre_head_layernorm, deepseek_moe_16b_chat.modeling_deepseek.DeepseekRMSNorm)
    elif model_type == MIXTRAL_MODEL:
        pre_head_layernorm = model.model.norm
        assert isinstance(pre_head_layernorm, mixtral_model.modeling_mixtral.MixtralRMSNorm)
    elif model_type == OPT_MODEL:
        pre_head_layernorm = model.model.decoder.final_layer_norm
        assert pre_head_layernorm is not None
    else:
        raise ValueError(f'Unknown model type {model_type}')
    return pre_head_layernorm

def get_mlp_bottleneck_size(model):
    model_type = get_model_type(model)
    if model_type == LLAMA_MODEL or model_type== QWEN_MODEL or model_type== DEEPSEEK_MODEL:
        return model.config.intermediate_size
    elif model_type == OPT_MODEL:
        return model.config.ffn_dim
    else:
        raise ValueError(f'Unknown model type {model_type}')

def replace_modules(
    root: torch.nn.Module,
    type_to_replace,
    new_module_factory,
    replace_layers: bool,
) -> None:
    """Replace modules of given type using the supplied module factory.

    Perform a depth-first search of a module hierarchy starting at root
    and replace all instances of type_to_replace with modules created by
    new_module_factory. Children of replaced modules are not processed.

    Args:
        root: the root of the module hierarchy where modules should be replaced
        type_to_replace: a type instances of which will be replaced
        new_module_factory: a function that given a module that should be replaced
            produces a module to replace it with.
    """
    for name, module in root.named_children():
        new_module = None
        if isinstance(module, type_to_replace):
            if replace_layers:  # layernorm_fusion.replace_layers case where transformer layers are replaced
                new_module = new_module_factory(module, int(name))
            else:  # layernorm_fusion.fuse_modules case where layernorms are fused
                new_module = new_module_factory(module)
        elif len(list(module.children())) > 0:
            replace_modules(module, type_to_replace, new_module_factory, replace_layers)

        if new_module is not None:
            setattr(root, name, new_module)


class RMSN(torch.nn.Module):
    """
    This class implements the Root Mean Square Normalization (RMSN) layer.
    We use the implementation from LLAMARMSNorm here:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L75
    """

    def __init__(self, mean_dim: int, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mean_dim = mean_dim
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(input_dtype)


def get_layer_io_save_path(args):
    return os.path.join(args.save_path, 'layer_io', f'{args.layer_idx:03d}.pt')

def capture_layer_io(model_type, layer, layer_input):
    def hook_factory(module_name, captured_vals, is_input):
        def hook(module, input, output):
            if is_input:
                captured_vals[module_name].append(input[0].detach().cpu())
            else:
                captured_vals[module_name].append(output.detach().cpu())
        return hook

    handles = []

    if model_type == LLAMA_MODEL:
        captured_inputs = {
            'k_proj': [],  # q_proj, v_proj has the same input as k_proj
            'o_proj': [],
            'gate_proj': [],  # up_proj has the same input as gate_proj
            'down_proj': []
        }

        captured_outputs = {
            'v_proj': [],
        }

        for name in captured_inputs.keys():
            module = getattr(layer.self_attn, name, None) or getattr(layer.mlp, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_inputs, True)))

        for name in captured_outputs.keys():
            module = getattr(layer.self_attn, name, None) or getattr(layer.mlp, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_outputs, False)))

    elif model_type == OPT_MODEL:
        captured_inputs = {
            'k_proj': [],  # q_proj, v_proj has the same input as k_proj
            'out_proj': [],
            'fc1': [],
            'fc2': []
        }
        captured_outputs = {
            'v_proj': [],
        }
        for name in captured_inputs.keys():
            # In OPT, fc1 and fc2 are directly contained in OPTDecoderLayer
            module = getattr(layer.self_attn, name, None) or getattr(layer, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_inputs, True)))

        for name in captured_outputs.keys():
            # In OPT, fc1 and fc2 are directly contained in OPTDecoderLayer
            module = getattr(layer.self_attn, name, None) or getattr(layer, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_outputs, False)))
    else:
        raise ValueError(f'Unknown model type {model_type}')

    # Process each sequence in the batch one by one to avoid OOM.
    for seq_idx in range(layer_input.shape[0]):
        # Extract the current sequence across all dimensions.
        seq = layer_input[seq_idx:seq_idx + 1].to(utils.DEV)
        # Perform a forward pass for the current sequence.
        layer(seq)

    # After processing all sequences, concatenate the accumulated inputs for each sub-layer across the batch.
    for module_name in captured_inputs:
        captured_inputs[module_name] = torch.cat(captured_inputs[module_name], dim=0)
    for module_name in captured_outputs:
        captured_outputs[module_name] = torch.cat(captured_outputs[module_name], dim=0)

    # Cleanup.
    for h in handles:
        h.remove()

    return {
        'input': captured_inputs,
        'output': captured_outputs
    }
