import torch, torch.nn as nn, torch.nn.functional as F, math, warnings
from typing import *
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    LlamaRMSNorm,
    LlamaAttention,
    Cache,
    repeat_kv,
    LlamaDecoderLayer,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.activations import ACT2FN
from .quant_ops import *


def build_rotary_matrix(cos, sin):
    bsz, seq_len, head_dim = cos.size()
    cos2d = cos.reshape(-1, head_dim)
    sin2d = sin.reshape(-1, head_dim)
    weight = torch.diag_embed(cos2d)
    sin_diag = torch.diag_embed(sin2d)
    weight[:, head_dim // 2 :] -= sin_diag[:, : head_dim // 2]
    weight[:, : head_dim // 2] += sin_diag[:, head_dim // 2 :]
    return weight # (l,128,128)

class QuantMLP(nn.Module):
    def __init__(self, org_module: nn.Module, config: LlamaConfig, args):
        super().__init__()

        self.gate_proj = QuantLinear(
            org_module.gate_proj, args.weight_quant_params, args.gate_proj_quant_params
        )
        self.down_proj = QuantLinear(
            org_module.down_proj, args.weight_quant_params, args.down_proj_quant_params
        )
        self.up_proj = QuantLinear(
            org_module.up_proj, args.weight_quant_params, args.up_proj_quant_params
        )
        self.silu = QuantSiLU(
            args.silu_quant_params,
        )
        self.mul = QuantMul(args.mul_quant_params)

    def forward(self, x, w8 = False):
        if x.shape[0]==0:
            return x
        if w8==True:
            up_proj_rst = F.linear(x, self.up_proj.w8_weight.to(self.up_proj.weight.device)) #self.up_proj(x)
            gate_proj_rst = F.linear(x, self.gate_proj.w8_weight.to(self.gate_proj.weight.device)) #self.gate_proj(x)
            act_rst = self.silu(gate_proj_rst)
            mul_rst = self.mul(up_proj_rst, act_rst)
            down_proj_rst = F.linear(mul_rst, self.down_proj.w8_weight.to(self.down_proj.weight.device)) #self.down_proj(mul_rst)
        else:
            up_proj_rst = self.up_proj(x)
            gate_proj_rst = self.gate_proj(x)
            act_rst = self.silu(gate_proj_rst)
            mul_rst = self.mul(up_proj_rst, act_rst)
            down_proj_rst = self.down_proj(mul_rst)
        return down_proj_rst

class QuantMoeBlock(nn.Module):
    def __init__(self, org_module: nn.Module, config: LlamaConfig, args):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        # gating
        self.gate = QuantLinear(
            org_module.gate, args.weight_quant_params, args.expert_gate_quant_params
        )
        self.experts = nn.ModuleList(
            [QuantMLP(org_module.experts[i_expert], config=config, args=args) for i_expert in range(self.num_experts)]
        )

        self.shared_expert = QuantMLP(org_module.shared_expert,config=config, args=args)
        #Qwen2MoeMLP(config, intermediate_size=config.shared_expert_intermediate_size)
        self.shared_expert_gate = QuantLinear(org_module.shared_expert_gate, args.weight_quant_params, args.shared_gate_quant_params )  
        #torch.nn.Linear(config.hidden_size, 1, bias=False)
        self.static_observer = False
        if False:
            self.top1_cnt = [0] * 60
            self.top2_cnt = [0] * 60
            self.top3_cnt = [0] * 60
            self.top4_cnt = [0] * 60
        # self.gate_res = []
        # self.use_float_expert = False
    def forward(self, hidden_states: torch.Tensor,cur_sample: int=0) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        
        if self.static_observer==True:
            routing_weights, selected_experts = torch.topk(routing_weights, 60, dim=-1)
        else:
            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if False:
            for m in range(selected_experts.shape[0]):
                self.top1_cnt[selected_experts[m][0]] = self.top1_cnt[selected_experts[m][0]]+1
                self.top2_cnt[selected_experts[m][1]] = self.top2_cnt[selected_experts[m][1]]+1
                self.top3_cnt[selected_experts[m][2]] = self.top3_cnt[selected_experts[m][2]]+1
                self.top4_cnt[selected_experts[m][3]] = self.top4_cnt[selected_experts[m][3]]+1

        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # if self.use_float_expert:
        #     routing_weights = self.gate_res[0]
        # else:
        #     self.gate_res.append(routing_weights)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            expert_layer.gate_proj.cur_sample = cur_sample
            expert_layer.up_proj.cur_sample = cur_sample
            expert_layer.down_proj.cur_sample = cur_sample
            idx, top_x = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            # normal experts prediction
            # for act quant 
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            # score_med = routing_weights[top_x, idx, None].median()
            # device = expert_layer.up_proj.weight.device
            # current_hidden_states = torch.zeros(current_state.shape).cuda().half().to(device)
            # current_hidden_states[torch.where(routing_weights[top_x, idx, None]>score_med)[0],:] = expert_layer(current_state[torch.where(routing_weights[top_x, idx, None]>score_med)[0],:].to(device)) * routing_weights[top_x, idx, None][torch.where(routing_weights[top_x, idx, None]>score_med)[0],:].to(device)
            # current_hidden_states[torch.where(routing_weights[top_x, idx, None]<=score_med)[0],:] = expert_layer(current_state[torch.where(routing_weights[top_x, idx, None]<=score_med)[0],:].to(device),w8=True) * routing_weights[top_x, idx, None][torch.where(routing_weights[top_x, idx, None]<=score_med)[0],:].to(device)
            # w8 w4 mix predict
            # if 0 in idx:
            #     w8_thres_idx = (idx == 0).nonzero(as_tuple=True)[0][-1]+1
            # else:
            #     w8_thres_idx = 0
            # current_hidden_states = torch.zeros(current_state.shape)
            # # w8
            # current_hidden_states[:w8_thres_idx] = expert_layer(current_state[:w8_thres_idx],w8=True) * routing_weights[top_x,idx,None][:w8_thres_idx]
            # # w4
            # current_hidden_states[w8_thres_idx:] = expert_layer(current_state[w8_thres_idx:]) * routing_weights[top_x,idx,None][w8_thres_idx:]

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype).to(final_hidden_states.device))
        
        # shared_score_thres = 0.032
        # shared_score = F.sigmoid(self.shared_expert_gate(hidden_states))
        # shared_w8_thres_idx = torch.where(shared_score>shared_score_thres)
        # shared_w4_thres_idx = torch.where(shared_score<=shared_score_thres)
        # shared_expert_output = torch.zeros(hidden_states.shape).to(hidden_states.device).half()
        # shared_expert_output[shared_w8_thres_idx[0]] = self.shared_expert(hidden_states[shared_w8_thres_idx[0]],w8=True)
        # shared_expert_output[shared_w4_thres_idx[0]] = self.shared_expert(hidden_states[shared_w4_thres_idx[0]])
        # shared_expert_output = shared_score * shared_expert_output
        self.shared_expert.gate_proj.cur_sample = cur_sample
        self.shared_expert.up_proj.cur_sample = cur_sample
        self.shared_expert.down_proj.cur_sample = cur_sample
        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output

        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        # if self.use_float_expert:
        #     self.gate_res.pop(0)
        return final_hidden_states, router_logits

class QuantAttention(nn.Module):
    def __init__(self, org_module: LlamaAttention, config: LlamaConfig, args):
        super().__init__()

        self.layer_idx = org_module.layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        self.q_proj = QuantLinear(
            org_module.q_proj,
            args.weight_quant_params,
            args.q_proj_quant_params,
        )
        self.k_proj = QuantLinear(
            org_module.k_proj,
            args.weight_quant_params,
            args.k_proj_quant_params,
        )
        self.ropeq = QuantROPE(act_quant_params=args.ropeq_quant_params)
        self.ropek = QuantROPE(act_quant_params=args.ropek_quant_params)
        self.v_proj = QuantLinear(
            org_module.v_proj,
            args.weight_quant_params,
            args.v_proj_quant_params,
        )
        self.o_proj = QuantLinear(
            org_module.o_proj, args.weight_quant_params, args.o_proj_quant_params
        )
        self.qk_matmul = QuantMatmul(args.qk_matmul_quant_params,is_qkmat=True)
        self.pv_matmul = QuantMatmul(args.pv_matmul_quant_params,is_pvmat=True)

        self.softmax = QuantSoftmax(args.softmax_quant_params, -1)

        self.rotary_emb = org_module.rotary_emb

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):  
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        )  # .transpose(1, 2) # b.l,h,d
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )  # .transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(
            1, 2
        )  # b h l d
        kv_seq_len = value_states.shape[-2]
        # past_key_value = getattr(self, "past_key_value", past_key_value)
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len,self.layer_idx)
        cos, sin = self.rotary_emb(value_states,seq_len=kv_seq_len)
        cos,sin = cos[position_ids.to(cos.device)],sin[position_ids.to(sin.device)]
        rotary_matrix = build_rotary_matrix(cos, sin).to(query_states.device)

        key_states = self.ropek(key_states, rotary_matrix).transpose(
            1, 2
        )  # b l h d -> b h l d
        query_states = self.ropeq(query_states, rotary_matrix).transpose(1, 2)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = self.qk_matmul(query_states, key_states.transpose(2, 3)) / (
            math.sqrt(self.head_dim)
        )  # b h l d @ b h d l -> b h l l
        # if attention_mask is not None:  # no matter the length, we just slice it
        #     attn_weights = attn_weights + attention_mask

        attn_weights = self.softmax(attn_weights,attention_mask).to(key_states.dtype)  # b h l l
        attn_output = self.pv_matmul(
            attn_weights,value_states
        )  # b h l l @ b h l d -> b h l d
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value

class QuantDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, ori_layer: LlamaDecoderLayer, args):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = QuantAttention(ori_layer.self_attn, config=config, args=args)
        self.mlp = QuantMoeBlock(ori_layer.mlp, config=config, args=args)
        self.input_layernorm = QuantRMSNorm(
            ori_layer.input_layernorm, args.norm_quant_params
        )
        self.post_attention_layernorm = QuantRMSNorm(
            ori_layer.post_attention_layernorm, args.norm_quant_params
        )

        self.resadd1 = QuantAdd(args.resadd1_quant_params)
        self.resadd2 = QuantAdd(args.resadd2_quant_params)
        self.use_weight_quant = False
        self.use_act_quant = False
        self.use_fully_quant = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cur_sample: int = 0,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = self.resadd1(residual, hidden_states)
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states,cur_sample)[0]
        hidden_states = self.resadd2(residual, hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs

    def set_quant_state(
        self,
        use_weight_quant: bool = False,
        use_act_quant: bool = False,
        use_fully_quant: bool =False,
    ):
        if use_fully_quant and (not use_act_quant):
            use_act_quant = True
            print("error: use_fully_quant must be used with use_act_quant")
        self.use_weight_quant = use_weight_quant
        self.use_act_quant = use_act_quant
        self.use_fully_quant = use_fully_quant
        self.self_attn.q_proj.use_weight_quant = use_weight_quant
        self.self_attn.k_proj.use_weight_quant = use_weight_quant
        self.self_attn.v_proj.use_weight_quant = use_weight_quant
        self.self_attn.o_proj.use_weight_quant = use_weight_quant
        # shared experts
        self.mlp.shared_expert_gate.use_weight_quant = use_weight_quant
        self.mlp.shared_expert.gate_proj.use_weight_quant = use_weight_quant
        self.mlp.shared_expert.up_proj.use_weight_quant = use_weight_quant
        self.mlp.shared_expert.down_proj.use_weight_quant = use_weight_quant
        # experts
        self.mlp.gate.use_weight_quant = use_weight_quant
        for i in range(60):
            self.mlp.experts[i].gate_proj.use_weight_quant = use_weight_quant
            self.mlp.experts[i].up_proj.use_weight_quant =use_weight_quant
            self.mlp.experts[i].down_proj.use_weight_quant =use_weight_quant
        self.input_layernorm.use_act_quant = use_act_quant or use_fully_quant 
        self.self_attn.ropek.use_act_quant = use_act_quant or use_fully_quant 
        self.self_attn.v_proj.use_act_quant = use_act_quant or use_fully_quant 
        self.self_attn.pv_matmul.use_act_quant = use_act_quant or use_fully_quant 
        self.post_attention_layernorm.use_act_quant = use_act_quant or use_fully_quant 
        self.mlp.shared_expert.mul.use_act_quant = use_act_quant or use_fully_quant
        # self.mlp.mul.use_act_quant = use_act_quant or use_fully_quant  
        for i in range(60):
            self.mlp.experts[i].mul.use_act_quant = use_act_quant or use_fully_quant
        self.self_attn.q_proj.use_act_quant = use_fully_quant
        self.self_attn.k_proj.use_act_quant = use_fully_quant
        self.self_attn.ropeq.use_act_quant = use_fully_quant
        self.self_attn.qk_matmul.use_act_quant = use_fully_quant
        self.self_attn.softmax.use_act_quant = use_fully_quant
        self.self_attn.o_proj.use_act_quant = use_fully_quant
        self.resadd1.use_act_quant = use_fully_quant
        self.mlp.shared_expert_gate.use_act_quant = use_fully_quant
        self.mlp.shared_expert.gate_proj.use_act_quant = use_fully_quant
        self.mlp.shared_expert.up_proj.use_act_quant = use_fully_quant
        self.mlp.shared_expert.down_proj.use_act_quant = use_fully_quant
        # experts
        self.mlp.gate.use_act_quant = use_fully_quant
        for i in range(60):
            self.mlp.experts[i].gate_proj.use_act_quant = use_fully_quant
            self.mlp.experts[i].up_proj.use_act_quant =use_fully_quant
            self.mlp.experts[i].down_proj.use_act_quant =use_fully_quant
            self.mlp.experts[i].silu.use_act_quant = use_fully_quant
        self.resadd2.use_act_quant = use_fully_quant