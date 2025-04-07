import math
import time
import tqdm
import torch
import torch.nn as nn
import utils
import quant_utils
import logging

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


@torch.no_grad()
def bit_calib(model, dataloader, dev, args):

    logging.info('-----bit calculate-----')
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)
    dtype = model.lm_head.weight.dtype
    # dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    bit_mask = []
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    bit_mask = torch.zeros(24,60)
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        sequential = list(full.keys())
        for k in range(len(sequential)):
            sequential[k]=[sequential[k]]


        gate_res = []
        def get_layer_output(name):
            def tmp(_, inp, out):
                gate_res.append(out)
            return tmp

        for names in sequential:
            subset = {n: full[n] for n in names}
            for name in subset:
                handles = []
                if 'mlp.gate' in name:
                    handles.append(subset[names[0]].register_forward_hook(get_layer_output(name)))
                    for j in range(args.nsamples):
                        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                for h in handles:
                    h.remove()
        top1_cnt = [0] * 60
        top2_cnt = [0] * 60
        top3_cnt = [0] * 60
        top4_cnt = [0] * 60
        score1_cnt = [0] * 60
        score2_cnt = [0] * 60
        score3_cnt = [0] * 60
        score4_cnt = [0] * 60
        for idx,gate in enumerate(gate_res):
            routing_weights = torch.nn.functional.softmax(gate, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights, 4, dim=-1)
            for m in range(selected_experts.shape[0]):
                top1_cnt[selected_experts[m][0]] = top1_cnt[selected_experts[m][0]]+1
                top2_cnt[selected_experts[m][1]] = top2_cnt[selected_experts[m][1]]+1
                top3_cnt[selected_experts[m][2]] = top3_cnt[selected_experts[m][2]]+1
                top4_cnt[selected_experts[m][3]] = top4_cnt[selected_experts[m][3]]+1
                score1_cnt[selected_experts[m][0]] = score1_cnt[selected_experts[m][0]]+routing_weights[m][0]
                score2_cnt[selected_experts[m][1]] = score2_cnt[selected_experts[m][1]]+routing_weights[m][1]
                score3_cnt[selected_experts[m][2]] = score3_cnt[selected_experts[m][2]]+routing_weights[m][2]
                score4_cnt[selected_experts[m][3]] = score1_cnt[selected_experts[m][3]]+routing_weights[m][3]
        top_cnt = torch.tensor([top1_cnt,top2_cnt,top3_cnt,top4_cnt])
        score_cnt = torch.tensor([score1_cnt,score2_cnt,score3_cnt,score4_cnt])
        ave_score = score_cnt.sum(dim=0)/top_cnt.sum(dim=0)
        first_row_indices = torch.argsort(ave_score, descending=True)
        threshold = ave_score[first_row_indices[10]]
        bit_mask[i][ave_score>threshold]=1
        
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    logging.info('-----bit accu Done-----\n')
    return bit_mask

