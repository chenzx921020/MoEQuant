import math
import time
import tqdm
import torch
import torch.nn as nn
import utils
import quant_utils
import logging
import torch.nn.functional as F
import math

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
    ):
        W = self.layer.weight.data.clone()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)])
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)])
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()

        if actorder:
            Q = Q[:, invperm]

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning('NaN in weights')
            import pprint
            pprint.pprint(self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point)
            raise ValueError('NaN in weights')

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
        utils.cleanup_memory(verbos=False)

def build_prompt(text):
    return text        
        
@torch.no_grad()
def gptq_fwrd(model, tokenizer, dataloader, dev, args, bit_mask):
    '''
    From GPTQ repo 
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    logging.info('-----GPTQ Quantization-----')
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)
    dtype = model.lm_head.weight.dtype
    inps = torch.zeros(
        (args.nsamples, 2048, model.config.hidden_size), dtype=dtype, device=dev
        # (args.nsamples, 512, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

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
            if args.llmqat_calib:
                texts = batch["text"]
                queries = [build_prompt(query) for query in texts]
                tokenizer.pad_token = tokenizer.eos_token# for mixtral
                inputs = tokenizer(queries, return_tensors="pt", truncation=True, max_length=model.seqlen,padding=True).to(dev)
                inputs['input_ids'] =  F.pad(inputs['input_ids'],(0,model.seqlen-inputs['input_ids'].shape[1]))
                # inputs = tokenizer(queries, return_tensors="pt", truncation=True, max_length=512,padding=True).to(dev)
                # inputs['input_ids'] =  F.pad(inputs['input_ids'],(0,512-inputs['input_ids'].shape[1]))
                model(inputs['input_ids'])
            else:
                # model(batch.to(dev))
                model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    # layers[0] = layers[0].cpu()
    # model.model.embed_tokens = model.model.embed_tokens.cpu()
    # model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    quantizers = {}
    # sequential = [
    #             ['self_attn.k_proj.module', 'self_attn.v_proj.module', 'self_attn.q_proj.module'],
    #             ['self_attn.o_proj.module'],
    #             ['mlp.up_proj.module', 'mlp.gate_proj.module'],
    #             ['mlp.down_proj.module']
    #         ]
    for i in range(len(layers)):
        print(f'\nLayer {i}:', flush=True, end=' ')
        layer = layers[i]#.to(dev)
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        if 'deepseek' in args.model.lower() and i>=1:
            full['mlp.gate'] = model.model.layers[i].mlp.gate
        sequential = list(full.keys())
        # adjust sequential
        if 'qwen' in args.model.lower():
            new_seq = sequential[:4]+list(reversed(sequential[-4:]))+sequential[4:-4]
            seq1 = []
            seq2 = []
            for element in new_seq:
                if "down_proj" in element:
                    seq2.append(element)
                else:
                    seq1.append(element)
            sequential = seq1+seq2
        elif 'deepseek' in args.model.lower():
            if i>0:
                last_value = sequential.pop()
                sequential.insert(4,last_value)
        elif 'mixtral' in args.model.lower():
            seq1 = []
            seq2 = []
            for element in sequential:
                if "w2" in element:
                    seq2.append(element)
                else:
                    seq1.append(element)
            sequential = seq1+seq2
        for k in range(len(sequential)):
            sequential[k]=[sequential[k]]
            
        for names in sequential:
            subset = {n: full[n] for n in names}
            gptq = {}
            for name in subset:
                print(f'{name}', end='  ', flush=True)
                layer_weight_bits = args.w_bits
                layer_weight_sym = not(args.w_asym)
                if 'lm_head' in name:
                    layer_weight_bits = 16
                    continue
                if args.int8_down_proj and 'down_proj' in name:
                    layer_weight_bits = 8
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits, perchannel=True, sym=layer_weight_sym, mse=args.w_clip
                )
            def add_batch_deepseek(name):
                def tmp(_, inp, out):
                    if name =='mlp.gate':
                        gptq[name].add_batch(inp[0], out)
                    else:
                        gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            def add_batch_qwen(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                if 'deepseek' in args.model.lower():
                    handles.append(subset[name].register_forward_hook(add_batch_deepseek(name)))
                elif 'qwen' in args.model.lower() or 'mixtral' in args.model.lower():
                    handles.append(subset[name].register_forward_hook(add_batch_qwen(name)))
            # all sample 
            # layer.mlp.static_observer=True
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                layer_w_groupsize = args.w_groupsize
                # torch.save(gptq[name].H,'./hessian_saved_llmqat/block'+str(i)+'_'+name+'_H_llmqat.pth')
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=layer_w_groupsize, actorder=args.act_order, static_groups=False
                )
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()
        # layer.mlp.static_observer=False
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        # layers[i] = layer.cpu()
        layers[i] = layer
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    logging.info('-----GPTQ Quantization Done-----\n')
    return quantizers



       
@torch.no_grad()
def rtn_fwrd(model, dev, args, bit_mask):
    '''
    From GPTQ repo 
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    assert args.w_groupsize ==-1, "Groupsize not supported in RTN!"
    layers = model.model.layers
    torch.cuda.empty_cache()

    quantizers = {}

    for i in tqdm.tqdm(range(len(layers)), desc="(RtN Quant.) Layers"):
        layer = layers[i]#.to(dev)

        subset = quant_utils.find_qlayers(layer,
                                            layers=[torch.nn.Linear])

        for name in subset:
            layer_weight_bits = args.w_bits
            if 'lm_head' in name:
                layer_weight_bits = 16
                continue
            if args.int8_down_proj and 'down_proj' in name:
                layer_weight_bits = 8

            quantizer = quant_utils.WeightQuantizer()
            quantizer.configure(
                layer_weight_bits, perchannel=True, sym=not(args.w_asym), mse=args.w_clip
            )
            W = subset[name].weight.data
            quantizer.find_params(W)
            subset[name].weight.data = quantizer.quantize(W).to(
                layer.self_attn.q_proj.weight.dtype)
            quantizers['model.layers.%d.%s' % (i, name)] = quantizer.cpu()
        layers[i] = layer#.cpu()
        torch.cuda.empty_cache()
        del layer
            
    utils.cleanup_memory(verbos=True)
    return quantizers
