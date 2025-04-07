import os,sys
import utils
import torch
import model_utils
import data_utils
import transformers
import quant_utils
import rotation_utils
import gptq_utils
import gptq_utils_moe
# import gptq_utils_hessian_group
import eval_utils
import hadamard_utils
import bit_mask 
from quant_layers.quant_layer import QuantDecoderLayer,QuantRMSNorm,QuantLinear,QuantEmbedding,Quantizer
from evaluation.evaluate_mmlu import eval_mmlu
from evaluation.evaluate_humaneval import eval_humaneval
from evaluation.evaluate_gsm8k import eval_gsm8k
from transformers import AutoModelForCausalLM
import torch.nn as nn
import json
import torch.nn.functional as F
from evaluation import eval_lm
from tqdm import tqdm
from datasets import load_dataset
import random
def build_prompt(text):
    return text 

def eval_ppl_c4(model,tokenizer,seqlen=2048,limit=-1):
    c4_testdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(c4_testdata) - 1)
            tmp = tokenizer(c4_testdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    c4_testloader = torch.hstack(valenc)
    c4_ppl = eval_ppl_(model, c4_testloader, seqlen, limit,"c4")
    print(f'c4 ppl : {c4_ppl}')


@torch.no_grad()
def eval_ppl_(model,test_loader,seqlen=2048,limit=-1,data_name='wiki'):
    nlls = []
    if data_name=='wiki':
        test_loader = test_loader['input_ids']
    # test_loader = test_loader['input_ids']
    nsamples = test_loader.numel() // seqlen
    # nsamples = 1
    # for i in tqdm(range(nsamples)):
    with tqdm(range(nsamples)) as pbar:
        pbar.set_description_str("evaling ppl")
        for i in pbar:
            batch = test_loader[:, (i * seqlen) : ((i + 1) * seqlen)].to('cuda')
            net_name = model.name.lower() if hasattr(model,"name") else type(model).__name__.lower()
            if "opt" in net_name:
                outputs = model.model.model.decoder(batch)
                hidden_states = outputs[0]
                logits = model.model.lm_head(hidden_states)
            elif "llama" in net_name or "mixtral" in net_name or "qwen" in net_name or 'deepseek' in net_name:
                outputs = model(batch)
                logits = outputs['logits'];outputs = None
            elif "falcon" in net_name:
                outputs = model.model.transformer(batch)
                hidden_states = outputs[0]
                logits = model.model.lm_head(hidden_states)
            elif "glm" in net_name:
                outputs = model(batch)
                logits = outputs['logits'];outputs = None
            shift_logits = logits[:, :-1, :]
            shift_labels = test_loader[:, (i * seqlen) : ((i + 1) * seqlen)][
                :, 1:
            ].to(logits.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)
            tmp_ppl =  torch.exp(torch.stack(nlls).sum() / ((i+1) * seqlen)).item()
            pbar.set_postfix_str(f"--{tmp_ppl:4.4}")
            if i == limit:
                break
    ppl = torch.exp(torch.stack(nlls).sum() / ((i+1) * seqlen))
    return ppl.item()



def main():
    args = utils.parser_gen()
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_id)
        wandb.config.update(args)
        
    transformers.set_seed(args.seed)
    if 'qwen' in args.model:
        model = model_utils.get_model(args.model, args.hf_token, args)
        test_loader = data_utils.get_loaders(
                args.eval_dataset,
                seed=args.seed,
                model=args.model,
                seqlen=2048,
                hf_token=args.hf_token,
                eval_mode=True
            )
    elif 'deepseek' in args.model:
        from deepseek_moe_16b_chat.modeling_deepseek import DeepseekForCausalLM
        model = DeepseekForCausalLM.from_pretrained(args.model,torch_dtype=torch.float16,
                                                    attn_implementation = "eager",device_map='cuda',trust_remote_code=True)
        model.seqlen = 2048
        test_loader = data_utils.get_loaders(
                args.eval_dataset,
                seed=args.seed,
                model=args.model,
                seqlen=2048,
                hf_token=args.hf_token,
                eval_mode=True
            )
    elif 'mixtral' in args.model.lower():
        from mixtral_model.modeling_mixtral import MixtralForCausalLM
        model = MixtralForCausalLM.from_pretrained(args.model,torch_dtype=torch.float16,
                                                    attn_implementation = "eager",device_map='auto',trust_remote_code=True)
        model.seqlen = 2048
        test_loader = data_utils.get_loaders(
                args.eval_dataset,
                seed=args.seed,
                model=args.model,
                seqlen=2048,
                hf_token=args.hf_token,
                eval_mode=True
            )
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False)
    
    # Rotate the weights
    if args.rotate:
        rotation_utils.fuse_layer_norms(model)
        rotation_utils.rotate_model(model, args)      
        utils.cleanup_memory(verbos=True)
        qlayers = model.model.layers
        if args.online_hadamard:
            for name in qlayers:
                if 'down_proj' in name:
                    if 'mlp.experts' in name:
                        had_K, K = hadamard_utils.get_hadK(model.config.moe_intermediate_size)
                    elif 'mlp.shared_expert' in name:
                        had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                    qlayers[name].online_full_had = True
                    qlayers[name].had_K = had_K
                    qlayers[name].K = K
                    qlayers[name].fp32_had = args.fp32_had
                if 'o_proj' in name:
                    had_K, K = hadamard_utils.get_hadK(model.config.num_attention_heads)
                    qlayers[name].online_partial_had = True
                    qlayers[name].had_K = had_K
                    qlayers[name].K = K
                    qlayers[name].had_dim = model.config.hidden_size//model.config.num_attention_heads
                    qlayers[name].fp32_had = args.fp32_had
    # else:
    #     rotation_utils.fuse_layer_norms(model)
    #     quant_utils.add_actquant(model) #Add Activation Wrapper to the model as the rest of the code assumes it is present
        
                
    if args.w_bits < 16:
        save_dict = {}
        if args.load_qmodel_path: # Load Quantized Rotated Model
            #assert args.rotate, "Model should be rotated to load a quantized model!"
            assert not args.save_qmodel_path, "Cannot save a quantized model if it is already loaded!"
            print("Load quantized model from ", args.load_qmodel_path)
            save_dict_w4 = torch.load(args.load_qmodel_path,map_location='cpu')
            model.load_state_dict(save_dict_w4["model"],strict=False)
            
            
        elif not args.w_rtn: # GPTQ Weight Quantization
            # assert "llama" in args.model, "Only llama is supported for GPTQ!"
            if args.EBSS_calib:
                dataset = []
                cnt = 0
                with open(args.calib_path, encoding='utf-8') as file:
                    for line in file:
                        dataset.append(json.loads(line))
                        cnt = cnt +1
                        if cnt==args.nsamples:
                            break
                trainloader = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=True) 
            else:
                if 'deepseek' in args.model or 'mixtral' in args.model or 'qwen' in args.model:
                    trainloader = data_utils.get_loaders(
                        args.cal_dataset, nsamples=args.nsamples,
                        seed=args.seed, model=args.model,
                        seqlen=2048, eval_mode=False
                    )
            # load bit settings
            bit_settings = None
            if args.AGQ_GPTQ:
                quantizers = gptq_utils_moe.gptq_fwrd(model, tokenizer, trainloader, utils.DEV, args, bit_settings)
            else:
                quantizers = gptq_utils.gptq_fwrd(model, tokenizer, trainloader, utils.DEV, args, bit_settings)
            save_dict["w_quantizers"] = quantizers
        else: # RTN Weight Quantization
            bit_settings=None
            quantizers = gptq_utils.rtn_fwrd(model, utils.DEV, args, bit_settings)
            save_dict["w_quantizers"] = quantizers
            
        if args.save_qmodel_path:
            save_dict["model"] = model.state_dict()
            torch.save(save_dict, args.save_qmodel_path)

    if args.quant_test:  
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False)  
        # ppl
        ppl = eval_ppl_(model,test_loader,seqlen=2048,limit=-1,data_name='wiki')
        print ('wikitext2 ppl is:', ppl)
        ppl_c4 = eval_ppl_c4(model,tokenizer,seqlen=2048, limit=-1)
        print ('c4 ppl is: ', ppl_c4)
        # gsm8k
        acc = eval_gsm8k(model,tokenizer,1)
        print("gsm8k acc:%.5f\n"%acc)
        
        # mmlu 
        acc = eval_mmlu(model,tokenizer)
        # humaneval
        if not os.path.exists(args.human_res):
            os.makedirs(args.human_res)
        acc = eval_humaneval(model,tokenizer,args.human_res)
        print ('just get preds, you should run the human eval: ')
        print ('python human_eval/evaluate_functional_correctness.py HumanEval_res.json --problem_file=data/HumanEval.json')
        # # boolq 
        acc = eval_lm(model,tokenizer, 'boolq', 0, 2)
        print("boolq acc:",acc)
        # hellaswag
        acc = eval_lm(model,tokenizer, 'hellaswag', 0, 2)
        print("hellaswag acc:",acc)
        # openbookqa
        acc = eval_lm(model,tokenizer, 'openbookqa', 0, 2)
        print("openbookqa acc:",acc)
        # mathqa
        acc = eval_lm(model,tokenizer, 'mathqa', 0, 2)
        print("mathqa acc:",acc)

if __name__ == '__main__':
    main()
