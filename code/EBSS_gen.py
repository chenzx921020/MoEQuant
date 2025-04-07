from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import sys
import os
# from deepseek_moe_16b_chat.modeling_deepseek import DeepseekForCausalLM
from mixtral_model.modeling_mixtral import MixtralForCausalLM

model_path = ''

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = MixtralForCausalLM.from_pretrained(model_path,device_map='auto')

n_vocab = (np.random.choice(tokenizer.vocab_size, size=128, replace=False)).tolist()

i_start = 0 #sys.argv[1]

inner_loop = 0
outer_loop = 0

if not os.path.exists("gen_data"):
    os.mkdir("gen_data")

for i in range(len(n_vocab)):
    print(i)
    j = random.randint(3,6)
    input_ids = torch.tensor([[n_vocab[i]]]).cuda()
    print("generating")
    outputs1 = model.generate(input_ids, do_sample=False, max_length=j)
    if i ==0:
        outputs = model.generate(outputs1, do_sample=True, max_length=2048)
    else:
        outputs = model.generate(outputs1, do_sample=False, num_beams=4, max_length=2048)
    gen_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    text_dict = {"text" : gen_text[0]}
    with open("gen_data/mixtral_EBSS."+str(i_start).zfill(2)+".jsonl", "a") as f:
        f.write(json.dumps(text_dict))
        f.write('\n')
