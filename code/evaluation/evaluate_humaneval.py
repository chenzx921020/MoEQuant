
import tqdm
import os
import jsonlines
import torch
"""
git clone https://github.com/openai/human-eval
$ pip install -e human-eval
evaluate_functional_correctness sample-output-file
"""

HUMANEVAL_PATH='data/humaneval'

def decode(tokens_list, tokenizer, raw_text_len):
    sents = []
    # print(len(tokens_list))
    for tokens in tokens_list:
        tokens = tokens.cpu().numpy().tolist()
        sent = tokenizer.decode(
            tokens[raw_text_len:])
        sent = sent.split("def ")[0]
        sent = sent.split('\n\n\n')[0]
        sent = sent.split("</s>")[0]
        sent = "    " + sent.lstrip()
        sents.append(sent)
    return sents

def generate_sample(model, tokenizer, input_txt):
    #print(f"Input text: {input_txt}\n")
    dev = 'cuda' #model.device
    inputs = tokenizer(input_txt, return_tensors="pt", truncation=True, max_length=2048).to(dev) #to('cuda')
    raw_text_len = len(inputs["input_ids"][0])
    outputs = model.generate(**inputs, do_sample=False, max_new_tokens=512)
    output_text = decode(outputs,tokenizer,raw_text_len)[0]
    #print(f"\nOutput text: \n{output_text}\n")
    return output_text

def eval_humaneval(model,tokenizer,logpath):
    f_output = jsonlines.Writer(open(os.path.join(logpath,"HumanEval_res.jsonl"), 'w', encoding='utf-8'))

    f = jsonlines.open(HUMANEVAL_PATH)
    with f_output as output:
        for jobj in tqdm.tqdm(f, desc='task_idx'):
            prompt = jobj['prompt']
            task_id = jobj['task_id']
            gen_sents = generate_sample(model, tokenizer, prompt)
            gen_jobjs = {'task_id': task_id, "completion": gen_sents} 
            output.write(gen_jobjs)
    f_output.close()

    # os.system("evaluate_functional_correctness HumanEval_res.jsonl")

    return 0
