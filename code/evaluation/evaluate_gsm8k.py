import random
import tqdm
import os
import re
import sys
import torch
import numpy as np
import argparse
import jsonlines
import datasets
from datasets import load_from_disk,load_dataset
import json
from tqdm import tqdm


GSM8K_PATH='data/gsm8k/'

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"
fewshot_prompt = ""

def doc_to_text(doc):
        return fewshot_prompt + "\nQuestion: " + doc + "\nLet's think step by step\n"

def decode(tokens_list, tokenizer, raw_text_len):
    sents = []
    # print(len(tokens_list))
    for tokens in tokens_list:
        tokens = tokens.cpu().numpy().tolist()
        sent = tokenizer.decode(
            tokens[raw_text_len:])
        sent = sent.split('<|endoftext|>')[0]
        sent = sent.split('\n\n\n')[0]
        sent = sent.split("\n\n")[0]
        sent = sent.split("Question:")[0]
        sents.append(sent)
    return sents

def generate_sample(model, tokenizer, input_txt):
    dev =model.device
    #print(f"Input text: {input_txt}\n")
    inputs = tokenizer(input_txt, return_tensors="pt", truncation=True, max_length=2048).to(dev) #to('cuda')
    raw_text_len = len(inputs["input_ids"][0])
    outputs = model.generate(**inputs, do_sample=False, max_new_tokens=512)
    output_text = decode(outputs,tokenizer,raw_text_len)[0]
    #print(f"\nOutput text: {output_text}\n")
    return output_text


def extract_answer_hf(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return eval(match_str)
    else:
        return INVALID_ANS

def extract_answer(completion):
    try:
        last_number = re.findall(r'\d+', completion)[-1]
        return eval(last_number)
    except:
        return INVALID_ANS

def is_correct( completion, answer):
    gold = extract_answer_hf(answer)
    assert gold != INVALID_ANS, "No ground truth answer found in the document."
    return extract_answer(completion) == gold

def eval_gsm8k(model, tokenizer, batch_size):
    dev = 'cuda:0'#model.device
    f_output = jsonlines.Writer(open("gsm8k_res.jsonl", 'w', encoding='utf-8'))

    fewshot_prompt = open(GSM8K_PATH).read()
    test = []
    with open(GSM8K_PATH, 'r') as file:
        for line in file:
            test.append(json.loads(line))

    acc_res = []
    dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size)
    for batch in tqdm(dataloader):
        texts = batch["question"]
        queries = [fewshot_prompt + "\nQuestion: " + query + "\nLet's think step by step\n" for query in texts]
        inputs = tokenizer(queries, padding=False, return_tensors="pt", truncation=True, max_length=2048).to(dev) #to('cuda')
        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=256)
        outputs = outputs.tolist()
        for idx in range(len(outputs)):
            #print(f"Input text: {texts[idx]}\n")
            raw_text_len = len(inputs["input_ids"][idx])
            tokens = outputs[idx]
            sent = tokenizer.decode(
                tokens[raw_text_len:])
            sent = sent.split('<|endoftext|>')[0]
            sent = sent.split('\n\n\n')[0]
            sent = sent.split("\n\n")[0]
            sent = sent.split("Question:")[0]
            response = sent
            #print(f"\nOutput text: {sent}\n")
            answer= batch["answer"][idx]
            acc = is_correct(response, answer)
            f_output.write(batch)
            acc_res.append(acc)

    f_output.close()
    print("Acc: ",np.mean(acc_res))
    return np.mean(acc_res)
