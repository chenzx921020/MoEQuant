# MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language Models via Expert-Balanced Sampling and Affinity Guidance

## Abstract
Mixture-of-Experts (MoE) large language models (LLMs), which leverage dynamic routing and sparse activation to enhance efficiency and scalability, have achieved higher performance while reducing computational costs. However, these models face significant memory overheads, limiting their practical deployment and broader adoption. Post-training quantization (PTQ), a widely used method for compressing LLMs, encounters severe accuracy degradation and diminished generalization performance when applied to MoE models. This paper investigates the impact of MoE’s sparse and dynamic characteristics on quantization and identifies two primary challenges: (1) Inter-expert imbalance, referring to the uneven distribution of samples across experts, which leads to insufficient and biased calibration for less frequently utilized experts; (2) Intra-expert imbalance, arising from MoE's unique aggregation mechanism, which leads to varying degrees of correlation between different samples and their assigned experts. To address these challenges, we propose MoEQuant, a novel quantization framework tailored for MoE LLMs. MoEQuant includes two novel techniques: 1) Expert-Balanced Self-Sampling (EBSS) is an efficient sampling method that efficiently constructs a calibration set with balanced expert distributions by leveraging the cumulative probabilities of tokens and expert balance metrics as guiding factors. 2) Affinity-Guided Quantization (AGQ), which incorporates affinities between experts and samples into the quantization process, thereby accurately assessing the impact of individual samples on different experts within the MoE layer. Experiments demonstrate that MoEQuant achieves substantial performance gains (more than 10 points accuracy gain in the HumanEval for DeepSeekMoE-16B under 4-bit quantization) and boosts efficiency.

## Installation

### Env

```
pip install -e .
```
And you need extraly install fast hadamard tranformation, please refer to https://github.com/Dao-AILab/fast-hadamard-transform/ .
You can put it in `third-party` to compile

### EBSS Data
We provide the EBSS data in `./EBSS_data` directory.

### Test Data 
You need to download test data in `datasets`, to ensure that the evaluation can reach them.

## MoEQuant implementation

A example for `Mixtral-8x7b`, including quantization and evaluation on 9 tasks:
```
CUDA_VISIBLE_DEVICES=0 python fake_quant/main.py --model /ssd/model/mixtral_8x7b_v1 --fp32_had  --a_bits 16 --w_bits 4 --v_bits 16 --k_bits 16 --bsz 1 --w_clip --save_qmodel_path ./moequant_mixtral_base.pth --quant_test --nsamples 128  --human_res ./res_moe_mixtral_base --EBSS_calib --calib_path ./gen_data/EBSS_mixtral_8x7b.json  --rotate --AGQ_GPTQ
```
### Results
After evaluation, you can get results as below:

| Model | Method    | WikiText2 PPL ↓ | C4 PPL ↓ | MMLU | HumanEval | GSM8K | BoolQ | Hellaswag | OpenBookQA | MathQA | AVG Accuracy |
| ----- | --------- | --------------- | -------- | ---- | --------- | ----- | ----- | --------- | ---------- | ------ | ------------ |
|Mixtral-8x7B| FP        |    3.84         |  6.87    | 70.50|  32.93    |  65.88| 85.23 |   64.88   |  35.80     |  42.41 |   56.80      |
| | RTN|5.41|8.13|62.20|28.05|27.90|80.85|61.73|32.20|37.35|47.18|
| | OmniQuant |    4.19         |  7.20    | 68.10|  34.75    |  57.01| 84.13 |   63.03   |  33.00     |  41.91 |   54.56      |
| |AWQ|5.01|7.98|62.75|25.00|38.67|79.97|62.11|33.60|38.43|48.64|
| |MoEQuant<sup>+</sup>| 5.15|7.84|64.66|25.45|50.66|81.03|62.73|34.00|39.77|51.19|
| | GPTQ|4.84|8.08|64.30|24.39|42.15|83.03|58.50|32.00|37.52|48.84|
| | Quarot+GPTQ|4.03|7.67|68.50|27.60|57.92|84.22|64.08|30.60|41.07|53.42|
| | MoEQuant<sup>++</sup>  |    4.12         |  7.34    | 69.60|  32.15    |  61.79| 84.98 |   64.05   |  33.60     |  42.95 |   **55.58**      |