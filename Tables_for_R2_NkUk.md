Table R1: W4A4 Performace of MoEquant on various models.

| Model            | Method          | WikiText2 PPL | Avg Accuracy |
| ---------------- | --------------- | :-----------: | :----------: |
| Qwen-MoE-14B     | QuaRot+GPTQ     |               |              |
|                  | MoEQuant        |               |              |
| DeepSeek-MoE-16B | QuaRot+GPTQ     |               |              |
|                  | QuaRot+MoEQuant |               |              |
| Mixtral-8x7B     | QuaRot+GPTQ     |               |              |
|                  | QuaRot+MoEQuant |               |              |

---

Table R2: Performance of different methods on different models for two tasks (HumanEval and GSM8k) that require multi-step reasoning.

| MODEL                 |   METHOD    | HuamnEval |   GSM8K   | Mean Acc  |  Gain  |
| --------------------- | :---------: | :-------: | :-------: | :-------: | :----: |
| QWEN-MoE-14b-CHAT     |     FP      |   21.34   |   30.71   |   26.03   |   -    |
|                       |     RTN     |   7.32    |   9.70    |   8.51    |   -    |
|                       | Quarot+GPTQ |   15.24   |   26.08   |   20.66   |   -    |
|                       | MoEQuant++  | **21.95** | **29.11** | **25.53** | 23.57% |
| DEEPSEEK-MoE-16b-CHAT |     FP      |   24.39   |   54.28   |   39.34   |   -    |
|                       |     RTN     |   10.41   |   28.88   |   19.65   |   -    |
|                       | Quarot+GPTQ |   13.41   |   47.08   |   30.25   |   -    |
|                       | MoEQuant++  | **21.95** | **48.97** | **35.46** | 17.22% |
| QWEN-MoE-14b          |     FP      |   32.32   |   62.55   |   47.44   |   -    |
|                       |     RTN     |   14.63   |   16.07   |   15.35   |   -    |
|                       | Quarot+GPTQ |   28.05   |   56.25   |   42.15   |   -    |
|                       | MoEQuant++  | **29.87** | **58.38** | **44.13** | 4.69%  |
| DEEPSEEK-MoE-16b      |     FP      |   26.83   |   20.16   |   23.50   |   -    |
|                       |     RTN     |   18.90   |   10.54   |   14.72   |   -    |
|                       | Quarot+GPTQ |   22.56   |   19.18   |   20.87   |   -    |
|                       | MoEQuant++  | **25.00** | **19.18** | **22.09** | 5.85%  |
| MIXTRAL-8x7B          |     FP      |   32.93   |   65.88   |   49.41   |   -    |
|                       |     RTN     |   28.05   |   27.90   |   27.98   |   -    |
|                       | Quarot+GPTQ |   27.60   |   57.92   |   42.76   |   -    |
|                       | MoEQuant++  | **32.15** | **61.79** | **46.97** | 9.84%  |

---

Table R3: Time Cost Comparison of GPTQ and MoEQuant

| Model            | Method   | Time Cost |
| ---------------- | -------- | --------- |
| Qwen-MoE-14B     | GPTQ     |           |
|                  | MoEQuant |           |
| DeepSeek-MoE-16B | GPTQ     |           |
|                  | MoEQuant |           |
| Mixtral-8x7B     | GPTQ     |           |
|                  | MoEQuant |           |

---

Table R4: 4-bit quantization performance of OmniQuant and MoEQuant on Qwen-MoE-14B.

| Method    | WikiText2 PPL ↓ | C4 PPL ↓ | MMLU | HumanEval | GSM8K | BoolQ | Hellaswag | OpenBookQA | MathQA | Accuracy AVG |
| --------- | --------------- | -------- | ---- | --------- | ----- | ----- | --------- | ---------- | ------ | ------------ |
| FP        |                 |          |      |           |       |       |           |            |        |              |
| OmniQuant |                 |          |      |           |       |       |           |            |        |              |
| MoEQuant  |                 |          |      |           |       |       |           |            |        |              |


Table R5: 4-bit quantization performance of OmniQuant and MoEQuant on DeepSeek-MoE-16B.

| Method    | WikiText2 PPL ↓ | C4 PPL ↓ | MMLU | HumanEval | GSM8K | BoolQ | Hellaswag | OpenBookQA | MathQA | Accuracy AVG |
| --------- | --------------- | -------- | ---- | --------- | ----- | ----- | --------- | ---------- | ------ | ------------ |
| FP        |                 |          |      |           |       |       |           |            |        |              |
| OmniQuant |                 |          |      |           |       |       |           |            |        |              |
| MoEQuant  |                 |          |      |           |       |       |           |            |        |              |

Table R6: 4-bit quantization performance of OmniQuant and MoEQuant on Mixtral-8x7B.

| Method    | WikiText2 PPL ↓ | C4 PPL ↓ | MMLU | HumanEval | GSM8K | BoolQ | Hellaswag | OpenBookQA | MathQA | Accuracy AVG |
| --------- | --------------- | -------- | ---- | --------- | ----- | ----- | --------- | ---------- | ------ | ------------ |
| FP        |                 |          |      |           |       |       |           |            |        |              |
| OmniQuant |                 |          |      |           |       |       |           |            |        |              |
| MoEQuant  |                 |          |      |           |       |       |           |            |        |              |
