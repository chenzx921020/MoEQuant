Table R1: W4A4 Performace of MoEquant on various models.

| Model            | Method          | WikiText2 PPL | Avg Accuracy |
| ---------------- | --------------- | :-----------: | :----------: |
| Qwen-MoE-14B     | Float           |    7.22       |     51.22    |
|                  | QuaRot+GPTQ     |    8.40       |     46.30    |
|                  | MoEQuant        |    8.54       |     49.02    |
| DeepSeek-MoE-16B | Float           |    6.51       |     40.86    |
|                  | QuaRot+GPTQ     |    7.82       |     32.33    |
|                  | QuaRot+MoEQuant |    7.90       |     36.84    |
| Mixtral-8x7B     | Float           |    3.84       |     56.80    |
|                  | QuaRot+GPTQ     |    4.82       |     50.22    |
|                  | QuaRot+MoEQuant |    5.03       |     53.15    |

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

Table R3: Time Cost Comparison of GPTQ and MoEQuant, for EBSS, we implement it on GPU A800, enabling the simultaneous generation of multiple batches of data

| Model            | Method   | Time Cost |
| ---------------- | -------- | --------- |
| Qwen-MoE-14B     | GPTQ     |   37 mins  |
|                  | MoEQuant |   54 mins  |
| DeepSeek-MoE-16B | GPTQ     |   41 mins  |
|                  | MoEQuant |   52 mins  | 
| Mixtral-8x7B     | GPTQ     |   73 mins  |
|                  | MoEQuant |   115 mins  |

---

Table R4: 4-bit quantization performance of OmniQuant and MoEQuant on Qwen-MoE-14B.

| Method    | WikiText2 PPL ↓ | C4 PPL ↓ | MMLU | HumanEval | GSM8K | BoolQ | Hellaswag | OpenBookQA | MathQA | Accuracy AVG |
| --------- | --------------- | -------- | ---- | --------- | ----- | ----- | --------- | ---------- | ------ | ------------ |
| FP        |    7.22         |  9.30    | 59.60| 32.32     | 62.55 |79.82  | 57.96     |  30.40     | 35.77  |   51.20      |
| OmniQuant |    7.67         |  9.98    | 56.30| 31.71     | 52.39 |78.20  | 56.58     | 29.40      | 33.63  |   48.31      |
| MoEQuant  |    7.55         |  9.62    | 58.30| 29.87     | 58.38 |78.04  | 56.87     | 30.20      | 35.50  |   49.59      |


Table R5: 4-bit quantization performance of OmniQuant and MoEQuant on DeepSeek-MoE-16B.

| Method    | WikiText2 PPL ↓ | C4 PPL ↓ | MMLU | HumanEval | GSM8K | BoolQ | Hellaswag | OpenBookQA | MathQA | Accuracy AVG |
| --------- | --------------- | -------- | ---- | --------- | ----- | ----- | --------- | ---------- | ------ | ------------ |
| FP        |     6.51        |   9.04   | 44.60|  26.83    | 20.16 | 72.72 | 58.06     |   32.20    | 31.49  |   40.86      |
| OmniQuant |     6.79        |   9.49   | 43.50|  21.95    | 18.65 | 73.82 | 56.67     |   32.40    | 31.02  |   39.72      |
| MoEQuant  |     6.78        |   9.22   | 42.20|  25.00    | 19.18 | 73.49 | 57.20     |   31.40    | 31.66  |   40.01      |

Table R6: 4-bit quantization performance of OmniQuant and MoEQuant on Mixtral-8x7B.

| Method    | WikiText2 PPL ↓ | C4 PPL ↓ | MMLU | HumanEval | GSM8K | BoolQ | Hellaswag | OpenBookQA | MathQA | Accuracy AVG |
| --------- | --------------- | -------- | ---- | --------- | ----- | ----- | --------- | ---------- | ------ | ------------ |
| FP        |    3.84         |  6.87    | 70.50|  32.93    |  65.88| 85.23 |   64.88   |  35.80     |  42.41 |   56.80      |
| OmniQuant |    4.19         |  7.20    | 68.10|  34.75    |  57.01| 84.13 |   63.03   |  33.00     |  41.91 |   54.56      |
| MoEQuant  |    4.12         |  7.34    | 69.60|  32.15    |  61.79| 84.98 |   64.05   |  33.60     |  42.95 |   55.58      |
