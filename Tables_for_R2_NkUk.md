**Table R1: Comparison of W4A4 Quantization Performance of MoEQuant and Quarot on 3 MoE LLMs.** QuaRort is overfitting due to the use of WikiText2 as the calibration set, and the average accuracy on the 7 downstream tasks is more reflective of PTQ performance.

| Model            | Method          | WikiText2 PPL | Avg Accuracy |
| ---------------- | --------------- | :-----------: | :----------: |
| Qwen-MoE-14B     | Float           |    7.22       |     51.22    |
|                  | QuaRot+GPTQ     |    8.40       |     46.30    |
|                  | MoEQuant        |    8.54       |     **48.62**    |
| DeepSeek-MoE-16B | Float           |    6.51       |     40.86    |
|                  | QuaRot+GPTQ     |    7.82       |     35.33    |
|                  | QuaRot+MoEQuant |    7.90       |     **37.84**    |
| Mixtral-8x7B     | Float           |    3.84       |     56.80    |
|                  | QuaRot+GPTQ     |    4.92       |     50.22    |
|                  | QuaRot+MoEQuant |    5.03       |     **53.15**    |

---

**Table R2: Performance of Different Quantization Methods on MoE LLMs across two Multi-step Reasoning Tasks (HumanEval and GSM8k).** "Gain" means the improvement of the current method compared to the previous method. 

| MODEL                 |   METHOD    | HuamnEval |   GSM8K   | AVG Accuracy |  Gain  |
| --------------------- | :---------: | :-------: | :-------: | :----------: | :----: |
| QWEN-MoE-14b-CHAT     |     FP      |   21.34   |   30.71   |    26.03     |   -    |
|                       |     RTN     |   7.32    |   9.70    |     8.51     |   -    |
|                       |    GPTQ     |   10.98   |   16.22   |    13.60     |   -    |
|                       | Quarot+GPTQ |   15.24   |   26.08   |    20.66     |   -    |
|                       | MoEQuant++  | **21.95** | **29.11** |  **25.53**   | 23.57% |
| DEEPSEEK-MoE-16b-CHAT |     FP      |   24.39   |   54.28   |    39.34     |   -    |
|                       |     RTN     |   10.41   |   28.88   |    19.65     |   -    |
|                       |    GPTQ     |   10.93   |   35.78   |    23.36     |   -    |
|                       | Quarot+GPTQ |   13.41   |   47.08   |    30.25     |   -    |
|                       | MoEQuant++  | **21.95** | **48.97** |  **35.46**   | 17.22% |
| QWEN-MoE-14b          |     FP      |   32.32   |   62.55   |    47.44     |   -    |
|                       |     RTN     |   14.63   |   16.07   |    15.35     |   -    |
|                       |    GPTQ     |   20.73   |   22.82   |    21.77     |   -    |
|                       | Quarot+GPTQ |   28.05   |   56.25   |    42.15     |   -    |
|                       | MoEQuant++  | **29.87** | **58.38** |  **44.13**   | 4.69%  |
| DEEPSEEK-MoE-16b      |     FP      |   26.83   |   20.16   |    23.50     |   -    |
|                       |     RTN     |   18.90   |   10.54   |    14.72     |   -    |
|                       |    GPTQ     |   21.34   |   11.60   |    16.47     |   -    |
|                       | Quarot+GPTQ |   22.56   |   19.18   |    20.87     |   -    |
|                       | MoEQuant++  | **25.00** | **19.18** |  **22.09**   | 5.85%  |
| MIXTRAL-8x7B          |     FP      |   32.93   |   65.88   |    49.41     |   -    |
|                       |     RTN     |   28.05   |   27.90   |    27.98     |   -    |
|                       |    GPTQ     |   24.39   |   42.15   |    24.27     |        |
|                       | Quarot+GPTQ |   27.60   |   57.92   |    42.76     |   -    |
|                       | MoEQuant++  | **32.15** | **61.79** |  **46.97**   | 9.84%  |

---

**Table R3: Time Cost Comparison of GPTQ and MoEQuant.** All test are conducted on a single A800 GPU. MoEQuant needs to generate 128 sequences of 512 length, in order to make full use of the computing power to improve efficiency, we stitch multiple sequences together in the batch dimension.

| Model            | Method   | Time Cost |
| ---------------- | -------- | --------- |
| Qwen-MoE-14B     | GPTQ     |   37 mins  |
|                  | MoEQuant |   54 mins  |
| DeepSeek-MoE-16B | GPTQ     |   41 mins  |
|                  | MoEQuant |   52 mins  | 
| Mixtral-8x7B     | GPTQ     |   73 mins  |
|                  | MoEQuant |   115 mins  |

---

**Table R4: 4-bit Quantization Performance of OmniQuant and MoEQuant on Qwen-MoE-14B.**

| Method    | WikiText2 PPL ↓ | C4 PPL ↓ | MMLU | HumanEval | GSM8K | BoolQ | Hellaswag | OpenBookQA | MathQA | AVG Accuracy |
| --------- | --------------- | -------- | ---- | --------- | ----- | ----- | --------- | ---------- | ------ | ------------ |
| FP        |    7.22         |  9.30    | 59.60| 32.32     | 62.55 |79.82  | 57.96     |  30.40     | 35.77  |   51.20      |
| OmniQuant |    7.67         |  9.98    | 56.30| 31.71     | 52.39 |78.20  | 56.58     | 29.40      | 33.63  |   48.31      |
| MoEQuant  |    7.55         |  9.62    | 58.30| 29.87     | 58.38 |78.04  | 56.87     | 30.20      | 35.50  |   **49.59**      |


**Table R5: 4-bit Quantization Performance of OmniQuant and MoEQuant on DeepSeek-MoE-16B.**

| Method    | WikiText2 PPL ↓ | C4 PPL ↓ | MMLU | HumanEval | GSM8K | BoolQ | Hellaswag | OpenBookQA | MathQA | AVG Accuracy |
| --------- | --------------- | -------- | ---- | --------- | ----- | ----- | --------- | ---------- | ------ | ------------ |
| FP        |     6.51        |   9.04   | 44.60|  26.83    | 20.16 | 72.72 | 58.06     |   32.20    | 31.49  |   40.86      |
| OmniQuant |     6.79        |   9.49   | 43.50|  21.95    | 18.65 | 73.82 | 56.67     |   32.40    | 31.02  |   39.72      |
| MoEQuant  |     6.78        |   9.22   | 42.20|  25.00    | 19.18 | 73.49 | 57.20     |   31.40    | 31.66  |   **40.01**      |

**Table R6: 4-bit Quantization Performance of OmniQuant and MoEQuant on Mixtral-8x7B.**

| Method    | WikiText2 PPL ↓ | C4 PPL ↓ | MMLU | HumanEval | GSM8K | BoolQ | Hellaswag | OpenBookQA | MathQA | AVG Accuracy |
| --------- | --------------- | -------- | ---- | --------- | ----- | ----- | --------- | ---------- | ------ | ------------ |
| FP        |    3.84         |  6.87    | 70.50|  32.93    |  65.88| 85.23 |   64.88   |  35.80     |  42.41 |   56.80      |
| OmniQuant |    4.19         |  7.20    | 68.10|  34.75    |  57.01| 84.13 |   63.03   |  33.00     |  41.91 |   54.56      |
| MoEQuant  |    4.12         |  7.34    | 69.60|  32.15    |  61.79| 84.98 |   64.05   |  33.60     |  42.95 |   **55.58**      |

---

**Table R7: Results of RTN, Omniquant, AWQ, GPTQ, Quarot+GPTQ and ours MoEQuant with 4-bit Weight Quantization among 9 Tasks on Qwen-MoE-14B, DeepSeekMoE-16B and Mixtral-8x7B.** where + denotes MoEQuant based on AWQ, ++ denotes MoEQuant based on Quarot+GPTQ. Notably, except for our proposed MoEQuant, other methods utilize Wikitext2 as the calibration dataset, which leads to overfitting on Wikitext2. Perplexity measured on the C4 dataset more accurately reflects the performance of different methods.

| Model | Method    | WikiText2 PPL ↓ | C4 PPL ↓ | MMLU | HumanEval | GSM8K | BoolQ | Hellaswag | OpenBookQA | MathQA | AVG Accuracy |
| ----- | --------- | --------------- | -------- | ---- | --------- | ----- | ----- | --------- | ---------- | ------ | ------------ |
| Qwen-moe-14B      | FP        |    7.22         |  9.30    | 59.60| 32.32     | 62.55 |79.82  | 57.96     |  30.40     | 35.77  |   51.20      |
|      | RTN       | 10.83 | 12.49 | 48.10 | 14.63 | 16.07| 72.11 | 51.42 | 25.80 | 30.08 | 36.89 |
|      | OmniQuant |    7.67         |  9.98    | 56.30| 31.71     | 52.39 |78.20  | 56.58     | 29.40      | 33.63  |   48.31      |
|      | AWQ | 8.59 | 10.93 |51.63 | 20.73 | 36.77| 71.96 | 54.78 | 30.40 | 31.39| 42.52 |
|      | MoEQuant<sup>+</sup> | 8.77 | 10.67 | 52.33| 22.10| 42.22|74.52|54.92|30.40|33.44|44.27|
|      | GPTQ | 8.00 | 10.99 | 53.70 | 20.73 | 22.82 | 73.52 | 52.70|29.40|28.27|40.16 |
|      | Quarot+GPTQ|7.43|10.11|57.90|28.05|56.25|78.77|56.54|29.00|36.48|49.00|
|      | MoEQuant<sup>++</sup>  | 7.55 |  9.62    | 58.30| 29.87     | 58.38 |78.04  | 56.87     | 30.20      | 35.50  |   **49.59**      |
| DeepSeek-MoE-16B|  FP        |     6.51        |   9.04   | 44.60|  26.83    | 20.16 | 72.72 | 58.06     |   32.20    | 31.49  |   40.86      |
| |RTN| 7.47|10.01|36.10|18.90|10.54|70.21|55.76|30.60|28.87|35.85|
| | OmniQuant |     6.79        |   9.49   | 43.50|  21.95    | 18.65 | 73.82 | 56.67     |   32.40    | 31.02  |   39.72      |
| | AWQ|6.80|9.50|40.57|25.00|17.06|71.65|56.42|32.20|31.76|39.23|
| | MoEQuant<sup>+</sup> | 6.94|9.32|41.20|25.00|18.90|71.98|56.79|32.12|31.82|39.68|
| |GPTQ|6.82|10.35|39.60|21.34|11.60|72.14|56.05|30.60|30.35|37.38|
| |Quarot+GPTQ|6.66|9.39|40.60|22.56|19.18|72.17|57.03|30.60|30.95|39.01|
| | MoEQuant<sup>++</sup>  |     6.78        |   9.22   | 42.20|  25.00    | 19.18 | 73.49 | 57.20     |   31.40    | 31.66  |   **40.01**      |
|Mixtral-8x7B| FP        |    3.84         |  6.87    | 70.50|  32.93    |  65.88| 85.23 |   64.88   |  35.80     |  42.41 |   56.80      |
| | RTN|5.41|8.13|62.20|28.05|27.90|80.85|61.73|32.20|37.35|47.18|
| | OmniQuant |    4.19         |  7.20    | 68.10|  34.75    |  57.01| 84.13 |   63.03   |  33.00     |  41.91 |   54.56      |
| |AWQ|5.01|7.98|62.75|25.00|38.67|79.97|62.11|33.60|38.43|48.64|
| |MoEQuant<sup>+</sup>| 5.15|7.84|64.66|25.45|50.66|81.03|62.73|34.00|39.77|51.19|
| | GPTQ|4.84|8.08|64.30|24.39|42.15|83.03|58.50|32.00|37.52|48.84|
| | Quarot+GPTQ|4.03|7.67|68.50|27.60|57.92|84.22|64.08|30.60|41.07|53.42|
| | MoEQuant<sup>++</sup>  |    4.12         |  7.34    | 69.60|  32.15    |  61.79| 84.98 |   64.05   |  33.60     |  42.95 |   **55.58**      |