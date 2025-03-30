Table R1: The influence of different calibration datasets on the balance of experts and quantization performance, the expert balance std denotes the standard deviation on frequency of experts

| Model | Calib Dataset | Expert Balance STD  | WIKITEXT2 | MMLU | HUMANEVAL | GSM8K | Accuracy AVG | 
| :--------------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DeepSeek-MoE-16B | Float |       | 6.51 | 44.60 | 26.83 | 20.16 | 30.53 |
|                  | RTN   |       | 7.47 | 36.10 | 18.90 | 10.54 | 21.84 | 
|                  | wikitext2 | 0.1327 | 6.67 | 40.60 | 22.56 | 19.18 | 27.45 |
|                  | humaneval | 0.0877 | 6.85 | 43.60 | 21.34 | 15.39 | 26.79 |
|                  | gsm8k     | 0.0728 | 6.79 | 43.40 | 21.95 | 18.59 | 27.98 |
|                  | EBSS      | 0.0052 | 6.77 | 44.00 | 23.78 | 18.19 | 28.65 |


---

Table R2: Influence of different temperatures $\tau$ of different models on the final average accuracy on 8 zero-shot tasks.

|        τ         |  1.0  |  1.1  |  1.2  |  1.3  |  1.4  |  1.5  |
| :--------------: | :---: | :---: | :---: | :---: | :---: | :---: |
| DeepSeek-MoE-16B | 39.82 | 39.89 | 40.01 | 39.89 | 39.69 | 39.71 |
|   QwenMoE-14B    | 49.47 | 49.53 | 49.59 | 49.57 | 49.59 | 49.55 |
|   Mixtral-8x7B   | 55.54 | 55.54 | 55.58 | 55.49 | 55.51 | 55.44 |

---

Table R3: Influence of different width $w$ in EBSS of different models on the final average accuracy on 8 zero-shot tasks.

|       $w$        |   2   |   3   |   4   |   5   |   6   |  10   | 20    |
| :--------------: | :---: | :---: | :---: | :---: | :---: | :---: | ----- |
| DeepSeek-MoE-16B | 39.77 | 39.80 | 40.01 | 39.98 | 40.01 | 40.00 | 40.10 |
|   QwenMoE-14B    | 49.19 | 49.45 | 49.59 | 49.59 | 49.61 | 49.60 | 49.63 |
|   Mixtral-8x7B   | 55.12 | 55.54 | 55.58 | 59.56 | 59.60 | 59.60 | 59.64 |

---

Table R4：Cosine similarity between GateLayer in DeepSeek-MoE-16B and the original output after introducing the approximation in Equation 17 under different c values

|    x    |  0.1   |  0.2   |  0.3   |  0.4   |  0.5   |  0.6   |  0.7   |  0.8   |  0.9   |
| :-----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| Cos Sim | 0.9090 | 0.9425 | 0.9685 | 0.9874 | 0.9926 | 0.9894 | 0.9949 | 0.9980 | 0.9996 |

---

Table R5：Effect of Gate-Layer on 3 MoE LLMs before and after AGQ

|         Model         | AGQ for Gate-Layer | Wiki PPL | ACC Mean |
| :-------------------: | :----------------: | :------: | :------: |
|   Qwen-MoE-14B-Chat   |         ×          |   8.74   |  44.41   |
|                       |         √          |   8.65   |  44.95   |
| DeepSeek-MoE-16B-Chat |         ×          |   7.77   |  45.87   |
|                       |         √          |   7.70   |  46.20   |
|     MIXTRAL-8x7B      |         ×          |   4.12   |  55.24   |
|                       |         √          |   4.12   |  55.58   |
