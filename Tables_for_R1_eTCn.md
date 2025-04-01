**Table R1:  Impact of Quantization Bitwidth on Relative Error and Perplexity for DeepSeek-MoE-16B on Wikitext2**

|Bitwidth|2|3|4|5|6|8|Fp16|
|---|---|---|---|---|---|---|---|
|**Mean Relative Error**|92.67%|38.01%|14.26%|7.13%|3.56%|0.90%|0.00%|
|**Perplexity** ↓|5e8|10.65|7.151|6.629|6.539|6.51|6.50|

---

**Table R2: Effect of Temperature Parameter (τ) on Average Accuracy Across 7 Tasks for Different MoE Models.**

|        τ         |  1.0  |  1.1  |  1.2  |  1.3  |  1.4  |  1.5  |
| :--------------: | :---: | :---: | :---: | :---: | :---: | :---: |
| DeepSeek-MoE-16B | 39.82 | 39.89 | **40.01** | 39.89 | 39.69 | 39.71 |
|   QwenMoE-14B    | 49.47 | 49.53 | **49.60** | 49.61 | 49.59 | 49.55 |
|   Mixtral-8x7B   | 55.54 | 55.56 | **55.58** | 55.58 | 55.49 | 55.50 |

---

**Table R3： Impact of Affinity Parameter (c) on Cosine Similarity Between Approximate and Original Outputs in DeepSeek-MoE-16B's GateLayer.**

|    c    |  0.1   |  0.2   |  0.3   |  0.4   |  0.5   |  0.6   |  0.7   |  0.8   |  0.9   |
| :-----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| Cos Sim | 0.9090 | 0.9425 | 0.9685 | 0.9874 | 0.9926 | 0.9894 | 0.9949 | 0.9980 | 0.9996 |

---

**Table R4：Performance Gains from Applying AGQ to the Gate-Layer of MoE Models on WikiPPL and Average Accuracy of 7 Tasks.**

|         Model         | AGQ for Gate-Layer | Wiki PPL | AVG Accuracy |
| :-------------------: | :----------------: | :------: | :------: |
|   Qwen-MoE-14B-Chat   |         ×          |   8.74   |  44.41   |
|                       |         √          |   8.65   |  **44.95**   |
| DeepSeek-MoE-16B-Chat |         ×          |   7.77   |  45.87   |
|                       |         √          |   7.70   |  **46.20**   |
|     MIXTRAL-8x7B      |         ×          |   4.12   |  55.24   |
|                       |         √          |   4.12   |  **55.58**   |

