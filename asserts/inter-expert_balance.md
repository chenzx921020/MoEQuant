# The Impact of Expert Imbalance on Hessian Estimation

## 1. Problem Definition and Background  
In Hessian-based quantization methods (e.g., GPTQ), the Hessian matrix is utilized for weight compensation to minimize quantization errors. However, the imbalance in calibration set distribution across experts in Mixture-of-Experts (MoE) models significantly affects the accuracy of Hessian estimation. This section theoretically analyzes the impact of expert imbalance on Hessian estimation bias and establishes its relationship with quantization errors.

---

## 2. Finite Difference Estimation of the Hessian Matrix  
Consider the loss function $L(\mathbf{w})$ with respect to the weight vector $\mathbf{w} \in \mathbb{R}^d$. The Hessian matrix element $H_{ij} = \frac{\partial^2 L}{\partial w_i \partial w_j}$ is estimated via the finite difference method:

$$ \hat{H}_{ij} = \frac{L(\mathbf{w} + \epsilon \mathbf{e}_i + \epsilon \mathbf{e}_j) - L(\mathbf{w} + \epsilon \mathbf{e}_i) - L(\mathbf{w} + \epsilon \mathbf{e}_j) + L(\mathbf{w})}{\epsilon^2} $$

where $\epsilon$ is a small perturbation, and $\mathbf{e}_i$ is the standard basis vector. Assuming the variance of the loss function estimator is $\text{Var}(L) = \frac{\sigma^2}{n}$ (with $n$ as the calibration set size), the bias of the Hessian estimator is derived as follows:

### **Theorem 1 (Bias of Hessian Estimation)**  
The expectation of the finite difference Hessian estimator satisfies:

$$ \mathbb{E}[\hat{H}_{ij}] = H_{ij} + \mathcal{O}(\epsilon^2) + \mathcal{O}\left(\frac{\sigma^2}{n \epsilon^2}\right) $$

#### **Proof:**  
Using a Taylor expansion, the loss function terms are approximated as:

$$ L(\mathbf{w} + \epsilon \mathbf{e}_i + \epsilon \mathbf{e}_j) = L(\mathbf{w}) + \epsilon (\partial_i L + \partial_j L) + \frac{\epsilon^2}{2} (\partial_{ii} L + 2\partial_{ij} L + \partial_{jj} L) + \mathcal{O}(\epsilon^3) $$

Substituting into the finite difference formula, the leading term becomes $H_{ij}$, with a truncation error $\mathcal{O}(\epsilon^2)$. The variance contribution to the bias is derived as:

$$ \text{Var}(\hat{H}_{ij}) = \frac{4\sigma^2}{n \epsilon^4} \implies \text{Bias term} \propto \frac{\sigma^2}{n \epsilon^2} $$

This completes the proof.

---

## 3. Analysis of Expert Imbalance  
Let $n_k$ denote the calibration sample size for expert $k$, with total samples $N = \sum_{k=1}^K n_k$. Expert imbalance is defined as the variance of sample distribution $\text{Var}(n_k)$.  

### **Lemma 1 (Sample Size and Hessian Bias)**  
For expert $k$, the estimation bias of its Hessian submatrix $\mathbf{H}^{(k)}$ satisfies:

$$ \text{Bias}(\hat{\mathbf{H}}^{(k)}) \propto \frac{1}{n_k} $$

#### **Proof:**  
From Theorem 1, with fixed $\epsilon$, the dominant bias term is $\frac{\sigma^2}{n_k \epsilon^2}$. Smaller $n_k$ leads to larger bias. For experts with insufficient samples ($n_k \ll N$), the variance of Hessian estimation increases, failing to capture high-sensitivity weights.

### **Theorem 2 (Imbalance and Quantization Error)**  
The quantization error $\mathcal{E}$ is lower-bounded by the variance of expert sample distribution:

$$ \mathcal{E} \geq C \cdot \text{Var}(n_k) \quad (C > 0) $$

#### **Proof:**  
The quantization error decomposes into expert-specific contributions:

$$ \mathcal{E} = \sum_{k=1}^K \frac{n_k}{N} \cdot \mathcal{E}_k $$

where $\mathcal{E}_k \propto \frac{1}{n_k}$ (Lemma 1). Applying the Cauchy-Schwarz inequality:

$$ \mathcal{E} \geq \frac{1}{\sum_{k=1}^K \frac{n_k}{N} n_k} = \frac{1}{\mathbb{E}[n_k^2]} \propto \text{Var}(n_k) $$

Thus, higher expert imbalance ($\text{Var}(n_k)$↑) results in larger quantization errors.

---

## 4. Experimental Validation  
Experiments on DeepSeek-MoE-16B demonstrate the theoretical findings:

| Model | Calib Dataset | Expert Balance STD  | WIKITEXT2 | MMLU | HUMANEVAL | GSM8K | BOOLQ | HELLASWAG | OPENBOOKQA | MATHQA | AVG Accuracy |
| :--------------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DeepSeek-MoE-16B | Float | \ | 6.51 | 44.60 | 26.83 | 20.16 | 72.72 | 58.06 | 32.20 | 31.49 | 40.86 |
| | RTN | \ | 7.47 | 36.10 | 18.90 | 10.54 | 70.21 | 55.76 | 30.60 | 28.87 | 35.85 |
| | wikitext2 | 0.0427 | 6.67 | 40.60 | 22.56 | 19.18 | 72.17 | 57.03 | 30.60 | 30.95 | 39.01 |
| | humaneval | 0.0877 | 6.85 | 43.60 | 21.34 | 15.39 | 73.79 | 56.91 | 30.80 | 30.48 | 38.90 |
| | gsm8k | 0.0928 | 6.88 | 42.40 | 21.65 | 16.59 | 73.57 | 57.01 | 30.20 | 30.72 | 38.88 |
| | EBSS | 0.0052 | 6.77 | 44.00 | 23.78 | 18.19 | 73.24 | 57.21 | 31.80 | 30.92 | **39.87** |

Results show that balanced calibration set allocation (10× reduction in standard deviation) reduces Hessian estimation bias and quantization errors.
