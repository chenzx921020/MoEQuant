# The Impact of Expert Imbalance on Hessian Estimation

## 1. Problem Definition and Background  
There exists a relationship between expert balance and quantization performance. Taking the GPTQ quantization method as an example, it relies on the Hessian matrix to perform weight compensation. Since the optimization of the objective function ensures that the input samples are positively correlated with the Hessian matrix. The imbalance in calibration set distribution across experts in Mixture-of-Experts (MoE) models significantly affects the accuracy of Hessian estimation. This section theoretically analyzes the impact of expert imbalance on Hessian estimation bias and establishes its relationship with quantization errors.

---

## 2. Finite Difference Estimation of the Hessian Matrix  
Consider the loss function $L(\mathbf{w})$ with respect to the weight vector $\mathbf{w} \in \mathbb{R}^d$. The Hessian matrix element $H_{ij} = \frac{\partial^2 L}{\partial w_i \partial w_j}$ is estimated via the finite difference method:

$$ \hat{H}_{ij} = \frac{L(\mathbf{w} + \epsilon \mathbf{e}_i + \epsilon \mathbf{e}_j) - L(\mathbf{w} + \epsilon \mathbf{e}_i) - L(\mathbf{w} + \epsilon \mathbf{e}_j) + L(\mathbf{w})}{\epsilon^2} $$

where $\epsilon$ is a tiny perturbation, and $\mathbf{e}_i$ is the standard basis vector. 

In many finite - difference formulas for the Hessian (such as the common four-point or five-point difference formulas), the second-order and third-order terms often cancel each other out among several symmetric "positive and negative perturbations"; while the fourth-order term is usually the main source of the final error. If the expansion is only carried out to the third-order, the residual (error term) left by the fourth-order term in the final formula cannot be seen, and it is also impossible to determine whether the impact of this residue on the Hessian approximation is $O(\epsilon^2)$ or larger/smaller.
### Symmetric difference cancellation oddterm
Like:
$$L(w+e_i \epsilon+e_j\epsilon), L (w+e_i\epsilon-e_j\epsilon), L (w-e_i\epsilon+e_j\epsilon), L (w-e_i\epsilon-e_j\epsilon)$$

This kind of symmetric addition and subtraction causes the singular powers of the first and third degree to appear in
pairs in the difference and cancel each other out. To know whetther the cubic term is completely canceled, we need
to write it down first: to know what error the quadratic term has after thie difference, we must keep it to the fourth
order in the Taylor expansion to see how its coefficients appear in the final formula.
### Keep it to the fourth order to determine the final truncation error
In the Hessian difference formula, it is often concerned with how accurate the "main term" is, and whether the
residual error" is $O(\epsilon^2)$ or $O(\epsilon^3)$. Only by explicitly writing the terms fourth-order expansion can we see
whether they are completely offset in the difference, or only a constantmultiple is left, so as to determine the error
order of the final formula.

We perform a fourth-order Taylor expansion on the loss function:

$$L(w+\epsilon e_i​+\epsilon e_j​) \approx L(w)+\epsilon \nabla_i​L+\epsilon \nabla_j​L+\frac{\epsilon ^2}{2}​(H_{ii}​+2H_{ij}​+H_{jj}​)+\frac{\epsilon ^3}{6}​(\nabla_{iii​}L+3\nabla_{ijj}​L)+\frac{\epsilon ^4}{24}​(\nabla_{iiii}​L+6\nabla_{iijj}​L+\nabla_{jjjj}​L)$$
$$L(w+\epsilon e_i​) \approx L(w)+\epsilon \nabla_i​L+\frac{\epsilon ^2}{2}​H_{ii}​+\frac{\epsilon ^3}{6}\nabla_{iii}​L+\frac{\epsilon ^4}{24}​\nabla_{iiii}​L$$
$$L(w+\epsilon e_j​) ​\approx L(w)+\epsilon \nabla_j​L+\frac{\epsilon ^2}{2}​H_{jj}​+\frac{\epsilon ^3}{6}\nabla_{jjj}​L+\frac{\epsilon ^4}{24}\nabla_{jjjj}​L​$$

$$ \hat{H_{ij}} \approx H_{ij} + \frac{\epsilon^2}{12}(\nabla_{iiii}L+6\nabla_{iijj}L+\nabla_{jjjj}L)+O(\epsilon^4) $$

Assuming the variance of the loss function estimator is $\text{Var}(L) = \frac{\sigma^2}{n}$ (with $n$ as the calibration set size). The larger the number, the smaller the estimated variance of the loss function, that is, the estimation of the true expected loss is more stable. The expectation of the finite difference Hessian estimator satisfies:

$$ \mathbb{E}[\hat{H}_{ij}] = H_{ij} + \mathcal{O}(\epsilon^2) + \mathcal{O}\left(\frac{\sigma^2}{n \epsilon^2}\right) $$

Estimate the deviation of the expected value of the Hessian

$$ Bias(\hat{H_{ij}}) = E[\hat{H_{ij}}]-H_{ij} \approx O(\epsilon^2)+O(\frac{\sigma^2}{n\epsilon^2}) $$

It can be seen that the second term of the formula is inversely proportional to the sample size n. **The larger the n, the smaller the corresponding statistical deviation, and vice versa.**


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
