# Neural Tangent Kernel (NTK)
## Infinite-Width Neural Networks as Kernel Methods

**University of Hertfordshire | Machine Learning and Neural Networks | 2025**

---

## Overview

This tutorial covers the Neural Tangent Kernel: the remarkable result that infinitely wide neural networks are exactly equivalent to kernel regression. Topics include: NTK definition as a Gram matrix of Jacobians, the infinite-width convergence proof, exponential loss decay rate (λ_min · η), empirical NTK computation in PyTorch, and the feature learning vs lazy training distinction.

## Repository Contents

| File | Description |
|------|-------------|
| `ntk_tutorial.docx` | Full tutorial document |
| `ntk_tutorial.ipynb` | Jupyter notebook with full PyTorch implementation |
| `README.md` | This file |
| `LICENSE` | MIT licence |

## How to Run

```bash
pip install torch matplotlib numpy scipy
jupyter notebook ntk_tutorial.ipynb
```

## Figures

| Figure | Content |
|--------|---------|
| Figure 1 | NTK kernel regression vs wide/narrow NN + kernel matrix |
| Figure 2 | Convergence to NTK as width increases + training dynamics |
| Figure 3 | NTK eigenspectrum + generalisation learning curve |
| Figure 4 | Feature learning vs lazy training (weight change + NTK drift) |
| Figure 5 | NTK vs RBF vs polynomial kernel comparison |
| Figure 6 | NTK theory map + regime applicability table |

## References

1. Jacot et al. (2018) 'Neural Tangent Kernel'. https://arxiv.org/abs/1806.07572
2. Lee et al. (2019) 'Wide Neural Networks Evolve as Linear Models'. https://arxiv.org/abs/1902.06720
3. Du et al. (2019) 'Gradient Descent Finds Global Minima'. https://arxiv.org/abs/1811.03804
4. Yang (2020) 'Tensor Programs II: NTK for Any Architecture'. https://arxiv.org/abs/2006.14548
5. Arora et al. (2019) 'On Exact Computation with the NTK'. https://arxiv.org/abs/1904.11955

## Licence

MIT
