---
title: State Space Models
date: YYYY-MM-DD HH:MM:SS +/-TTTT
categories: [study]
tags: [deep_learning]     # TAG names should always be 
math: true
---

The challenge is to model long sequential data. The possible approaches are:
- RNN
- CNN 
- Neural ODE 

## Neural ODE 
Neural Rough Differential Equations for Long Time Series [1].




> **Definition (Controlled Differential Equations)**:   
> Let $a, b  \in \mathbb{R}$ with $a <b$   
> $\xi \in \mathbb{R}^{w}, w \in \mathbb{N}$   
> $X: [a, b] \rightarrow \mathbb{R}^v, v \in \mathbb{N}$ - a differentiable function   
> $f: \mathbb{R}^w \rightarrow  \mathbb{R}^{w \times v}$ - a continuous function  
> Then $Z: [a, b] \rightarrow \mathbb{R}^w$ is defined as the unique solution to the *controlled differential equation*:   
> $Z_a = \xi$, $Z_t = Z_a  + \int_{a}^{t} f(Z_s) \dot{X_s}ds$ for $t \in (a, b]$

Using $\dot{X_s}$ causes the solution to depend continuously on the evolution of $X$. We say that
the solution is *driven by the control X*.


> **Definition (Neural Controlled Differential Equation)**:   
> $x_i \in \mathbb{R}^{v-1}, v \in \mathbb{N}$, $t_i \in \mathbb{R}$   
> x = $((t_0, x_0), (t_1, x_1), \dots, (t_n, x_n))$, and $t_0 < \dots < t_n$   
> Let $X: [t_0, t_1] \rightarrow \mathbb{R}^v$ be the interpolation of data such that $X_{t_i} = (t_i, x_i)$   
> Let
 


## How to model sequential data using Neural ODE? 













# References
[1] Morrill, James, et al. "Neural rough differential equations for long time series." International Conference on Machine Learning. PMLR, 2021.
