---
title: State Space Models
date: 2024-05-10 12:34:00 +0400 

categories: [study]
tags: [deep_learning, in_progress]     # TAG names should always be 

mathjax: true
toc: true
---

# Introduction 

Recently, structured state space sequence models (SSMs)
have emerged as a promising class of architectures for sequence modeling
These models can be interpreted as a combination of recurrent neural networks (RNNs) and convolutional neural networks (CNNs), with inspiration
from classical state space models (Kalman 1960).


#  State Space Models: A Continuous-time Latent State Model

> **Definition (SSM)**:   
> Let $~~~~ u(t) \in \mathbb{R}$ - input signal   
> $~~~~~~~~~~ x(t) \in \mathbb{R}^N$ - latent state    
> $~~~~~~~~~~ y(t) \in \mathbb{R}$ - output signal    
> Than the State Space Model with parameters $\textbf{A, B, C, D}$ is defined as:
> $$\begin{equation}
\begin{cases}
x^{'}(t) = \textbf{A}(t)x(t) + \textbf{B}(t)u(t) \\
y(t) = \textbf{C}(t)x(t)  + \textbf{D}(t)u(t)
\end{cases} ~~~
\end{equation}
$$

> **Remark**:    
> $~~~~$ We will assume that $\textbf{D}$ = 0 since $\textbf{D}u$ can be seen as a skip connection and is easy to compute 



> **Proposition(The Convolutional Representation)**    
> If we assume that  $\textbf{A, B, C, D}$ doesn't depend on time, than:
> $$ \begin{cases}
x^{'}(t) = \textbf{A}(t)x(t) + \textbf{B}(t)u(t) \\
y(t) = \textbf{C}(t)x(t)  + \textbf{D}(t)u(t) \\
x(0) = 0 \\
\textbf{D} = 0 
\end{cases} ~~~~~\underset{Time ~~ Invariant}{\iff} ~~~~~ 
\begin{equation}
\begin{cases}
K(t) = \textbf{C}e^{t\textbf{A}}\textbf{B} \\
y(t) = (\textbf{K} * u)(t) 
\end{cases}~~~
\end{equation}
$$

**Proof:**   
Let's assume that $\textbf{A, B, C, D}$ are time-invariant. Then

$$\frac{dx}{dt}(t) - \textbf{A}x(t) = \textbf{B}u(t)$$

Next, multiply both sides by  $e^{-\textbf{A}t}$

$$e^{-\textbf{A}t}\frac{dx}{dt}(t) - e^{-\textbf{A}t}\textbf{A}x(t) = e^{-\textbf{A}t}\textbf{B}u(t)$$

If we let $u(t) = x(t)$ and $v(t) = e^{-\textbf{A}t}$, then

$$
\frac{d}{dt}(u(t) \cdot v(t)) =  e^{-\textbf{A}t}\textbf{B}u(t) 
$$

Substituting $u$ and $v$: 

$$
\frac{d}{dt}(x(t) \cdot e^{-\textbf{A}t}) =  e^{-\textbf{A}t}\textbf{B}u(t) 
$$

Integrating over $(\tau,~ t)$:

$$
\int_{\tau}^{t} \frac{d}{dt}(x(\lambda) \cdot e^{-\textbf{A}\lambda})d\lambda =  \int_{\tau}^{t} e^{-\textbf{A}\lambda}\textbf{B}u(\lambda)d\lambda 
$$

$$x(t) \cdot e^{-\textbf{A}t} - x(\tau) \cdot e^{-\textbf{A}\tau} = \int_{\tau}^{t} e^{-\textbf{A}\lambda}\textbf{B}u(\lambda)d\lambda $$

Multiplying by $e^{\textbf{A}t}$:

$$x(t)  =  x(\tau) \cdot e^{\textbf{A}(t - \tau )} + \int_{\tau}^{t} e^{-\textbf{A}\lambda}\textbf{B}u(\lambda)d\lambda $$

Out of interest, note that

$$x(t) = \underbrace{x(\tau) \cdot e^{\textbf{A}(t - \tau)}}_{\text{complementary function}} + \underbrace{\int_{\tau}^t e^{\textbf{A}(t - \lambda)}\textbf{B}u(\lambda) \, \text{d}\lambda}_{\text{particular integral}}$$

Unfortunately, this expression for $x(t)$ is still ambiguous, as we have not yet specified a value for $\tau$. We do this using an initial condition. For example, suppose we know that $x(\tau)=x_0$ for $\tau=0$. Then, we can substitute $\tau=0$ into the equation above to get

$$
\begin{align}
x(t) &= x(0) \cdot e^{\textbf{A}t} + \int_{0}^t e^{\textbf{A}(t - \lambda)}\textbf{B}u(\lambda) \, \text{d}\lambda \\
x(t) &= x_0 \cdot e^{\textbf{A}t} + \int_{0}^t e^{\textbf{A}(t - \lambda)}\textbf{B}u(\lambda) \, \text{d}\lambda
\end{align}
$$

We now have an exact expression for $x(t)$. We can now substitute this expression into the observation equation in (1) to determine an expression for the output  $y(t)$:

$$
\begin{cases}
y(t) = \textbf{C}x(t)  + \textbf{D}u(t) \\ 
x(t) = x_0 \cdot e^{\textbf{A}t} + \int_{0}^t e^{\textbf{A}(t - \lambda)}\textbf{B}u(\lambda) \, \text{d}\lambda
\end{cases}
$$

$$ y(t) = \textbf{C} \left[x_0 \cdot e^{\textbf{A}t} + \int_{0}^t e^{\textbf{A}(t - \lambda)}\textbf{B}u(\lambda) \, \text{d}\lambda \right]  + \textbf{D}u(t) $$

It is now possible to express the output $y(t)$ as the convolution of two functions. To see this, suppose that $\textbf{D}=0$, such that the equation above simplifies to

$$ y(t) =  \textbf{C}x_0 \cdot e^{\textbf{A}t} + \int_{0}^t e^{\textbf{A}(t - \lambda)}\textbf{B}u(\lambda) \, \text{d}\lambda  $$
Next, suppose that $x_0=0$, such that:

$$ y(t) =  \textbf{C}\int_{0}^t e^{\textbf{A}(t - \lambda)}\textbf{B}u(\lambda) \, \text{d}\lambda  $$

if $u(\lambda) \neq 0$ when $t \in [0, t]$ and $u(\lambda) = 0$ otherwise, then: 
$$ y(t) =  \int_{0}^t \textbf{C}e^{\textbf{A}(t - \lambda)}\textbf{B}u(\lambda) \, \text{d}\lambda = (\textbf{C}  e^{\textbf{A}(t)} \textbf{B} ) * u(t) = K(t) * u(t)$$
 $~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\square$

> **Remark:**    
> The function $K(t)$ is called the impulse response which can also be defined as
the output of the system when the input $u(t) = \delta(t)$ is the impulse or Dirac delta function







- Efficiently Modeling Long Sequences with Structured State Spaces.
- “Improving the Gating Mechanism of Recurrent Neural Networks
- “On the Parameterization and Initialization of Diagonal State Space Models
- Combining Recurrent, Convolutional, and Continuous-time Models with the Linear State Space Layer
- “How to Train Your HIPPO: State
Space Models with Generalized Basis Projections
- “Diagonal State Spaces are as Effective as Structured State
Spaces
- What Makes Convolutional Models
Great on Long Sequence Modeling?
- Mega: Moving Average Equipped Gated Attention
- Resurrecting Recurrent Neural Networks for Long Sequences
- Simplified State Space Layers for Sequence
Modeling


