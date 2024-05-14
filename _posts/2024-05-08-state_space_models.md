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
> Let:   
> $~~~~~~~~ u(t) \in \mathbb{R}$ - input signal   
> $~~~~~~~~ x(t) \in \mathbb{R}^N$ - latent state    
> $~~~~~~~~ y(t) \in \mathbb{R}$ - output signal    
> Than the State Space Model with parameters $\textit{\textbf{A, B, C, D}}$ is defined as:
>     
> $$\begin{equation}
\begin{cases}
x^{'}(t) = \textit{\textbf{A}}(t)x(t) + \textit{\textbf{B}}(t)u(t) \\
y(t) = \textit{\textbf{C}}(t)x(t)  + \textit{\textbf{D}}(t)u(t)
\end{cases} ~~~
\end{equation}
$$

> **Remark**:    
> $~~~~$ We will assume that $\textit{\textbf{D}}$ = 0 since $\textit{\textbf{D}}u$ can be seen as a skip connection and is easy to compute 


## The Convolutional Representation

> **Lemma 1**    
>  If we assume that  $\textbf{A, B, C}$ doesn't depend on time, than $x(t)$ has a form of 
> $$ x(t) = x(0) \cdot e^{\textit{\textbf{A}}t} + \int_{0}^t e^{\textit{\textbf{A}}(t - \lambda)}\textit{\textbf{B}}u(\lambda) \, \text{d}\lambda 
$$

<details>
  <summary><a  href="https://dsp.stackexchange.com/questions/23988/why-is-the-output-of-an-lti-system-expressed-as-the-convolution-of-the-input-wit  
">Proof</a></summary>


Let's assume that $\textbf{A, B, C}$ are time-invariant. Then

$$\frac{dx}{dt}(t) - \textit{\textbf{A}}x(t) = \textit{\textbf{B}}u(t)$$

Next, multiply both sides by  $e^{-\textit{\textbf{A}}t}$

$$e^{-\textit{\textbf{A}}t}\frac{dx}{dt}(t) - e^{-\textit{\textbf{A}}t}\textit{\textbf{A}}x(t) = e^{-\textit{\textbf{A}}t}\textit{\textbf{B}}u(t)$$

If we let $p(t) = x(t)$ and $v(t) = e^{-\textit{\textbf{A}}t}$, then

$$
\frac{d}{dt}(p(t) \cdot v(t)) =  e^{-\textit{\textbf{A}}t}\textit{\textbf{B}}u(t) 
$$

Substituting $p$ and $v$: 

$$
\frac{d}{dt}(x(t) \cdot e^{-\textit{\textbf{A}}t}) =  e^{-\textit{\textbf{A}}t}\textit{\textbf{B}}u(t) 
$$

Integrating over $(\tau,~ t)$:

$$
\int_{\tau}^{t} \frac{d}{dt}(x(\lambda) \cdot e^{-\textit{\textbf{A}}\lambda})d\lambda =  \int_{\tau}^{t} e^{-\textit{\textbf{A}}\lambda}\textit{\textbf{B}}u(\lambda)d\lambda 
$$

$$x(t) \cdot e^{-\textit{\textbf{A}}t} - x(\tau) \cdot e^{-\textit{\textbf{A}}\tau} = \int_{\tau}^{t} e^{-\textit{\textbf{A}}\lambda}\textit{\textbf{B}}u(\lambda)d\lambda $$

Multiplying by $e^{\textit{\textbf{A}}t}$:

$$x(t)  =  x(\tau) \cdot e^{\textit{\textbf{A}}(t - \tau )} + \int_{\tau}^{t} e^{\textit{\textbf{A}}(t - \lambda)}\textit{\textbf{B}}u(\lambda)d\lambda $$

Out of interest, note that

$$x(t) = \underbrace{x(\tau) \cdot e^{\textit{\textbf{A}}(t - \tau)}}_{\text{complementary function}} + \underbrace{\int_{\tau}^t e^{\textit{\textbf{A}}(t - \lambda)}\textit{\textbf{B}}u(\lambda) \, \text{d}\lambda}_{\text{particular integral}}$$

Unfortunately, this expression for $x(t)$ is still ambiguous, as we have not yet specified a value for $\tau$. We do this using an initial condition. For example, suppose we know that $x(\tau)=x_0$ for $\tau=0$. Then, we can substitute $\tau=0$ into the equation above to get

$$x(t) = x(0) \cdot e^{\textit{\textbf{A}}t} + \int_{0}^t e^{\textit{\textbf{A}}(t - \lambda)}\textit{\textbf{B}}u(\lambda) \, \text{d}\lambda
$$
 $~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\square$

 </details>


> **Proposition 1**    
> If we assume that  $\textbf{A, B, C}$ doesn't depend on time and $x(0) = 0$, than:
>    
> $$ \begin{cases}
x^{'}(t) = \textit{\textbf{A}}x(t) + \textit{\textbf{B}}u(t) \\
y(t) = \textit{\textbf{C}}x(t)  
\end{cases} ~~~~~\underset{Time ~~ Invariant}{\iff} ~~~~~ 
\begin{equation}
\begin{cases}
K(t) = \textit{\textbf{C}}e^{t\textit{\textbf{A}}}\textit{\textbf{B}} \\
y(t) = (K * u)(t) 
\end{cases}~~~
\end{equation}
$$

<details>
  <summary><a  href="https://dsp.stackexchange.com/questions/23988/why-is-the-output-of-an-lti-system-expressed-as-the-convolution-of-the-input-wit  
">Proof</a></summary>
Using Lemma 1: 

$$
\begin{align}
x(t) &= x(0) \cdot e^{\textit{\textbf{A}}t} + \int_{0}^t e^{\textit{\textbf{A}}(t - \lambda)}\textit{\textbf{B}}u(\lambda) \, \text{d}\lambda \\
x(t) &= x_0 \cdot e^{\textit{\textbf{A}}t} + \int_{0}^t e^{\textit{\textbf{A}}(t - \lambda)}\textit{\textbf{B}}u(\lambda) \, \text{d}\lambda
\end{align}
$$

We now have an exact expression for $x(t)$. We can now substitute this expression into the observation equation in (1) to determine an expression for the output  $y(t)$:

$$
\begin{cases}
y(t) = \textit{\textbf{C}}x(t)   \\ 
x(t) = x_0 \cdot e^{\textit{\textbf{A}}t} + \int_{0}^t e^{\textit{\textbf{A}}(t - \lambda)}\textit{\textbf{B}}u(\lambda) \, \text{d}\lambda
\end{cases}
$$

$$ y(t) = \textit{\textbf{C}} \left[x_0 \cdot e^{\textit{\textbf{A}}t} + \int_{0}^t e^{\textit{\textbf{A}}(t - \lambda)}\textit{\textbf{B}}u(\lambda) \, \text{d}\lambda \right]  $$

It is now possible to express the output $y(t)$ as the convolution of two functions:

$$ y(t) =  \textit{\textbf{C}}x_0 \cdot e^{\textit{\textbf{A}}t} +  \textit{\textbf{C}}\int_{0}^t e^{\textit{\textbf{A}}(t - \lambda)}\textit{\textbf{B}}u(\lambda) \, \text{d}\lambda  $$

Next, since we assumed that $x_0=0$:

$$ y(t) =  \textit{\textbf{C}}\int_{0}^t e^{\textit{\textbf{A}}(t - \lambda)}\textit{\textbf{B}}u(\lambda) \, \text{d}\lambda  $$

if $u(\lambda) \neq 0$ when $\lambda \in [0, t]$ and $u(\lambda) = 0$ otherwise, then: 

$$ y(t) =  \int_{0}^t \textit{\textbf{C}}e^{\textit{\textbf{A}}(t - \lambda)}\textit{\textbf{B}}u(\lambda) \, \text{d}\lambda = (\textit{\textbf{C}}  e^{\textit{\textbf{A}}(t)} \textit{\textbf{B}} ) * u(t) = K(t) * u(t)$$

 $~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\square$
 </details>




<!-- -------------------------------------------------- -->
 

The function $K(t)$ is called the impulse response which can also be defined as
the output of the system when the input $u(t) = \delta(t)$ is the impulse or Dirac delta function

> **Definition  (TSSM)**:    
> $~~~~$ Models of type (2) are called time-invariant state space models.


These are particularly important because the equivalence
to a convolution makes TSSMs parallelizable and very fast to compute.

> **Definition  (TSSM)**:    
> $~~~~$ Given a TSSM $(\textit{\textbf{A, B}})$, $e^{t\textit{\textbf{A}}}\textit{\textbf{B}}$ is a vector of $N$ functions which we call the $\textit{\textbf{SSM basis}}$. The individual basis functions are denoted as  $K_n(t) = \textbf{e}^{\intercal}_{n}e^{t\textit{\textbf{A}}}\textit{\textbf{B}}$, which satisfty 
> $$ x_n(t) = (u * K_n)(t) = \int_{-\infty}^{t} K_n(t - s)u(s)ds$$
> Here $\textbf{e}^{\intercal} $ is the one-hot basis vector 

This definition is motivated by noting that the SSM convolutional kernel is a linear combination of the SSM basis controlled by the vector of coefficients $\textit{\textbf{C}}$
$$K(t) = \sum_{n=0}^{N-1} \textit{\textbf{C}}_n K_n(t)$$

## **Discrete-time SSM: The Recurrent Representatioт**
To be applied on a discrete input sequence $(u_0, u_1, \dots)$ instead of continuous function $u(t)$, (1) must be discretized by a step size $\Delta$ that represents the resolution of the input. Conceptually, the inputs $u_k$ can be viewed as sampling an implicit underlying continuous signal $u(t)$, where $u_k = u(k\Delta)$.

> **Proposition**   
> The SSM discritezed using bilinear method has a form of 
> $$\begin{equation}
\begin{cases}
x_{k+1} = \overline{ \textit{\textbf{A}}} x_{k} +  \overline{\textit{\textbf{B}}} u_{k} \\
y_{k+1} =  \overline{\textit{\textbf{C}}} x_{k+1}
\end{cases}, ~\text{where} ~~~~~~~~~~~~
\begin{cases}
\overline{ \textit{\textbf{A}}} = ( \textit{\textbf{I}} + \frac{\Delta}{2}\textit{\textbf{A}}) (  \textit{\textbf{I}} - \frac{\Delta}{2}\textit{\textbf{A}})^{-1}\\
\overline{\textit{\textbf{B}}} =   (  \textit{\textbf{I}} - \frac{\Delta}{2}\textit{\textbf{A}})^{-1}\Delta \textit{\textbf{B}} \\ 
\overline{\textit{\textbf{C}}}  = \textit{\textbf{C}}
\end{cases}
\end{equation}$$

<details>
  <summary><a  href="https://en.wikipedia.org/wiki/Discretization">Proof</a></summary>
  By Lemma 1 we have: 

  $$ x(t) = x(0) \cdot e^{\textit{\textbf{A}}t} + \int_{0}^t e^{\textit{\textbf{A}}(t - \lambda)}\textit{\textbf{B}}u(\lambda) \, \text{d}\lambda 
$$
Discretization:

$$x_k := x(k\Delta)$$

 $$ x_k= \orange{x(0) \cdot e^{\textit{\textbf{A}}k \Delta} + \int_{0}^{k\Delta} e^{\textit{\textbf{A}}(k\Delta - \lambda)}\textit{\textbf{B}}u(\lambda) \, \text{d}\lambda }
$$

 $$ x_{k+1} = x(0) \cdot e^{\textit{\textbf{A}}(k+1)\Delta} + \int_{0}^{(k+1)\Delta} e^{\textit{\textbf{A}}((k+1)\Delta - \lambda)}\textit{\textbf{B}}u(\lambda) \, \text{d}\lambda =
$$
$$  =  e^{\textit{\textbf{A}}\Delta} \left[ \orange{x(0) \cdot e^{\textit{\textbf{A}}k\Delta} + \int_{0}^{k\Delta} e^{\textit{\textbf{A}}(k\Delta - \lambda)}\textit{\textbf{B}}u(\lambda) \, \text{d}\lambda }\right]  + \int_{k\Delta}^{(k+1)\Delta} e^{\textit{\textbf{A}}((k + 1)\Delta - \lambda)}\textit{\textbf{B}}u(\lambda) \, \text{d}\lambda =
$$

Next, we change variables in the integral and assume that $u$ is constant during the integral, so $u(\lambda) = u_k$:

$$  \underbrace{=}_{v(\lambda) = (k + 1)\Delta - \lambda}  e^{\textit{\textbf{A}}\Delta} x_k  + \int_{v(k\Delta)}^{v((k+1)\Delta)} e^{\textit{\textbf{A}}v}\, (-\text{d}v) \textit{\textbf{B}}u_k  =
$$

$$ = e^{\textit{\textbf{A}}\Delta} x_k  - \left(\int_{\Delta}^{0} e^{\textit{\textbf{A}}v}\, \text{d}v \right) \textit{\textbf{B}}u_k  =
$$

$$= e^{\textit{\textbf{A}}\Delta} x_k  + \left(\int_{0}^{\Delta} e^{\textit{\textbf{A}}v}\, \text{d}v \right) \textit{\textbf{B}}u_k =
$$

$$
= e^{\textit{\textbf{A}}\Delta} x_k  + \textit{\textbf{A}}^{-1} \left(e^{\textit{\textbf{A}} \Delta } - I \right)\textit{\textbf{B}}u_k  
$$

Bilinear approximation:

$$e^{A\Delta} = \frac{e^{\frac{\Delta}{2} A}}{e^{-\frac{\Delta}{2} A}} \approx \frac{ I + \frac{\Delta}{2} A}{I -\frac{\Delta}{2} A} = \left(I + \frac{\Delta}{2} A\right)\left(I -\frac{\Delta}{2} A\right)^{-1}$$
So, indeed $\overline{ \textit{\textbf{A}}} = \left(I + \frac{\Delta}{2} A\right)\left(I -\frac{\Delta}{2} A\right)^{-1}$

 $$\overline{ \textit{\textbf{B}}} =  \textit{\textbf{A}}^{-1}( e^{A\Delta} - I) B = $$
 $$= \textit{\textbf{A}}^{-1}\left(\left(I + \frac{\Delta}{2} A\right)\left(I -\frac{\Delta}{2} A\right)^{-1} - I\right) B = $$

  $$= \textit{\textbf{A}}^{-1}\left(I -\frac{\Delta}{2} A\right)^{-1}\left(\left(I + \frac{\Delta}{2} A\right) - \left(I -\frac{\Delta}{2} A\right) \right) B = $$
  
  $$= \textit{\textbf{A}}^{-1}\left(I -\frac{\Delta}{2} A\right)^{-1}A \Delta  B = $$
  $$
   = \left(I -\frac{\Delta}{2} A\right)^{-1} \Delta  B  
  $$

$~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\square$

</details>
