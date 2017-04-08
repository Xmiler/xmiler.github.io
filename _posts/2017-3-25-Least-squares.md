---
layout: post
title: Least squares
---

What we want to do is to solve regression problem exploting linear model, i.e. in a certain sense find the best linear mapping $$\mathcal{X}\rightarrow\mathbb{R}$$, where $$\mathcal{X}$$ - *n*-dimensional space of the input feature vectors. 

More specifically given input vector $$x$$ and target $$y$$ we aim to find so called *weights* vector $$\theta$$: $$L(\theta^{T}x, y)\rightarrow\min$$, where $$L(\hat{y}, y)$$ - loss function that has *quadratic* form in our case.

So two substances completely defines current topic in the scope of machine learning domain - *linear regression* and *quadratic loss function*.


## The problem

As for any machine learning problem we are needed for training set which in our supervision case consist of *m* input vectors from $$\mathbb{R}^n$$ and *m* target variables from $$\mathbb{R}$$.

Let represent set of input vectors by $$m \times n$$ matrix $$X$$ and appropriate targets by *m*-dimension vector $$\mathbf{y}$$. Then the least squares problem takes the following form: $$L=\frac{1}{2}(X\theta-\mathbf{y})^{T}(X\theta-\mathbf{y})\xrightarrow[\theta]{}\min$$. Or equavalently: $$\frac{\partial L}{\partial\theta}=0$$

$$
L = \frac{1}{2}(\theta^{T}X^{T}X\theta-\theta^{T}X^{T}\mathbf{y}-\mathbf{y}^{T}X\theta+\mathbf{y}^{T}\mathbf{y})
$$

$$
\frac{\partial L}{\partial\theta} = X^{T}X\theta-\frac{1}{2}X^{T}\mathbf{y}-\frac{1}{2}\mathbf{y}X^{T}=0
$$

$$
X^{T}X\theta-X^{T}\mathbf{y}=0
$$

$$
X^{T}X\theta=X^{T}\mathbf{y}
$$

$$
\theta=(X^{T}X)^{-1}X^{T}\mathbf{y}
$$
