---
layout: post
title: Least squares
---

## Introduction

What we want to do is to solve regression problem exploting linear model, i.e. in a certain sense find the best linear mapping $$\mathcal{X}\rightarrow\mathbb{R}$$, where $$\mathcal{X}$$ - *n*-dimensional space of the input feature vectors. 

More specifically given input vector $$x$$ and target $$y$$ we aim to find so called *weights* vector $$\theta$$: $$L(\theta^{T}x, y)\rightarrow min$$, where $$L(\hat{y}, y)$$ - loss function that has *quadratic* form in our case.

So two substances completely defines current topic in the scope of machine learning domain - *linear regression* and *quadratic loss function*.


## The problem

Consider training set represented by $$m \times n$$ matrix $$X$$, where $$m$$'s row - training sample consisted from $$n$$ features. Also let $$\boldsymbol{y}$$ be the $$m$$-dimension target vector. obtain some $$n$$-dimensional vector $$\theta$$ (so called *weights*) which 

...
