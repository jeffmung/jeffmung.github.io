---
layout: post
title: "[논문 리뷰] DiffWave: A Versatile Diffusion Model for Audio Synthesis"
image: https://i.ibb.co/mGPWBkq/thumbnail.png
date: 2024-02-20
tags: 
categories: Paper-Review
use_math: true
---

<br><br>

## 논문 개요

DiffWave는 오디오 합성 분야에 diffusion을 적용한 가장 초기 모델들 중 하나입니다. WaveNet [(Aaron van den Oord et al., 2016)](http://arxiv.org/abs/1609.03499) 구조와 diffusion 모델을 혼합한 방법으로 기존 연구되었던 모델들에 비해 좋은 성능을 보여주면서 작은 모델 크기와 빠른 속도라는 이점을 가지고 있습니다.

기존 WaveNet이나 diffusion 모델과 차별화되는 특징으로는 bidirectional dilated convolution을 사용한다는 것과 생성 시에는 빠른 샘플링을 위해 분산을 적절하게 스케쥴링하여 학습에 사용한 diffusion 스텝보다 훨씬 더 적은 스텝으로 역방향 과정(reverse process)을 진행하는 알고리즘을 사용한다는 것이 있습니다.

이 연구는 Baidu와 NVIDIA에서 이루어졌으며 2021년 ICLR에 발표되었습니다.

<br><br>

## Diffusion Probabilistic Models

분자가 서서히 확산하여 완전한 무질서 상태가 되는 것과 같이 데이터를 이루고 있는 구성 요소들이 조금씩 노이즈를 더해가며 움직이면서 최종적으로 화이트 노이즈가 되는 과정을 생각해볼 수 있습니다. 이 과정의 확률 분포를 학습하고 반대 방향으로도 되돌릴 수 있다면 화이트 노이즈를 기존의 데이터와 비슷하게 변화시킬 수도 있을 것입니다. 이러한 아이디어로 개발된 생성 모델이 diffusion 모델입니다.

\(\small L\)차원의 데이터 \(\small x_0 \in \mathbb{R}^L\)에 대해서 데이터 분포를 $\small q_{\text{data}}(x_0)$라 하고 $\small x_t \in \mathbb{R}^L$을 diffusion 스텝 $\small t = 0, 1, \ldots, T$에 대한 확률 변수라고 정의하겠습니다. 총 $\small T$ 스텝의 diffusion 모델은 정방향 확산 과정(diffusion process)와 역방향 과정(reverse process)으로 이루어져 있습니다. 아래 그림은 두 과정을 나타냅니다.

<p align="center">
    <img src="https://i.ibb.co/PrG18xd/diffusion.png" alt="diffusion" border="0">
</p>

확산 과정은 다음 식과 같이 데이터 $\small x_0$로부터 잠재 변수(latent variable) $\small x_T$로 이어지는 고정된 Markov 체인으로 정의됩니다.

<br>
\begin{align}
q(x_1, \cdots, x_T \vert x_0) = \prod_{t=1}^{T} q(x_t \vert x_{t-1})
\end{align}
<br>

이때 각각의 $\small q(x_t \vert x_{t-1})$은 작은 값의 상수 $\small \beta_t$에 대해서 $\small \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)$로 고정됩니다. 즉, $\small q(x_t \vert x_{t-1})$은 $\small x_{t-1}$의 분포에 작은 가우시안 노이즈를 더하는 것과 같습니다. $\small T$가 충분히 크다면 전체 과정은 분산의 스케쥴 $\small \beta_1, \cdots, \beta_T$에 따라 데이터 $\small x_0$를 점진적으로 잠재 변수 $\small x_T$로 변화시키고 이 잠재 변수의 분포는 등방성(isotropic) 가우시안 분포가 됩니다.

역방향 과정은 $\small \theta$에 의해 매개변수화(parameterize)된 $\small x_T$로부터 $\small x_0$로 이어지는 Markov 체인으로 정의됩니다.

<br>
$$
\begin{align}
p_{\theta} (x_0, \cdots, x_{T-1} \vert x_T) = \prod_{t=1}^{T} p_{\theta}(x_{t-1} \vert x_t)
\end{align}
$$
<br>

이때 $\small p_{\text{latent}}(x_t) = \mathcal{N}(0, I)$는 등방성 가우시안 분포이고 전이 확률 분포 $\small p_{\theta}(x_{t-1} \vert x_t)$는 공유되는 매개변수 $\small \theta$에 의해 $\small \mathcal{N}(x_{t-1}; \mu_{\theta}(x_t, t), \sigma_{\theta}(x_t, t)^2 I)$로 매개변수화됩니다. 즉, $\small \mu_{\theta}$와 $\small \sigma_{\theta}$는 두 개의 입력 $\small x_t \in \mathbb{R}^N$와 $\small t \in \mathbb{N}$를 받고, $\small \mu_{\theta}$는 평균에 해당하는 $\small L$ 차원의 벡터를, $\small \sigma_{\theta}$는 표준편차에 해당하는 실수를 출력합니다. $\small p_{\theta}(x_{t-1} \vert x_t)$의 목표는 확산 과정 중에 추가된 노이즈를 제거하는(denoise) 것입니다.

### Sampling

모델이 생성하는 과정, 또는 샘플링 과정은, 먼저 $\small x_T \sim \mathcal{N}(0, I)$를 샘플링하고 $\small t = T-1, \cdots, 1$에 대해서 차례로 $\small x_{t-1} \sim p_{\theta} (x_{t-1} \vert x_t)$를 샘플링합니다. 그러면 최종 출력 $\small x_0$가 샘플링된, 혹은 생성된 데이터가 됩니다.

### Training

학습의 손실 함수를 얻는 전체 과정의 전개와 증명은 이 [블로그 포스트](https://lilianweng.github.io/posts/2021-07-11-diffusion-models)에 잘 정리되어 있으니 참고하는 것을 추천합니다. OpenAI에서 일하는 Lilian Weng의 블로그인데 이외에도 공부에 도움이 되는 좋은 내용들이 많이 있습니다. 여기서는 너무 세부적인 수식들은 생략하고 설명하겠습니다. 

모델의 목표는 우도(likelihood) $\small p_{\theta} (x_0)$를 최대화하는 것인데 이것은 추정하기 어려운(intractable) 확률분포입니다. 하지만 이러한 형태의 식은 변분 추론(variational inference)에 의해 다음과 같이 ELBO로 전개되고 모델은 ELBO를 최대화하는 것으로 학습될 수 있습니다.

<br>
\begin{align}
    \mathbb{E}_{q_{\text{data}} (x_0)} \log p_{\theta(x_0)} &= \mathbb{E}_{q_{\text{data}}(x_0)} \log \int p_{\theta} (x_{0:T-1} \vert x_T) \cdot p_{\text{latent}} (x_T) d x_{1:T} 
\end{align}
<br>
