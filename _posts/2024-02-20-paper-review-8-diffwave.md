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

이 연구는 Baidu와 NVIDIA에서 이루어졌으며 2021년 ICLR에 발표되었습니다. 새로운 아이디어 제안보다는 오디오 분야에 diffusion을 처음 도입했다는 것이 큰 의미를 갖는 논문이기 때문에 이 포스트에서는 diffusion 모델에 대해서도 충분히 자세하게 다루려고 합니다.

<br><br>

## Diffusion Probabilistic Models

분자가 서서히 확산하여 완전한 무질서 상태가 되는 것과 같이 데이터를 이루고 있는 구성 요소들이 조금씩 노이즈를 더해가며 움직이면서 최종적으로 화이트 노이즈가 되는 과정을 생각해볼 수 있습니다. 이 과정의 확률 분포를 학습하고 반대 방향으로도 되돌릴 수 있다면 화이트 노이즈를 기존의 데이터와 비슷하게 변화시킬 수도 있을 것입니다. 이러한 아이디어로 개발된 생성 모델이 diffusion 모델입니다.

$\small L$차원의 데이터 $\small x_0 \in \mathbb{R}^L$에 대해서 데이터 분포를 $\small q_{\text{data}}(x_0)$라 하고 $\small x_t \in \mathbb{R}^L$을 diffusion 스텝 $\small t = 0, 1, \ldots, T$에 대한 확률 변수라고 정의하겠습니다. 총 $\small T$ 스텝의 diffusion 모델은 정방향 확산 과정(diffusion process)와 역방향 과정(reverse process)으로 이루어져 있습니다. 아래 그림은 두 과정을 나타냅니다.

<p align="center">
    <img src="https://i.ibb.co/PrG18xd/diffusion.png" alt="diffusion" border="0">
</p>

확산 과정은 다음 식과 같이 데이터 $\small x_0$로부터 잠재 변수(latent variable) $\small x_T$로 이어지는 고정된 Markov 체인으로 정의됩니다.

<br>

\begin{equation}
q(x_1, \cdots, x_T \vert x_0) = \prod_{t=1}^{T} q(x_t \vert x_{t-1})
\end{equation}

<br>

이때 각각의 $\small q(x_t \vert x_{t-1})$은 작은 값의 상수 $\small \beta_t$에 대해서 $\small \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)$로 고정됩니다. 즉, $\small q(x_t \vert x_{t-1})$은 $\small x_{t-1}$의 분포에 작은 가우시안 노이즈를 더하는 것과 같습니다. $\small T$가 충분히 크다면 전체 과정은 분산의 스케쥴 $\small \beta_1, \cdots, \beta_T$에 따라 데이터 $\small x_0$를 점진적으로 잠재 변수 $\small x_T$로 변화시키고 이 잠재 변수의 분포는 등방성(isotropic) 가우시안 분포가 됩니다.

역방향 과정은 $\small \theta$에 의해 매개변수화(parameterize)된 $\small x_T$로부터 $\small x_0$로 이어지는 Markov 체인으로 정의됩니다.

<br>

\begin{equation}
p_{\theta} (x_0, \cdots, x_{T-1} \vert x_T) = \prod_{t=1}^{T} p_{\theta}(x_{t-1} \vert x_t)
\end{equation}

<br>

이때 $\small p_{\text{latent}}(x_t) = \mathcal{N}(0, I)$는 등방성 가우시안 분포이고 전이 확률 분포 $\small p_{\theta}(x_{t-1} \vert x_t)$는 공유되는 매개변수 $\small \theta$에 의해 $\small \mathcal{N}(x_{t-1}; \mu_{\theta}(x_t, t), \sigma_{\theta}(x_t, t)^2 I)$로 매개변수화됩니다. 즉, $\small \mu_{\theta}$와 $\small \sigma_{\theta}$는 두 개의 입력 $\small x_t \in \mathbb{R}^N$와 $\small t \in \mathbb{N}$를 받고, $\small \mu_{\theta}$는 평균에 해당하는 $\small L$ 차원의 벡터를, $\small \sigma_{\theta}$는 표준편차에 해당하는 실수를 출력합니다. $\small p_{\theta}(x_{t-1} \vert x_t)$의 목표는 확산 과정 중에 추가된 노이즈를 제거하는(denoise) 것입니다.

### Sampling

모델이 생성하는 과정, 또는 샘플링 과정은, 먼저 $\small x_T \sim \mathcal{N}(0, I)$를 샘플링하고 $\small t = T-1, \cdots, 1$에 대해서 차례로 $\small x_{t-1} \sim p_{\theta} (x_{t-1} \vert x_t)$를 샘플링합니다. 그러면 최종 출력 $\small x_0$가 샘플링된, 혹은 생성된 데이터가 됩니다.

### Training

Diffusion 모델 학습의 손실 함수를 얻는 전체 과정의 전개와 증명은 이 [블로그 포스트](https://lilianweng.github.io/posts/2021-07-11-diffusion-models)에 잘 정리되어 있으니 참고하는 것을 추천합니다. OpenAI에서 일하는 Lilian Weng의 블로그인데 이외에도 공부에 도움이 되는 좋은 내용들이 많이 있습니다. 여기서는 너무 세부적인 수식들은 생략하고 설명하겠습니다. 

모델의 목표는 우도(likelihood) $\small p_{\theta} (x_0)$를 최대화하는 것인데 이것은 추정하기 어려운(intractable) 확률분포입니다. 하지만 이러한 형태의 식은 변분 추론(variational inference)에 의해 다음과 같이 ELBO로 전개되고 모델은 ELBO를 최대화하는 것으로 학습될 수 있습니다.

<br>
\begin{align}
\mathbb{E}\_{q\_{\text{data}} (x\_0)} \log p\_{\theta}(x\_0)
&= \mathbb{E}\_{q\_{\text{data}}(x\_0)} \log \int p\_{\theta} (x\_0, \cdots, x\_{T-1} \vert x\_T) \cdot p\_{\text{latent}} (x\_T) d x\_{1:T} \newline
&= \mathbb{E}\_{q\_{\text{data}}(x\_0)} \log \left( \mathbb{E}\_{q(x\_{1:T} \vert x\_0)} \frac{p_{\theta} (x\_0, \cdots, x\_{T-1} \vert x_T) \cdot p_{\text{latent}(x\_T)}}{q(x\_1, \cdots, x\_T \vert x\_0)} \right) \\\\
&\geq \mathbb{E}\_{q(x\_0, \cdots, x\_T)} \log \frac{p_{\theta} (x\_0, \cdots, x\_{T-1} \vert x_T) \cdot p_{\text{latent}(x\_T)}}{q(x\_1, \cdots, x\_T \vert x\_0)} := \text{ELBO}
\end{align}
<br>

이 ELBO를 최대화하기 위해 최소화해야 하는 손실 함수는 더 전개되어 다음과 같이 KL 발산들의 조합으로 정리될 수 있습니다.

<br>
$$
\begin{align}
\- \text{ELBO}
&= \mathbb{E}\_{q(x\_0, \cdots, x\_T)} \log \frac{q(x\_1, \cdots, x\_T \vert x\_0)}{p_{\theta} (x\_0, \cdots, x\_{T-1} \vert x_T) \cdot p_{\text{latent}(x\_T)}} \\
&= \mathbb{E}\_q \left[ \underbrace{D\_\text{KL}(q(x\_T \vert x\_0) \parallel p\_\theta(x\_T))}\_{L\_T} + \sum\_{t=2}^T \underbrace{D\_\text{KL}(q(x\_{t-1} \vert x\_t, x\_0) \parallel p\_\theta(x\_{t-1} \vert x\_t))}\_{L\_{t-1}} \underbrace{\- \log p\_\theta(x\_0 \vert x\_1)}\_{L\_0} \right]
\end{align}
$$
<br>

여기서 $\small L\_{T} $는 상수이고 $\small L\_0 $는 무시할 수 있을 정도로 영향력이 작기 때문에 결국 $\small L\_{T-1} + \cdots + L\_1$을 최소화시키는 문제가 됩니다. 다시 상기해보자면 학습시키는 대상은 역방향 과정의 확률 분포 $\small p\_{\theta}(x\_{t-1} \vert x\_t) = \mathcal{N}(x\_{t-1}; \mu\_{\theta}(x\_t, t), \sigma\_{\theta}(x\_x, t)^2 I)$를 추정하는 신경망입니다. 그리고 $\small q(x\_{t-1} \vert x\_t, x\_0) = \mathcal{N}(x\_{t-1}; \tilde{\mu}\_t(x\_t, x\_0), \tilde{\beta}\_t I)$라고 기호 $\small \tilde{\mu}\_t$와 $\small \tilde{\beta}\_t$를 새로 정의하겠습니다. 가우시안 분포 $\small p = \mathcal{N}(\mu\_1, \sigma\_1^2)$와 $\small q = \mathcal{N}(\mu\_2, \sigma\_2^2)$의 KL 발산은 다음과 같습니다.

<br>
\begin{equation}
D\_{\text{KL}}(p \Vert q) = \log \frac{\sigma\_2}{\sigma\_1} + \frac{\sigma\_{1}^2 + (\mu\_1 - \mu\_2 )^2}{2\sigma\_2^2} - \frac{1}{2}
\end{equation}
<br>

학습을 더 간단하게 하기 위해서는 $\small p\_{\theta}(x\_{t-1} \vert x\_t)$의 분산을 $\small q(x\_{t-1} \vert x\_t, x\_0)$의 분산과 같은 상수 $\small \tilde{\beta}\_t$로 고정하고 평균 $\small \mu\_\theta$만 매개변수에 의해 학습되도록 근사할 수 있습니다. 학습되는 매개변수에 영향을 받는 항만 남기면 다음 식이 나옵니다.
 
<br>
\begin{equation}
L\_{t-1} = \mathbb{E}\_q \frac{\lVert \tilde{\mu}\_t(x\_t, x\_0) - \mu\_{\theta}(x\_t, t) \rVert\_2^2}{2\tilde{\beta}\_t}
\end{equation}
<br>

$\small q(x\_{t-1} \vert x\_t, x\_0)$의 평균과 분산을 구하기 위해 먼저 $\small q(x\_t \vert x\_{t-1}) = \mathcal{N}(x\_t; \sqrt{1-\beta\_t}x\_{t-1}, \beta\_t I)$부터 다시 유도를 시작해보겠습니다. $\small \alpha_t = 1 - \beta_t$이고 $\small \bar{\alpha}\_t = \prod\_{s=1}^t \alpha\_s$라고 정의합니다. 재매개변수화 트릭(reparameterization trick)을 사용하면 $\small x\_t$는 다음과 같이 얻어집니다.

<br>
$$
\begin{gather}
\begin{align}
x\_t
&= \sqrt{\alpha_t} x\_{t-1} + \sqrt{1 - \alpha\_t} \epsilon\_{t-1} \\
&= \sqrt{\alpha\_t \alpha\_{t-1}}x\_{t-2} + \sqrt{1 - \alpha\_t \alpha\_{t-1}} \bar{\epsilon}\_{t-2} \\
&= \cdots \\\
&= \sqrt{\bar{\alpha}\_t}x\_0 + \sqrt{1 - \bar{\alpha}\_t}\epsilon
\end{align}
\end{gather}
$$
<br>

이때 $\small \epsilon\_{t-1}, \epsilon\_{t-2},\cdots \sim \mathcal{N}(0, I)$이고 $\small \bar{\epsilon}\_{t-2}$는 가우시안 분포의 성질에 따라 $\small \epsilon\_{t-1}$과 $\small \epsilon\_{t-2}$를 하나로 합친 것입니다. $\small \epsilon$은 그렇게 계속 합쳐진 $\small \bar{\epsilon}\_0$에 해당합니다. 결과적으로 $\small q(x\_t \vert x\_{t-1}) = \mathcal{N}(x\_t; \sqrt{\bar{\alpha}\_t}x\_0, (1-\bar{\alpha}\_t) I)$가 됩니다.

그리고 $\small q(x\_{t-1} \vert x\_t, x\_0)$를 Bayes 룰과 가우시안 분포의 정의를 이용하여 전개하면 다음과 같이 됩니다. 세부 과정은 생략합니다.

<br>
$$
\begin{align}
q(x\_{t-1} \vert x\_t, x\_0)
&= q(x\_t \vert x\_{t-1}, x\_0) \frac{ q(x\_{t-1} \vert x\_0) }{ q(x\_t \vert x\_0) } \\
&\propto \exp\Big( -\frac{1}{2} \big( (\frac{\alpha\_t}{\beta\_t} + \frac{1}{1 - \bar{\alpha}\_{t-1}}) x\_{t-1}^2 - (\frac{2\sqrt{\alpha\_t}}{\beta\_t}x\_t + \frac{2\sqrt{\bar{\alpha}\_{t-1}}}{1 - \bar{\alpha}\_{t-1}} x\_0) x\_{t-1} + C(x\_t, x\_0) \big) \Big)
\end{align}
$$
<br>

$\small C(x\_t, x\_0)$는 $\small x\_{t-1}$을 포함하지 않는 상수항이고 가우시안 분포의 정의에 의해 여기서 평균과 분산을 구할 수 있습니다.

<br>
$$
\begin{eqnarray}
\tilde{\beta}\_t
&= 1 / ( \frac{\sqrt{\alpha\_t}}{\beta\_t}x\_t + \frac{\sqrt{\bar{\alpha}\_{t-1}}}{1 - \bar{\alpha}\_{t-1}} x\_0 ) \\
&= \frac{1 - \bar{\alpha}\_{t-1}}{1 - \bar{\alpha}\_t} \cdot \beta\_t \\
\\
\tilde{\mu}\_t (x\_t, x\_0)
&= ( \frac{\sqrt{\alpha\_t}}{\beta\_t}x\_t + \frac{\sqrt{\bar{\alpha}\_{t-1}}}{1 - \bar{\alpha}\_{t-1}} x\_0 ) / ( \frac{\alpha\_t}{\beta\_t} + \frac{1}{1 - \bar{\alpha}\_{t-1}} ) \\
&= \frac{1}{\sqrt{\alpha\_t}} \Big( x\_t - \frac{1 - \alpha\_t}{\sqrt{1 - \bar{\alpha}\_t}} \epsilon \Big)
\end{eqnarray}
$$
<br>

중간 과정으로 위해서 구한 $\small x\_t = \sqrt{\bar{\alpha}\_t}x\_0 + \sqrt{1 - \bar{\alpha}\_t}\epsilon$ 식에 의해 $\small x\_0$를 대입하고 정리한 과정이 생략되어 있습니다. 이제 $\small p\_{\theta}(x\_{t-1} \vert x\_t)$의 평균 $\small \mu\_{\theta}$를 다음과 같이 매개변수화 되도록 설정하면 손실 함수를 더 간단하게 만들 수 있습니다.

<br>
$$
\begin{align}
\mu\_{\theta} (x\_t, t) = \frac{1}{\sqrt{\alpha\_t}} \Big( x\_t - \frac{1 - \alpha\_t}{\sqrt{1 - \bar{\alpha}\_t}} \epsilon\_{\theta}(x\_t, t) \Big)
\end{align}
$$
<br>

$\small \epsilon\_{\theta}(x\_t, t) : \mathbb{R}^L \times \mathbb{N} \rightarrow \mathbb{R}^L$는 $\small x\_t$와 확산 스텝 $\small t$를 입력으로 받는 신경망입니다. 앞서 말했던 것과 같이 표준편차는 $\small \sigma\_{\theta}(x\_t, t) = \tilde{\beta}\_t^{\frac{1}{2}}$의 고정된 값으로 정의합니다. 손실 함수 $\small L\_{t-1}$을 정리해보겠습니다.

<br>
$$
\begin{align}
\small L\_{t-1}
&= \mathbb{E}\_{x\_0, \epsilon} \frac{1}{2 \tilde{\beta}\_t} \lVert \frac{1}{\sqrt{\alpha\_t}} \Big( x\_t - \frac{1 - \alpha\_t}{\sqrt{1 - \bar{\alpha}\_t}} \epsilon \Big) - \frac{1}{\sqrt{\alpha\_t}} \Big( x\_t - \frac{1 - \alpha\_t}{\sqrt{1 - \bar{\alpha}\_t}} \epsilon\_{\theta}(x\_t, t) \Big) \rVert\_2^2 \\
\\
&= \kappa\_t \mathbb{E}\_{x\_0, \epsilon} \lVert \epsilon - \epsilon\_{\theta} (\sqrt{\bar{\alpha}\_t}x\_0 + \sqrt{1 - \bar{\alpha}\_t}\epsilon, t) \rVert\_2^2
\end{align}
$$
<br>

$\small \kappa\_t=\frac{\beta\_t}{2\alpha\_t (1-\bar{\alpha}\_{t-1})}$는 계수를 정리한 것입니다. 이 계수도 $\small t$에 의해 달라지는 일종의 가중치이지만 실험적으로 다음과 같은 가중치 없는 손실 함수가 더 좋은 학습 성능을 보여준다는 것을 DDPM (Denoising Diffusion Probabilistic Models) [(Ho et al., 2020)](https://arxiv.org/abs/2006.11239) 논문에서 제안했습니다.

<br>
\begin{equation}
\min\_{\theta} L(\theta) = \mathbb{E}\_{t \sim [1,T], x\_0, \epsilon} \lVert \epsilon - \epsilon\_{\theta} (\sqrt{\bar{\alpha}\_t}x\_0 + \sqrt{1 - \bar{\alpha}\_t}\epsilon, t) \rVert\_2^2
\end{equation}
<br>

DiffWave도 이 손실 함수를 그대로 사용합니다. 학습과 샘플링 알고리즘은 아래에 요약되어 있습니다.

<p align="center">
    <img src="https://i.ibb.co/wdXn2r3/algorithm.png" alt="algorithm" border="0">
</p>

### Fast Sampling

알고리즘 1에 의해 모델을 학습한 뒤 $\small t=T$의 화이트 노이즈로부터 $\small t=0$의 사람의 목소리로 점차 디노이징을 해나가는 샘플링 과정을 들어보면 $\small t=0$ 근처의 스텝에서 디노이징이 가장 효과적으로 됩니다. 이 예시는 [데모 웹사이트](https://diffwave-demo.github.io/)의 Section IV에서 직접 들어볼 수 있습니다. 이러한 결과로부터 영감을 받아 DiffWave는 빠른 샘플링을 위하여 $\small T$ (e.g. 200) 스텝이 아닌 훨씬 적은 $\small T\_{\text{infer}}$ (e.g. 6) 스텝으로 디노이징을 하는 알고리즘을 사용합니다.

$\small T_{\text{infer}} \ll T$는 샘플링 시 역방향 과정의 스텝 수이고 $\small \\{ \eta\_t \\}\_{t=1}^{T\_{\text{infer}}}$는 사용자에 의해 정의된 분산 스케쥴로 학습 시의 분산 스케쥴 $\small \\{ \beta\_t \\}\_{t=1}^{T}$와는 독립적입니다. $\small \alpha_\{t}$와 $\small \tilde{\beta}\_t$에 등에 대응하는 변수들도 다음과 같이 정의해줍니다.

<br>
\begin{equation}
\gamma\_t = 1 - \eta\_t , \qquad \bar{\gamma}\_t = \prod\_{s=1}^t \gamma\_s , \qquad \tilde{\eta}\_t = \frac{1 - \bar{\gamma}\_{t-1}}{1 - \bar{\gamma}\_t} \eta\_t \quad \text{for} \quad t > 1 \quad \text{and} \quad \tilde{\eta}\_1 = \eta\_1
\end{equation}
<br>

샘플링 과정의 스텝 $\small s$에서 노이즈를 제거하기 위해 $\small t$를 선택하고 $\small \epsilon\_{\theta}(\cdot, t)$를 이용해야 합니다. 이를 위해 사용자가 정의한 분산 스케쥴과 학습 분산 스케쥴의 노이즈 레벨을 맞춰줘야 합니다. 이상적으로는 $\small \sqrt{\bar{\alpha}\_t} = \sqrt{\bar{\gamma}\_s}$가 되면 좋지만 일일이 이 값을 맞춰주기는 어렵습니다. 따라서 $\small \sqrt{\bar{\gamma}\_s}$가 학습 노이즈 레벨 $\small \sqrt{\bar{\alpha}\_{t+1}}$과 $\small \sqrt{\bar{\alpha}\_t}$ 사이의 값이면 그에 맞춰 $\small t$를 보간해줍니다. 보간된 확산 스텝 $\small t\_{s}^{\text{align}}$는 다음 식과 같이 얻어집니다.

<br>
\begin{equation}
t\_s^{\text{align}} = t + \frac{\sqrt{\bar{\alpha}\_t} - \sqrt{\bar{\gamma}\_s}}{\sqrt{\bar{\alpha}\_t} - \sqrt{\bar{\alpha}\_{t+1}}} \qquad \text{if} \quad \sqrt{\bar{\gamma}\_s} \in \[ \sqrt{\bar{\alpha}\_{t+1}}, \sqrt{\bar{\alpha}\_{t+1}} \]
\end{equation}
<br>

$\small t\_s^{\text{align}}$는 정수가 아닌 소수가 될 수 있습니다. 최종적으로 $\small \mu\_{\theta}$와 $\small \sigma\_{\theta}$는 다음과 같이 얻어집니다.

<br>
$$
\begin{align}
\mu\_{\theta}^{\text{fast}} (x\_s, s) &= \frac{1}{\sqrt{\gamma\_s}} \Big( x\_s - \frac{1 - \eta\_s}{\sqrt{1 - \bar{\gamma}\_s}} \epsilon\_{\theta}(x\_s, t\_s^{\text{align}}) \Big) \\
\\
\sigma\_{\theta}^{\text{fast}} (x\_s, s) &= \tilde{\eta}\_s^{\frac{1}{2}}
\end{align}
$$
<br>

빠른 샘플링 알고리즘은 아래에 요약되어 있습니다.

<p align="center">
<img src="https://i.ibb.co/717ZGwb/fast-sampling.png" alt="fast-sampling" border="0">
</p>

논문에서는 실험의 종류에 따라 $\small \\{ \eta\_t \\}\_{t=1}^{T\_{\text{infer}}}$를 $\small \\{ 0.0001, 0.001, 0.01, 0.05, 0.2, 0.7 \\}$ 또는 $\small \\{ 0.0001, 0.001, 0.01, 0.05, 0.2, 0.5 \\}$로 설정하여 사용합니다.

<br><br>

## DiffWave Architecture

DiffWave에서 학습되는 신경망은 $\small \epsilon_{\theta} : \mathbb{R}^L \times \mathbb{N} \rightarrow \mathbb{R}^L$입니다. 이 모델은 WaveNet과 거의 동일한 구조를 가지고 있지만 자기회귀적(autoregressive) 생성을 하지 않기 때문에 bidirectional dilated convolution을 사용한다는 점이 다릅니다. 여기서 bidirectional이라는 것은 bidirectional RNN에서 처럼 정방향과 역방향 양쪽으로 학습하는 개념을 말하는 것이 아니고 아래 그림과 같이 causal convolution이 아니라는 것입니다.

<p align="center">
<img src="https://i.ibb.co/kKsMpdP/bi-dilconv.png" alt="bi-dilconv" border="0">
</p>

아래 그림은 모델 구조를 나타낸 것입니다. 그림에 모델의 모든 구성 요소와 입출력 텐서의 모양까지 정확하게 잘 나와 있어 같이 보면서 글을 읽으면 이해하기 쉽습니다.

<p align="center">
<img src="https://i.ibb.co/5YmkzNg/architecture.png" alt="architecture" border="0">
</p>

모델은 $\small N$개의 residual 층으로 이루어져 있으며 각 층에 있는 bidirectional dilated convolution의 커널 크기는 3이고 채널 개수는 $\small C$입니다. Dilation은 각각의 층에서 두 배씩 늘어나고 $\small \[ 1, 2, 4, \cdots, 2^{n-1} \]$이 하나의 블록으로 구성되어 반복됩니다. 모든 residual 층에 대한 스킵 커넥션이 다 더해진 뒤 $\small 1 \times 1$ 컨볼루션을 거쳐서 최종적으로 1개 채널에 길이는 오디오 길이 $\small L$과 같은 시퀀스가 출력됩니다.

<br><br>

## Diffusion-step Embedding

신경망에는 데이터 $\small x\_0$ 뿐만 아니라 확산 스텝 $\small t$도 입력으로 들어가야 됩니다. 먼저 트랜스포머에서 위치 인코딩(positional encoding)을 하듯이 각각의 $\small t$가 128 차원의 벡터가 되도록 다음과 같이 인코딩 해줍니다.

<br>
\begin{equation}
t\_{\text{embedding}} = \left[ \sin (10^{\frac{0 \times 4}{63}} t), \cdots, \sin (10^{\frac{63 \times 4}{63}}), \cos (10^{\frac{0 \times 4}{63}} t), \cdots, \cos (10^{\frac{63 \times 4}{63}} t) \right]
\end{equation}
<br>

그 뒤 세 개의 완전연결(fully connected, FC)층을 통과시키는데 처음 두 개는 모든 residual 층들에 대해 파라미터를 공유하는 FC입니다. 마지막 FC는 그림에도 표현되어 있듯이 각각의 층마다 따로 존재하고 $\small C$차원의 임베딩 벡터를 출력합니다. 이 벡터는 오디오 길이만큼 브로드캐스팅 되어 각 residual 층의 입력과 더해(sum)집니다.

<br><br>

## Conditional Generation

음성 합성 분야에는 오디오 입력과 함께 다른 조건부 입력을 넣어주는 것이 보편화된 태스크가 많이 있습니다. DiffWave에는 각각의 경우에 따라 조건부 입력들을 처리하는 방법이 설계되어 있습니다.

### Local Conditioner

신경망 보코더는 언어적 특성(linguistic feature), 멜 스펙트로그램, 텍스트 등을 조건부로 입력합니다. DiffWave는 신경망 보코더 태스크에서 멜 스펙트로그램에 대해 조건화됩니다. 먼저 오디오를 80개의 주파수 구간으로 나누어진 멜 스펙트로그램으로 변환합니다. 스펙트로그램은 프레임 단위로 샘플 단위의 오디오 길이보다 짧으므로 2-D 전이 컨볼루션(transposed convolution)을 통해 오디오 길이와 같아지도록 업샘플링 해줍니다.

그 뒤 각 층마다 따로 존재하는 $\small 1 \times 1$ 컨볼루션 층을 통해 80개의 멜 밴드를 $\small 2C$개의 채널로 만들어줍니다. 이것은 dilated convolution 층의 출력에 더해집니다. $\small 2C$개의 채널을 만드는 이유는 dilated convolution에서 tanh 게이트와 sigmoid 필터에 각각 $\small C$개 채널씩 통과되기 때문입니다.

### Global Conditioner

발화자의 ID 등은 전역 조건으로 부여됩니다. 이러한 전역 조건부 태스크에 대해서는 $\small d\_{\text{label}} = 128$ 차원의 공유되는 임베딩을 사용합니다. 각각의 residual 층에서는 $\small d\_{\text{label}}$을 층마다 따로 존재하는 $\small 1 \times 1$ 컨볼루션 층을 통해 $\small 2C$ 개 채널로 만들어주고 dilated convolution 층의 출력에 더해줍니다.

<br><br>

## Unconditional Generation

조건화 되어 있지 않은 태스크에서는 모델이 조건부 입력 없이 일관성을 유지하도록 오디오를 생성하는 것이 중요합니다. 이를 위해서는 신경망의 수용 영역(receptive field) 크기 $\small r$이 오디오 길이 $\small L$보다 커야 할 필요가 있습니다. DiffWave에서는 bidirectional dilated convolution을 사용하므로 왼쪽 맨 끝이나 오른쪽 맨 끝에서 수용 영역이 오디오 전체를 커버하려면 실질적으로 $\small r \geq L$이 되어야 합니다.

Dilated convolution 층을 쌓을 때 수용 영역의 크기는 $\small r = (k - 1) \sum\_i d\_i + 1$이 됩니다. 여기서 $\small d\_i$는 $\small i$ 번째 residual 층의 dilation을 말하고 $\small k$는 커널 크기입니다. 예를 들어 dilation 사이클 $\small \[ 1, 2, \cdots, 512 \]$과 $\small k = 3$으로 이루어진 30개 층의 dilated convolution은 수용 영역 크기가 $\small r=6139$입니다. 이 크기는 16 kHz 샘플 레이트의 오디오에 대해서 겨우 0.38초에 해당되는 길이입니다.

추가적으로 층의 개수를 늘렸을 때에는 생성되는 오디오의 품질이 떨어지는 것이 관찰되었습니다. 이전 연구 결과를 보면 WaveNet에서는 6139 정도의 수용 영역 크기도 효과적으로 사용되기 어렵습니다. 하지만 DiffWave는 오디오 $\small x\_0$를 만들기 위해 $\small x\_T$부터 $\small x\_0$까지 역방향 과정을 반복하면서 수용 영역의 크기가 $\small T \times r$로 증가되는 효과가 있기 때문에 조건화 되지 않은 태스크에 대해 이점을 갖습니다.

<br><br>

## 실험

실험 결과에 대한 오디오 데모 샘플은 [데모 웹페이지](https://diffwave-demo.github.io/)에서 들어볼 수 있습니다.

### Neural Vocoding

신경망 보코더 태스크는 LJ speech [(Keith Ito, 2017)](https://keithito.com/LJ-Speech-Dataset/) 데이터셋을 사용합니다. 이 데이터셋은 22.05 kHz 샘플 레이트의 24시간 짜리 음성을 포함하고 있습니다. 총 데이터 수는 13,000개이며 발화자는 모두 동일한 여성입니다. 비교 모델은 WaveNet, ClariNet [(Wei Ping et al., 2019)](https://arxiv.org/abs/1807.07281), WaveGlow [(Ryan Prenger et al., 2019)](https://ieeexplore.ieee.org/abstract/document/8683143), 그리고 WaveFlow입니다 [(Wei Ping et al., 2020)](https://proceedings.mlr.press/v119/ping20a.html).

DiffWave 모델은 확산 스텝 $\small T \in \\{ 20, 40, 50, 200 \\}$와 residual 채널 수 $\small C \in \\{ 64, 128 \\}$에 따라 다양하게 비교합니다. 또한 $\small T\_{\text{infer}} = 6$을 사용한 빠른 샘플링 버전도 있습니다. 조건부 입력으로는 80개 주파수 대역의 멜 스펙트로그램을 사용합니다. 실험 결과는 아래 표에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/wJ3PQdZ/vocoder-result.png" alt="vocoder-result" border="0">
</p>

평가는 5개 스케일의 MOS로 진행했습니다. 수치 차이가 작은 편인데 실제로 샘플을 들어봐도 모델 간에 큰 차이는 없습니다. 테스트 시 실제 데이터(ground truth)의 멜 스펙트로그램을 조건부로 생성하기 때문에 대체로 실제와 비슷한 자연스러운 음성을 만들 수 있는 것으로 보입니다.

생성된 오디오의 품질은 비슷하지만 모델 파라미터 수에서는 DiffWave가 확실히 우위를 보입니다. 생성 시간 측면에서는 $\small \text{DiffWave}\_{\text{BASE}} (T=20)$가 실시간보다 2.1배 빠른 반면 WaveNet은 실시간에 비해 500배 느립니다. 하지만 가장 빠른 모델은 실시간보다 40배 빠른 WaveFlow라고 합니다.

### Unconditional Generation

조건화 되어 있지 않은 태스크는 SC09 [(Pete Warden, 2018)](https://arxiv.org/abs/1804.03209) 데이터셋을 사용합니다. 이 데이터셋에는 2,032명의 발화자가 $\small 0 \sim 9$의 숫자를 말한 총 8.7시간 짜리 음성이 들어 있습니다. 16 kHz 샘플레이트의 데이터는 모두 1초 길이이며 총 개수는 31,158개입니다.

비교 모델은 WaveNet과 멜 스펙트로그램 조건화를 제거한 Parallel WaveGAN입니다 [(Ryuichi Yamamoto et al., 2020)](https://ieeexplore.ieee.org/abstract/document/9053795). WaveNet은 residual 채널 수가 각각 128, 256인 두 가지 버전으로 실험했습니다. 결과는 아래 표에 정리되어 있습니다.

<p align="center">
<img src="https://i.ibb.co/RDD4hnM/unconditional-result.png" alt="unconditional-result" border="0">
</p>

평가에는 따로 학습시킨 분류기를 이용한 여러 정량 지표와 MOS를 사용했습니다. Fréchet inception distance(FID)와 [(Martin Heusel et al., 2017)](https://proceedings.neurips.cc/paper_files/paper/2017/hash/8a1d694707eb0fefe65871369074926d-Abstract.html) inception score(IS)는 [(Tim Salimans et al., 2016)](https://proceedings.neurips.cc/paper_files/paper/2016/hash/8a3363abe792db2d8761d6403605aeb7-Abstract.html) 생성된 샘플들의 품질과 다양성을 측정하고 IS의 경우 클래스가 명확하게 분류되는 것도 중요한 기준입니다. Modified inception score(mIS)는 [(Swaminathan Gurumurthy et al., 2017)](https://openaccess.thecvf.com/content_cvpr_2017/html/Gurumurthy_DeLiGAN__Generative_CVPR_2017_paper.html) IS와 비슷하지만 같은 클래스 내에서의 다양성도 측정합니다. AM score는 [(Zhiming Zhou et al., 2017)](https://arxiv.org/abs/1703.02000) IS에 더해 레이블의 주변 분포(marginal distribution)도 고려합니다. Number of statistically-different bins(NDB)는 [(Eitan Richardson and Yair Weiss, 2018)](https://proceedings.neurips.cc/paper_files/paper/2018/hash/0172d289da48c48de8c5ebf3de9f7ee1-Abstract.html) 생성된 샘플의 다양성을 측정합니다.

다양한 지표에 대해 모두 DiffWave가 가장 우수한 성능을 보여줍니다. 데모 오디오 샘플을 들어봤을 때에도 DiffWave의 결과가 꽤 자연스러우며 WaveGAN이나 WaveNet-256과 비교했을 때 큰 차이가 느껴집니다.

### Class-conditional Generation

마찬가지로 SC09 데이터셋에 대해서 숫자 레이블을 조건부로 입력한 실험도 진행했습니다. 실험 초기에는 Conditional WaveGAN도 [(Chae Young Lee et al., 2018)](https://arxiv.org/abs/1809.10636) 실험에 포함시켜 보았지만 생성된 오디오에 노이즈가 많고 품질이 떨어져 최종적으로는 WaveNet과만 비교합니다. 또한 생성 시에 레이블을 입력으로 넣어주기 때문에 평가 지표 중 AM score와 NDB는 큰 의미를 갖지 못해 제외합니다. FID는 같은 숫자 클래스 내에서 계산하는 것으로 수정하고 분류 정확도(accuracy)를 추가합니다. 결과는 아래 표에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/n3KsBcY/class-conditional-result.png" alt="class-conditional-result" border="0">
</p>

수치적으로 DiffWave가 큰 차이의 우세를 보이고 데모 샘플을 들어보았을 때에도 마찬가지입니다. 특히 이 실험에서의 생성 음성은 실제 데이터와 비교했을 때에도 품질이 많이 떨어지지 않고 상당히 자연스럽게 들립니다.

### Zero-shot Speech Denoising

SC09 데이터셋에는 데이터 증강(augmentation)에 사용할 수 있도록 숫자의 발화에 배경음으로 여섯 종류의 노이즈가 더해진 데이터가 있습니다. 이 노이즈 데이터는 학습 시에는 사용되지 않지만 테스트 시 조건화 되지 않은 DiffWave 모델에 $\small t=25$ 시점으로 넣어주면 역방향 과정을 거쳐서 노이즈가 제거되고 숫자의 깔끔한 발화만 남습니다. DiffWave 모델을 학습할 때에는 $\small t=T$의 화이트 노이즈에 대해서만 학습하고 이러한 특정 종류의 노이즈에 대한 지식은 제공되지 않는다는 의미에서 제로샷 디노이징이 가능하다는 것을 보여줍니다.

### Interpolation in Latent Space

숫자 레이블에 조건화 된 DiffWave 모델을 이용하여 $\small t=50$의 잠재 공간에서 두 발화자 $\small a, b$의 목소리를 보간하는 실험도 진행했습니다. 먼저 $\small t=50$에서 각 발화자의 데이터 $\small x\_t^a \sim q(x\_t \vert x\_0^a)$와 $\small x\_t^b \sim q(x\_t \vert x\_0^b)$를 샘플링합니다. 그 뒤 $\small x\_t^a$와 $\small x\_t^b$ 사이에서 $\small x\_t^{\lambda} = (1 - \lambda) x\_t^a + \lambda x\_t^b$로 선형 보간합니다. 이때 $\small 0 < \lambda < 1$입니다. 최종적으로 역방향 과정을 통해 $\small x\_0^{\lambda} \sim p\_{\theta}(x\_0^{\lambda} \vert x\_t^{\lambda})$를 샘플링합니다.

이 실험 결과의 데모 오디오 샘플을 들어보면 자연스럽게 목소리가 보간되지는 않습니다. 보간의 중간 지점에서 두 목소리가 겹쳐서 들리거나 연관성이 떨어지는 것 같은 음색이 나타납니다. 이 실험은 논문의 주 내용은 아니고 이러한 방식으로 보간을 시도해 볼 수 있다는 가능성을 보여주는 일종의 보너스에 해당합니다.

<br><br>

## Reference

[Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao and Bryan Catanzaro. DiffWave: A Versatile Diffusion Model for Audio Synthesis. In ICLR, 2021.](https://openreview.net/forum?id=a-xFK8Ymz5J)

[Pytorch Implementation of DiffWave](https://github.com/lmnt-com/diffwave)