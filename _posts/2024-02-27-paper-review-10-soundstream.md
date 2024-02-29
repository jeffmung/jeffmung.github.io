---
layout: post
title: "[논문 리뷰] SoundStream: An End-to-End Neural Audio Codec"
image: https://i.ibb.co/JmbFwzR/thumbnail.png
date: 2024-02-27
tags: 
categories: paper-review
use_math: true
---

<br><br>

## 논문 개요

<!-- excerpt-start -->

오디오 코덱은 음성이나 음악과 같은 오디오 신호를 압축하여 효과적으로 저장하거나 전송하고 다시 복원하여 재생할 수 있게 하는 알고리즘입니다. SoundStream은 dilated CNN을 기반으로 한 인코더와 디코더, 벡터 양자화(vector quantization), 그리고 적대적 손실(adversarial) 등을 활용하여 낮은 비트레이트에서도 높은 오디오 품질을 얻을 수 있는 신경망 오디오 코덱입니다.

<p align="center">
<img src="https://i.ibb.co/2hKjTFr/comparison.png" alt="comparison" border="0">
</p>

위의 그림은 기존의 전통적인 코덱인 Opus와 [(Jean-Marc Valin et al., 2012)](https://www.rfc-editor.org/rfc/rfc6716.html) EVS [(Martin Dietz et al., 2015)](https://ieeexplore.ieee.org/document/7179063), 그리고 구글에서 개발한 저비트레이트용 신경망 코덱 Lyra와 [(Willem Bastiaan Kleijn et al., 2021)](https://ieeexplore.ieee.org/abstract/document/9415120) 비교한 SoundStream의 성능을 보여줍니다. SoundStream은 3 kbps의 낮은 비트레이트에서 높은 품질을 보여주는 것 뿐만 아니라 하나의 모델을 여러 비트레이트에 대해서 사용할 수 있는 특징도 있습니다. 예를 들어 위 그림의 SoundStream - scalable은 18 kbps에서 학습시킨 모델을 3 kbps에서 동작시킨 것입니다.

또한 압축과 디노이징을 하나의 모델에서 동시에 수행하는 방법도 제안합니다. 디노이징 수행 여부를 선택하는 조건부 학습을 적용하여 배경 노이즈 제거가 필요한 경우 추가적인 시간 지연 없이 압축과 디노이징을 하나의 모델 내에서 수행할 수 있습니다.

<br><br>

## SoundStream Model Architecture

모델의 입력은 샘플 레이트 $\small f_s$ 로 만들어진 오디오 신호 $\small x \in \mathbb{R}^T$ 입니다. $\small x$ 는 차례대로 인코더, 잔차 벡터 양자화기(residual vector quantizer, RVQ), 디코더로 구성된 SoundStream 모델을 통과합니다. 학습에는 재구성 손실(reconstruction loss) 뿐만 아니라 판별자(discriminator)를 이용한 적대적 손실(adversarial loss)도 사용합니다. 그 구조는 아래 그림과 같습니다.

<p align="center">
<img src="https://i.ibb.co/rfwKRjk/architecture.png" alt="architecture" border="0">
</p>

SoundStream은 배경 노이즈를 제거하기 위한 디노이징 조건부 입력도 선택적으로 사용할 수 있도록 설계되어 있습니다. 추론(inference) 시에는 송신기(transmitter) 쪽의 인코더와 양자화기(quantizer)에서 압축된 신호가 수신기(receiver)로 전송되어 다시 오디오 신호로 디코딩 됩니다.

<br><br>

## Encoder Architecture

인코더의 구조는 아래 그림에 요약되어 있습니다.

<p align="center">
<img src="https://i.ibb.co/ssRwGwL/encoder.png" alt="encoder" border="0">
</p>

그림의 맨 왼쪽에 나와 있듯이 인코더의 구성은 순서대로 커널 크기 7과 $\small C_{\text{enc}}$ 개의 채널을 가진 1D 컨볼루션 층, $\small B_{\text{enc}}$ 개의 **EncoderBlock**, 그리고 임베딩 차원 $\small D$ 를 만들어주는 커널 크기 3과 스트라이드 1의 마지막 컨볼루션 층으로 이루어져 있습니다. 

각각의 **EncoderBlock**은 세 개의 **ResidualUnit**으로 이루어져 있습니다. 이 **ResidualUnit**에는 dilated causal convolution 층이 있고 하나의 **EncoderBlock** 내에서 세 **ResidualUnit**의 dilation 비율은 차례로 1, 3, 9입니다. 각각의 **EncoderBlock** 내에서 세 개의 **ResidualUnit**을 지날 때에는 채널 수와 시퀀스 길이가 유지되다가 마지막 컨볼루션 층에서 스트라이드를 통한 다운샘플링이 이루어지고 이때 채널 개수는 두 배가 됩니다.

**EncoderBlock**의 수 $\small B_{\text{enc}}$ 와 각각의 스트라이드 크기는 입력과 임베딩 시퀀스 사이의 시간적 해상도 비율, 혹은 재샘플링 비율(resampling ratio)을 나타냅니다. 예를 들어 $\small B_{\text{enc}} = 4$ 이고 각각의 스트라이드가 $\small (2, 4, 5, 8)$ 이면 하나의 임베딩은 $\small M= 2 \cdot 4 \cdot 5 \cdot 8 = 320$ 개의 입력 샘플들마다 계산됩니다. 따라서 입력이 $\small x \in \mathbb{R}^T$일 때 인코더의 출력은 $\small \text{enc}(x) \in \mathbb{R}^{\frac{T}{M} \times D}$ 가 됩니다.

<br><br>

## Decoder Architecture

디코더의 구조는 아래 그림과 같이 인코더와 반대로 되어 있습니다.

<p align="center">
<img src="https://i.ibb.co/KWb1Zpx/decoder.png" alt="decoder" border="0">
</p>

처음에 1D 컨볼루션 층이 있고 인코더와 같은 구조의 **ResidualUnit**을 사용하는 $\small B_{\text{dec}}$ 개의 **DecoderBlock** 이 나옵니다. 여기에서는 전이 컨볼루션(transposed convolution)을 기반으로 업샘플링이 이루어지며 스트라이드는 **EncoderBlock**과 반대 순서를 갖습니다. 업샘플링이 될 때 마다 채널 개수는 절반이 됩니다. 마지막에는 커널 크기 7과 1개 채널의 1D 컨볼루션 층이 다시 파형(waveform) 도메인의 $\small \hat{x}$ 를 만듭니다.

마지막 **DecoderBlock**의 채널 개수를 $\small C\_{\text{dec}}$ 라고 할 때, 위 그림처럼 인코더와 디코더의 채널 개수가 $\small C_{\text{enc}} = C_{\text{dec}} = C$ 로 같을 수도 있지만 비대칭적으로 $\small C_{\text{enc}} \neq C_{\text{dec}}$ 일 수도 있습니다. 이 경우에 대한 실험 결과도 뒤에 있습니다.

<br><br>

## Residual Vector Quantizer

벡터 양자화기(VQ)의 역할은 인코더의 출력 벡터 $\small \text{enc}(x)$ 를 bps로 표현된 타겟 비트레이트 $\small R$ 로 양자화하는 것입니다. 예를 들어 VQ-VAE에서 [(Aaron van den Oord et al., 2017)](https://proceedings.neurips.cc/paper/2017/hash/7a98af17e63a0ac09ce2e96d03992fbc-Abstract.html) 사용하는 것과 같은 벡터 양자화기의 코드북(codebook) 크기가 $\small N$ 이라면, 연속적인 벡터 시퀀스 $\small \text{enc}(x) \in \mathbb{R}^{S \times D}$ 는 원핫 벡터 시퀀스 $\small \text{VQ}(\text{enc}(x)) \in \mathbb{R}^{S \times N}$ 로 매핑됩니다. 이 양자화된 시퀀스는 $\small S \log_2 N$ bits 로 표현될 수 있습니다.

그러면 타겟 비트레이트가 $\small R=6000$ bps인 경우를 생각해보겠습니다. 인코더에서 스트라이드 $\small (2, 4, 5, 8)$을 통해 재샘플링 비율을 $\small M=320$ 으로 설정하면 샘플 레이트 $\small f_s = 24000$ Hz 오디오의 1초 길이는 75개의 프레임으로 표현됩니다. 따라서 한 프레임 당 배정되는 정보량은 $\small r=6000 / 75 = 80$ bits입니다. 일반적인 VQ를 사용하여 이만한 정보를 표현하려면 코드북의 크기가 $\small 2^{80}$ 이 되어야 합니다. 이러한 코드북을 학습시키는 것은 현실적으로 불가능합니다.

잔차 벡터 양자화기(RVQ)를 사용하면 이러한 문제를 해결할 수 있습니다. RVQ는 $\small N_q$ 개의 VQ 층으로 이루어져 있습니다. 처음 입력 벡터가 첫 번째 VQ를 통과하면서 코드북에 있는 가장 가까운 벡터가 배정됩니다. 그리고 배정된 코드북 벡터와 입력 벡터 사이의 잔차가 계산되어 다음 VQ를 통과합니다. 그러면 두 번째 코드북의 가장 가까운 벡터가 배정되고 또 잔차가 계산됩니다. 이런 식으로 $\small N_q$ 번 반복하여 최종적으로 $\small N_q$ 개의 코드북 벡터로 양자화됩니다. 아래에 알고리즘이 요약되어 있습니다.

<p align="center">
<img src="https://i.ibb.co/4Tf3VkG/rvq-algorithm.png" alt="rvq-algorithm" border="0">
</p>

코드북 크기 $\small N$ 의 VQ가 $\small N_q$ 개 있는 RVQ에서 각각의 VQ 층에 배정되는 정보량은 $\small r_i = r/{N_q} = \log_2 N$ 이 됩니다. 예를 들어 $\small N_q = 8$ 이면 각 VQ의 코드북 크기는 $\small N= 2^{r / N_q} = 2^{80 / 8} = 1024$ 입니다.

RVQ의 학습은 기본적으로 VQ-VAE-2의 [(Ali Razavi et al., 2019)](https://proceedings.neurips.cc/paper/2019/hash/5f8e2fa1718d1bbcadf1cd9c7a54fb8c-Abstract.html) 방법을 따라서 진행합니다. 그 손실 함수는 아래 식과 같습니다.

<br>
\begin{equation}
\mathcal{L}\_{\text{VQ}} = \lVert x - \hat{x} \rVert\_2^2 + \lVert sg[\text{enc}(x)] - \mathbf{e} \rVert\_2^2 + \beta \lVert sg[\mathbf{e}] - \text{enx}(x) \rVert\_2^2
\end{equation}
<br>

여기서 $\small \mathbf{e}$ 는 $\small \text{enc}(x)$ 에 대해 배정된 코드북 벡터, $\small \beta$는 손실 항들 사이에 비율을 조정하는 계수, 그리고 $\small sg$ 는 stop-gradient를 의미합니다. 이 중 두 번째 항인 코드북 벡터를 인코더의 출력에 가까워지도록 학습하는 손실에는 지수 이동 평균(exponential moving average) 업데이트를 사용합니다. 코드북 벡터를 배정된 인코더 출력 벡터들의 중간 위치로 만들고 싶은데 미니배치마다 배정되는 인코더 출력 벡터들의 종류와 수가 달라지기 때문입니다. 그 식은 다음과 같습니다.

<br>
\begin{align}
c\_j &\leftarrow \lambda c_j + (1 - \lambda) \sum_i \mathbb{1} [\text{enc}(x\_i) = \mathbf{e}\_j] \newline
\newline
\mathbf{e}\_j &\leftarrow \lambda \mathbf{e}\_j + (1 - \lambda) \sum\_i \frac{\mathbb{1} [\text{enc}(x\_i) = \mathbf{e}\_j] \cdot \text{enc}(x\_i)}{c\_j}
\end{align}
<br>

이때 $\small c\_j$ 는 미니배치 안에서 코드북 벡터 $\small \mathbf{e}\_j$ 가 배정된 인코더 출력 벡터들의 수에 대한 추정량이고 $\small j \in \\{ 1, 2, \ldots, N \\}$ 입니다. $\small \mathbb{1}[\cdot]$ 은 인디케이터 함수(indicator function)이며 $\small \text{enc}(x\_i) = \mathbf{e\_j}$ 는 미니배치 안의 인코더 출력 $\small \text{enc}(x\_i)$ 가 가장 가까운 코드북 벡터 $\small \mathbf{e}\_j$ 에 배정되었다는 의미입니다.

SoundStream에서는 여기에 추가적인 개선을 위한 두 가지 방법을 더 사용합니다. 첫 번째는 코드북 벡터를 초기화할 때 첫 배치에 대해 k-means 알고리즘을 돌려서 학습된 센트로이드들을 사용하는 것입니다. 이 방법은 코드북을 더 빠르게 입력 분포에 가깝게 만들어주어 효과적으로 학습되게 합니다.

두 번째는 몇 배치가 지났는데도 배정되지 않은 코드북 벡터가 있으면 현재 배치에서 임의로 샘플링된 벡터로 이를 대체하는 것입니다. 이에 대한 구체적인 구현으로는 지수 이동 평균 업데이트를 할 때 계산되는 $\small c\_j$ 가 2 이하로 떨어지는 코드북 벡터를 대체하도록 합니다.

RVQ에서 코드북의 크기 $\small N$ 을 고정하면 VQ 층의 개수 $\small N_q$ 가 비트레이트를 결정하게 됩니다. 따라서 원칙적으로는 타겟 비트레이트마다 하이퍼파라미터 설정을 맞춘 SoundStream 모델을 따로 학습시켜야 합니다. 하지만 이 논문에서는 여러 타겟 비트레이트에 대해서 작동할 수 있는 하나의 확장 가능한(scalable) 모델을 학습시키는 방법을 제안합니다.

각각의 입력이 들어올 때 먼저 $\small n_q$ 를 $\small [1, N_q]$ 안에서 균일하게 샘플링하고 $\small i=1, \ldots, n_q$ 에 대한 VQ $\small Q_i$ 만 사용합니다. VQ 층에 대해 적용한 일종의 드롭아웃(dropout)과 같은 것이라고 볼 수 있습니다. 결과적으로 모델은 $\small n_q = 1, \ldots, N_q$ 의 모든 값에 해당하는 타겟 비트레이트에 대해 오디오를 인코딩하고 디코딩할 수 있도록 학습됩니다. 추론 시에는 필요한 비트레이트에 맞게 $\small n_q$ 값이 선택됩니다.

<br><br>

## Discriminator Architecture

적대적 손실 함수를 계산하기 위한 판별자는 파형을 받는 것과 STFT를 받는 두 종류를 같이 사용합니다.

파형 기반 판별자는 각각 원본, 2배 다운샘플링, 4배 다운샘플링의 스케일에서 작동하는 서로 다른 세 개의 판별자를 사용합니다. 각 판별자는 시작 컨볼루션 층 한개, 4배씩 다운샘플링하고 4배씩 채널 개수가 늘어나는 컨볼루션 그룹 4개, 그리고 최종 출력이 1개 채널의 로짓(logit)이 되도록 만들어주는 추가적인 2개의 컨볼루션 층으로 이루어져 있습니다. 이 구조는 MelGAN에서 [(Kundan Kumar et al., 2019)](https://proceedings.neurips.cc/paper/2019/hash/6804c9bca0a615bdb9374d00a9fcba59-Abstract.html) 사용한 것과 같습니다.

STFT 기반 판별자는 단일 스케일에서만 작동하고 스펙트로그램의 실수부와 허수부를 모두 입력으로 사용합니다. 먼저 윈도우 크기 $\small W=1024$ 와 홉(hop) 크기 $\small H=256$ 으로 STFT를 적용한 뒤 커널 크기 7과 채널 수 32의 2D 컨볼루션 층을 통과시킵니다. 그 뒤에는 6개의 유닛이 있고 각 유닛은 커널 크기 3인 컨볼루션 층과 커널 크기 $\small (s_t, s_f)$ 인 컨볼루션 층으로 이루어져 있습니다. $\small (s_t, s_f)$ 는 시간 축과 주파수 축에 대한 다운샘플링 비율이고 (1, 2) 또는 (2, 2)의 값을 번갈아 사용합니다. 그 구조는 아래 그림에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/Sd9KRR2/stft-discrminator.png" alt="stft-discrminator" border="0">
</p>

참고로 이 그림에는 커널 크기 7의 시작 컨볼루션 층이 빠져 있습니다. 마지막 유닛의 출력 모양은 $\small T/(H \cdot 2^3) \times F / 2^6$ 이 되고 여기서 $\small T$ 는 원래 신호의 시간 축 샘플 개수, $\small F=W/2$ 는 주파수 구간의 개수입니다. 이후 커널 크기 $\small 1 \times F/2^6$ 과 채널 1개의 마지막 컨볼루션 층에서 다운샘플링 된 시간 축으로 1차원의 로짓만 남도록 만들어줍니다.

<br><br>

## Training Objective

모델 학습의 손실 함수는 적대적 손실(adversarial loss) $\small \mathcal{L}\_{\mathcal{G}}^{\text{adv}}$, 특징 손실(feature loss) $\small \mathcal{L}\_{\mathcal{G}}^{\text{feat}}$, 재구성 손실(reconstruction loss) $\small \mathcal{L}\_{\mathcal{G}}^{\text{rec}}$ 을 같이 사용합니다. 여기서 $\mathcal{G}$ 는 GAN의 생성자(generator)를 뜻하고 $\small \mathcal{G}(x) = \text{dec}(\text{RVQ}(\text{enc}(x)))$ 입니다. 판별자는 적대적 손실 $\small \mathcal{L}\_{\mathcal{D}}$ 로 학습됩니다.

적대적 손실을 표현하기 위해 $\small k \in \\{0, \ldots, K\\}$ 를 판별자들의 인덱스라 하고 $\small k=0$ 은 STFT 기반 판별자, $\small k \in \\{ 1, \ldots, K \\}$ 는 각각 다른 스케일의 파형 기반 판별자에 해당한다고 정의하겠습니다. $\small T_k$ 는 $\small k$ 번째 판별자의 시간 축에서 로짓의 개수입니다. 판별자의 손실 $\small \mathcal{L}\_{\mathcal{D}}$ 는 다음과 같습니다.

<br>
\begin{equation}
\mathcal{L}\_{\mathcal{D}} = \mathbb{E}\_x \left[ \frac{1}{K}\sum\_k \frac{1}{T\_k} \sum\_t \max(0, 1 - \mathcal{D}\_{k, t}(x)) \right] +
\mathbb{E}\_x \left[ \frac{1}{K}\sum\_k \frac{1}{T\_k} \sum\_t \max(0, 1 + \mathcal{D}\_{k, t}(\mathcal{G}(x))) \right]
\end{equation}
<br>

생성자의 적대적 손실은 다음과 같습니다.

<br>
\begin{equation}
\mathcal{L}\_{\mathcal{G}}^{\text{adv}} = \mathbb{E}\_x \left[ \frac{1}{K} \sum\_{k, t} \max(0, 1 - \mathcal{D}\_{k, t}(\mathcal{G}(x))) \right]
\end{equation}
<br>

특징 손실은 각 판별자의 내부 층에서 추출된 특징들을 실제 데이터와 생성된 오디오에 대해 비교하는 것입니다. 이는 다음 식을 통해 계산됩니다.

<br>
\begin{equation}
\mathcal{L}\_{\mathcal{G}}^{\text{feat}} = \mathbb{E}\_x \left[ \frac{1}{KL} \sum\_{k, l} \frac{1}{T_{k,l}} \sum\_t \lvert \mathcal{D}\_{k,t}^{(l)}(x) - \mathcal{D}\_{k,t}^{(l)}(\mathcal{G}(x)) \lvert \right]
\end{equation}
<br>

이때 $\small L$ 은 각 판별자 내부 층의 개수, $\small \mathcal{D}\_{k,t}^(l)$ 은 판별자 $\small k$ 의 $\small l \in \\{ 1, \ldots, L \\}$ 번째 층에서 출력된 $\small t$ 번째 특징, 그리고 $\small T\_{k,l}$ 은 시간 축 길이를 나타냅니다.

마지막으로 재구성 손실은 DDSP에서 [(Jesse Engel et al., 2020)](https://openreview.net/forum?id=B1x1ma4tDr) 제안한 멀티스케일 스펙트럼 손실(multi-scale spectral loss)을 사용합니다. 구체적으로는 다음 식과 같이 spectral energy distance [(Alexey Alexeevich Gritsenko et al., 2020)](https://proceedings.neurips.cc/paper/2020/hash/9873eaad153c6c960616c89e54fe155a-Abstract.html) 논문에서 사용한 설정을 그대로 적용합니다.

<br>
\begin{equation}
\mathcal{L}\_{\mathcal{G}}^{\text{rec}} = \sum\_{s \in 2^6, \ldots, 2^{11}} \sum\_t \lVert S\_t^s (x) - S\_t^s (\mathcal{G}(x)) \rVert\_1\ +
\alpha\_s \sum\_t \lVert \log S\_t^s (x) - \log S\_t^s (\mathcal{G}(x)) \rVert_2
\end{equation}
<br>

여기서 $\small S\_t^s (x)$ 는 윈도우 크기 $\small s$, 홉 크기 $\small s/4$, 그리고 64개 주파수 구간으로 계산된 멜 스펙트로그램의 $\small t$ 번째 프레임을 의미하고 $\small \alpha\_s = \sqrt{s/ 2}$ 를 사용합니다.

전체 생성자 손실은 다음 식과 같고 가중치 계수는 $\small \lambda\_{\text{adv}} = 1$, $\small \lambda\_{\text{feat}} = 100$, 그리고 $\small \lambda\_{\text{rec}} = 1$ 을 기본값으로 설정합니다.

<br>
\begin{equation}
\mathcal{L}\_{\mathcal{G}} = \lambda\_{\text{adv}} \mathcal{L}\_{\mathcal{G}}^{\text{adv}} + \lambda\_{\text{feat}} \mathcal{L}\_{\mathcal{G}}^{\text{feat}} + \lambda\_{\text{rec}} \mathcal{L}\_{\mathcal{G}}^{\text{rec}}
\end{equation}
<br>

<br><br>

## Joint Compression and Enhancement

전통적인 오디오 신호 처리에서는 압축과 음질 강화가 서로 다른 모듈에서 이루어졌습니다. 예를 들어 송신기 쪽에서 오디오가 압축되기 전 음질 강화 알고리즘을 적용한다거나 수신기 쪽에서 디코딩이 완료된 이후에 강화 알고리즘이 적용되는 식입니다. 이러한 방식에서는 각각의 모듈에서 발생하는 레이턴시로 인해 전체 레이턴시에 추가적인 영향이 생길 수 있습니다.

반면 SoundStream은 하나의 모델에서 압축과 디노이징 과정을 같이 수행함으로써 전체 레이턴시를 추가적으로 증가시키지 않습니다. 디노이징은 추론 시에 활성화와 비활성화 모드를 선택할 수 있도록 합니다. 이를 위해 학습 데이터셋은 $\small (\text{inputs}, \text{targets}, \text{denoising})$ 의 튜플 형태를 사용합니다. $\small \text{denoising} = \text{true}$  일 때 $\small \text{targets}$ 은 $\small \text{inputs}$ 에서 노이즈가 제거된 오디오입니다. $\small \text{denoising} = \text{false}$ 일 때에는 $\small \text{targets} = \text{inputs}$ 입니다. 또한 $\small \text{inputs}$ 가 이미 노이즈가 없는 상태라면 $\small \text{targets} = \text{inputs}$ 이고 $\small \text{denoise}$ 는 $\small \text{true}$ 와 $\small \text{false}$ 둘 다 될 수 있습니다.

디노이징 조건을 처리하는 층은 모델 내부 어느 곳에든 넣을 수 있습니다. 논문에서는 실험 결과 인코더나 디코더의 제일 안쪽에 적용하는 것이 효과적이었다고 합니다. 예를 들어 인코더 쪽이라고 하면 압축된 벡터가 디노이징 조건 입력에 따라 다른 벡터로 매핑된 뒤 RVQ를 통해 양자화 되는 것입니다. 또는 디코더 쪽이면 코드북의 양자화된 벡터가 디노이징 조건 입력에 따라 다른 벡터로 매핑됩니다.

이러한 매핑 과정의 수행에는 Feature-wise Linear Modulation(FiLM) [(Ethan Perez et al., 2018)](https://ojs.aaai.org/index.php/AAAI/article/view/11671) 층을 이용합니다. 이 층은 모델 내부에 삽입되어 직전의 신경망에서 추출된 특징을 다음과 같이 변환합니다.

<br>
\begin{equation}
\tilde{a}\_{n, c} = \gamma\_{n,c} a\_{n,c} + \beta\_{n,c}
\end{equation}
<br>

여기서 $\small a\_{n,c}$ 는 $\small c$ 번째 채널의 $\small n$ 번째 층의 활성화(activation) 출력을 의미하고 $\small \tilde{a}\_{n,c}$ 는 조건에 따라 다르게 매핑된 결과입니다. 매핑에 사용되는 가중치(weight) $\small \gamma\_{n, c}$ 와 편향(bias) $\small \beta\_{n,c}$ 은 $\small \text{denoising} = \text{true}$ 또는 $\small \text{denoising} = \text{false}$ 를 나타내는 원핫 벡터를 입력으로 받은 FilM의 선형 층(linear layer)에 의해 계산됩니다. 디노이징 모드를 결정하는 조건부 벡터는 시간에 따라 변하는 시퀀스가 될 수도 있습니다.

<br><br>

## 실험

실험에는 세 종류의 데이터셋을 사용합니다. 노이즈 없는 음성으로는 LibriTTS [(Heiga Zen et al., 2019)](https://arxiv.org/abs/1904.02882) 데이터셋을 사용하고 노이즈 데이터셋을 위해서는 Freesound의 [(Eduardo Fonseca et al., 2017)](https://archives.ismir.net/ismir2017/paper/000161.pdf) 노이즈 샘플과 LibriTTS를 혼합합니다. 음악 데이터셋으로는 MagnaTagTune을 [(Edith Law et al., 2009)](https://ismir2009.ismir.net/proceedings/OS5-5.pdf?ref=https://githubhelp.com) 사용합니다. 또한 현실 세계의 Noisy/reverberant speech 데이터셋도 수집하여 사용합니다.

주관적 평가 지표로는 여러 개의 오디오 신호를 동시에 숨겨진 참조 신호(hidden reference)와 비교하는 MUSHRA 점수를 사용합니다. 하이퍼파라미터 선택과 제거 실험(ablation test)을 위한 정량 평가 지표로는 ViSQOL을 [(Michael Chinen et al., 2020)](https://ieeexplore.ieee.org/abstract/document/9123150) 사용합니다.

비교 모델로는 Opus, EVS, 그리고 Lyra를 사용합니다. Opus는 6 kbps에서 510 kbps까지의 비트레이트 범위를 지원하고 유튜브 스트리밍, 마이크로소프트 팀즈, 줌 등에 활발하게 사용되고 있습니다. EVS는 5 kbps에서 128 kbs까지의 비트레이트를 지원하며 3GPP에 의해 표준화된 가장 최근의 코덱입니다. Lyra는 3 kbps에서 작동하는 자기회귀적(autoregressive) 생성 코덱입니다.

실험 결과에 대한 오디오 샘플들은 [데모 웹페이지](https://google-research.github.io/seanet/soundstream/examples/)에서 들어볼 수 있습니다.

### Comparisons with other Codecs

아래 그림은 왼쪽부터 낮은 비트레이트, 중간 비트레이트, 높은 비트레이트에서 주관적 평가를 시행한 결과입니다.

<p align="center">
<img src="https://i.ibb.co/MGpMRTz/mushra-result.png" alt="mushra-result" border="0">
</p>

동일한 비트레이트에서 SoundStream의 성능이 다른 모델들에 비해 훨씬 더 뛰어나고 더 높은 비트레이트의 Opus나 EVS에 비교해도 비슷한 품질을 보여줍니다. 보다 구체적인 데이터 종류에 따른 결과도 아래 그림에 정리되어 있습니다.

<p align="center">
<img src="https://i.ibb.co/0Qq0gjc/mushra-detail.png" alt="mushra-detail" border="0">
</p>

### Objective Quality Metrics

아래 그림은 정량적 지표로 평가한 결과를 나타냅니다.

<p align="center">
<img src="https://i.ibb.co/x2MxKQJ/objective-result.png" alt="objective-result" border="0">
</p>

맨 왼쪽 그림은 비트레이트에 따른 SoundStream의 성능 변화를 보여줍니다. Empirical entropy bound는 모든 VQ 층에서 양자화가 불필요하게 중복되지 않도록 하여 이론적으로 가능한 가장 작은 비트레이트를 나타냅니다. 예를 들어 18 kbps의 비트레이트에 해당하는 성능을 동일하게 내기 위해 이론적으로 가능한 최소 비트레이트를 계산하면 약 20% 정도의 메모리를 절약할 수 있습니다.

가운데 그림은 데이터 종류에 따른 결과이고 데이터의 다양성이 높은 음악이 가장 높은 난이도를 보여줍니다.

### Bitrate Scalability

위 그림에서 세 번째는 하나의 모델을 여러 비트레이트에 대해 적용하는 확장 가능성에 대한 실험 결과를 보여줍니다. Bitrate specific은 타겟 비트레이트에 맞춰서 학습과 평가를 진행한 것이고, not bitrate scalable은 18 kbps에 맞춰 학습하고 평가할 때에는 첫 $\small n_q$ 개의 VQ만 사용한 것, 그리고 scalable은 양자화기 레벨에서의 드롭아웃 기법을 적용하여 학습한 모델입니다.

Bitrate specific에 비해 not bitrate scalable 모델은 성능이 떨어지고 scalable 모델은 비슷한 성능을 유지하는 것을 볼 수 있습니다. 그리고 특이한 점은 9 kbps와 12 kbps에서는 오히려 scalable 모델이 더 높은 성능을 보여줍니다.

위 섹션의 주관적 평가 결과 그림에서도 12 kbps에서는 scalable 모델이 더 높은 성능을 보여주는 결과가 있습니다. 이러한 결과로 보아 양자화기 드롭아웃이 확장 가능성 뿐만 아니라 규제(regularization)의 효과도 제공하여 성능에 도움을 줄 가능성이 있습니다.

### Ablation Studies

첫 번째 제거 실험은 인코더를 같이 학습시키는 것에 대한 영향입니다. 예를 들어 Lyra는 학습 가능한 인코더가 아닌 고정된 멜 필터 뱅크(mel-filterbank)를 사용하고 속도 측면에서 이점이 있습니다. 비슷하게 SoundStream의 인코더를 멜 필터 뱅크로 대체했을 때 VisQOL이 3.96에서 3.33으로 크게 떨어졌습니다. 이러한 결과는 학습 가능한 인코더의 추가적인 표현력이 성능 향상에 큰 도움을 준다는 것을 보여줍니다.

두 번째는 인코더와 디코더 용량에 대한 실험입니다. 아래 표는 그 결과를 정리한 것입니다.

<p align="center">
<img src="https://i.ibb.co/4pncZjG/capacity-ablation.png" alt="capacity-ablation" border="0">
</p>

Real-time factor(RTF)는 SoundStream으로 인코딩과 디코딩을 할 때 걸리는 시간과 입력 오디오의 시간 사이의 비율을 나타냅니다. 기본 모델은 $\small C\_{\text{enc}} = C\_{\text{dec}} = 32$ 인데 $\small RTF > 2.3 \times$ 이므로 실시간으로 작동할 수 있습니다. 인코더와 디코더의 채널 크기를 반으로 줄이면 품질에 거의 손실이 없으면서 약 3배 이상의 속도 향상 효과가 있습니다.

비대칭적인 구조의 모델에 대한 실험 결과를 보면 인코더가 디코더보다 더 작은 경우가 반대의 경우보다 적은 품질 하락과 높은 속도 향상의 효과를 나타냅니다. 이러한 결과는 이미지 압축 분야에서 가벼운 인코더와 무거운 디코더를 쓰는 다른 연구 결과들과도 상응합니다.

다음으로는 RVQ에서 VQ 층의 깊이와 코드북 크기를 조정하는 실험도 진행했습니다. 하나의 프레임에 사용되는 비트 수가 $\small N_q \log_2 N$ 이기 때문에 서로 다른 조합의 $\small N_q$ 와 $\small N$ 으로 같은 타겟 비트레이트를 얻을 수 있습니다. 아래 표는 여러 조합에 대한 실험 결과를 보여줍니다.

<p align="center">
<img src="https://i.ibb.co/PQdVcgz/rvq-combination.png" alt="rvq-combination" border="0">
</p>

표에 있는 세 가지 조합 중에서는 코드북 크기가 1024이고 VQ 개수가 8개인 경우 가장 좋은 성능을 나타냅니다. 하지만 의외로 코드북 사이즈가 2인 1 bit VQ를 80층 쌓은 경우에도 성능 하락 폭이 크지는 않습니다.

마지막으로 모델 구조에서 기인하는 레이턴시에 대한 실험이 있습니다. 스트라이드의 곱으로 정의되는 구조적 레이턴시 $\small M$ 은 기본 설정에서 $\small 2 \cdot 4 \cdot 5 \cdot 8 = 320$ 샘플의 값을 갖고 이는 24 kHz 오디오에 대해 13.3 ms에 해당합니다. 같은 비트레이트에서 레이턴시가 늘어나면 한 프레임 당 필요한 비트 수도 늘어나므로 RVQ의 크기나 깊이도 늘어나야 합니다.

<p align="center">
<img src="https://i.ibb.co/DpnmZrs/latency.png" alt="latency" border="0">
</p>

위의 표는 6 kbps에서 코드북 크기를 고정하고 구조적 레이턴시에 따라 VQ의 개수 $\small N_q$ 를 조절한 결과입니다. 세 가지 경우 모두 오디오 품질에는 큰 차이가 없고, 레이턴시를 증가시킬 때 인코딩된 하나의 프레임이 대응하는 오디오 샘플 수가 증가하므로 RTF에 큰 이득을 주는 것을 볼 수 있습니다.

### Joint Compression and Enhancement

디노이징을 할 수 있는 SoundStream 모델에 대한 실험 결과는 아래 그림에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/7Ns3vvW/denoising.png" alt="denoising" border="0">
</p>

첫 번째와 두 번째 그림은 디노이징 조건을 각각 인코더쪽과 디코더쪽에서 처리하도록 만든 모델에 해당합니다. 두 가지 경우 모두 디노이징을 활성화했을 때 큰 품질 향상을 보여줍니다. 데모 샘플을 들어보면 음성만 남기고 배경 노이즈가 제거된 결과가 확실하게 나타납니다. 또한 두 가지 경우 사이에서 성능 상에 큰 차이는 없습니다.

선택적인 디노이징이 가능하도록 모델을 설계하는 것은 자연의 소리를 처리할 때와 같은 경우에는 디노이징이 오히려 오디오 품질을 떨어뜨릴 수 있기 때문입니다. 이러한 경우에 가능한 단점은 고정적으로 디노이징을 수행하는 모델보다 성능 측면에서 떨어지는 것입니다. 하지만 조건 없이 항상 디노이징을 하도록 학습된 모델에 대한 결과인 마지막 그림과 앞의 두 그림을 비교해보면 선택적인 디노이징이 가능한 모델들의 성능이 떨어지지 않습니다.

### Joint vs. Disjoint Compression and Enhancement

SoundStream은 압축과 디노이징을 하나의 모델에서 수행하는 방법을 제안합니다. 압축과 디노이징을 분리하여 수행하는 것과 제안된 방식을 비교하기 위해 SoundStream으로 압축을 하고 SEANet으로 [(Marco Tagliasacchi et al., 2020)](https://arxiv.org/abs/2009.02095) 디노이징을 하는 실험을 진행했습니다. 압축과 디노이징의 순서는 양 방향을 모두 비교합니다. 그 결과는 아래 표에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/BG5cM2D/joint-denoising.png" alt="joint-denoising" border="0">
</p>

하나의 모델에서 압축과 디노이징을 같이 수행하는 것이 레이턴시 측면에서 큰 이점이 있지만 성능 상으로는 분리하는 것에 비해 조금 떨어지는 것이 보입니다. 하지만 큰 차이는 아니고 SNR이 클수록 그 차이도 줄어듭니다.

<br><br>

## Reference

[Neil Zeghidour, Alejandro Luebs, Ahmed Omran, Jan Skoglund and Marco Tagliasacchi. SoundStream: An End-to-End Neural Audio Codec. In TASLP, 2021.](https://ieeexplore.ieee.org/abstract/document/9625818/citations#citations)

[Pytorch Implementation of SoundStream](https://github.com/lucidrains/audiolm-pytorch/blob/main/audiolm_pytorch/soundstream.py)