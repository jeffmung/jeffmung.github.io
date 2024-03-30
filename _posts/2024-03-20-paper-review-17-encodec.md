---
layout: post
title: "High Fidelity Neural Audio Compression [TMLR, 2023]"
description: EnCodec은 Meta AI에서 개발한 신경망 오디오 코덱(neural audio codec)으로 낮은 비트레이트에서도 복원 품질(high fidelity)이 높고 빠른 계산 시간으로 실시간(realtime) 활용이 가능합니다.
image: https://i.ibb.co/ynBJCyv/thumbnail.png
date: 2024-03-20
tags: 
categories: paper-review
use_math: true
---

## 논문 개요

<!-- excerpt-start -->

EnCodec은 Meta AI에서 개발한 신경망(neural) 오디오 코덱으로 낮은 비트레이트에서도 복원 품질(fidelity)이 높고 빠른 계산 시간으로 실시간 활용이 가능하다는 특징을 가지고 있습니다. 모델 구조와 학습 방법 등이 Google Research에서 2021년 발표한 SoundStream과 [(Neil Zeghidour et al., 2021)](https://ieeexplore.ieee.org/abstract/document/9625818/) 많은 부분에서 유사하지만 이 논문에서 보고된 결과 상으로 EnCodec이 좀 더 우세한 성능을 보여줍니다.

모델의 구조는 양자화기(quantizer)가 포함된 인코더-디코더 구조이고 적대적 손실(adversarial loss)을 이용하기 위한 판별자(discriminator)와 선택적으로 엔트로피 코딩을 할 수 있는 작은 트랜스포머 언어 모델을 사용합니다. 전체 구조는 아래 그림에 요약되어 있습니다.

<p align="center">
<img src="https://i.ibb.co/pJfNyct/architecture.png" alt="architecture" border="0">
</p>

<br><br>

## Encoder and Decoder Architecture

인코더에 입력된 시간 도메인의 오디오 파형 신호는 커널 크기 7의 컨볼루션 층으로 시작하여 여러 개의 인코더 블럭을 통과합니다. 각각의 인코더 블럭은 잔차 유닛(residual unit)과 커널 크기 K와 스트라이드 S의 다운샘플링 컨볼루션으로 이루어져 있습니다. $\small K = 2S$ 이고 4개 블럭의 스트라이드는 $\small S=(2, 4, 5, 8)$ 입니다. 다운샘플링 층에서 채널의 개수는 2배씩 늘어납니다.

그 뒤에는 두 층 짜리 LSTM이 있어 시퀀스 문맥(context) 정보가 추출됩니다. 마지막에는 커널 크기 7과 출력 채널 128개의 컨볼루션 층이 128차원의 연속적인 잠재(latent) 시퀀스를 만듭니다.

디코더는 인코더를 반대로 뒤집은 구조입니다. 다운샘플링 컨볼루션은 업샘플링을 위한 전치 컨볼루션(transposed convolution)이 됩니다. 마지막에는 커널 크기 7의 컨볼루션 층을 통해 모노 또는 스테레오의 오디오가 만들어집니다.

스트리밍용(streamable) 모델은 컨볼루션이 현재보다 과거의 타임스텝에만 적용되어야 하므로 필요한 패딩이 모두 앞쪽에 적용되고 비스트리밍용(non-streamable) 모델은 패딩이 시퀀스의 앞과 뒤에 절반씩 적용됩니다.

<br><br>

## Residual Vector Quantization

양자화는 SoundStream에서 사용하는 것과 같은 Residual Vector Quantizer (RVQ)를 이용하여 이루어집니다. 계층적인 구조의 RVQ는 이전 레벨에서 배정된 코드북 벡터와 원래 임베딩 벡터의 잔차(residual)를 다음 레벨의 코드북을 통해 양자화하고 잔차를 계산하는 것을 반복하는 방법입니다. 이러한 과정은 아래의 알고리즘에 요약되어 있습니다.

<p align="center">
<img src="https://i.ibb.co/5jbgGFD/algorithm.png" alt="algorithm" border="0">
</p>

코드북을 업데이트 할 때는 exponential moving average (EMA)를 사용하고 어떤 클러스터의 크기가 2 아래로 떨어지면 그 임베딩 벡터는 현재 배치의 벡터 중 하나로 대체됩니다. 학습 시에는 최대 32개의 코드북을 사용하고 추론 시에 타겟 비트레이트에 맞는 코드북 개수를 선택하여 하나의 모델이 여러 타겟 비트레이트에 대응될 수 있도록 합니다. 이러한 방법들은 모두 SoundStream에서 사용한 것과 동일합니다.

<br><br>

## Language Modeling and Entropy Coding

엔트로피 코딩의 일종인 산술 부호화(arithmetic coding)를 이용하면 정보량을 더 효율적으로 사용할 수 있습니다. 예를 들어 A, B, C의 글자를 코딩하고 싶을 때 A=00, B=01, C=10으로 각각 2 bit씩 할당하고 11이라는 코드를 낭비시키는 것이 아니라 각각을 다양한 길이의 코드로 표현하고 사용 빈도에 따라 나올 확률이 높은 글자는 더 적은 정보량으로 표현하는 방법입니다.

이러한 방법을 사용하기 위해서는 어떤 타임스텝에서 각각의 코드가 얼만큼의 확률로 나오는지에 대한 확률 분포가 주어져야 합니다. EnCodec에서는 작은 트랜스포머 언어 모델을 학습시켜서 이 확률 분포를 예측함으로써 산술 부호화를 통한 더 효율적인 정보량의 사용을 가능하게 합니다.

트랜스포머는 RVQ로 양자화가 된 이후에 사용되며 이전 타임스텝의 코드북 인덱스를 받아서 다음 타임스텝의 코드북 인덱스에 대한 확률 분포를 예측합니다. 이를 위해 트랜스포머의 출력은 코드북 개수 $\small N_q$ 개의 선형 층(linear layer)을 거쳐서 각각 코드북 크기만큼의 차원을 갖는 소프트맥스 확률 분포를 만듭니다. 이러한 산술 부호화의 사용은 계산량을 증가시키지만 비트레이트 측면에서는 더 효율적입니다.

<br><br>

## Discriminator

SoundStream과 마찬가지로 생성된 샘플의 품질을 향상시키기 위해 판별자(discriminator)도 사용합니다. 판별자는 다중 스케일의 STFT (multi-scale STFT, MS-STFT)를 기반으로 스펙트로그램을 입력으로 받습니다. 이때 STFT의 복소수 값(complex value)을 구성하는 실수부와 허수부가 연결되어(concatentated) 들어갑니다.

STFT의 스케일은 윈도우 길이 $\small [2048, 1024, 512, 256, 128]$ 의 5개를 사용합니다. 스테레오 오디오의 경우에는 왼쪽과 오른쪽 채널을 따로 분리하여 처리합니다. 각각의 판별자는 2D 컨볼루션 층과 확장된 컨볼루션(dilated convolution), 그리고 마지막의 1개 채널을 출력하는 2D 컨볼루션 층으로 이루어져 있습니다. 그 구조는 아래 그림에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/z2B3mM5/discriminator.png" alt="discriminator" border="0">
</p>

## Training Objective

EnCodec 모델 학습의 손실 함수는 시간과 주파수 도메인에서의 재구성 손실(reconstruction loss), 적대적 손실, 판별자 신경망에서 나오는 특징 매칭 손실(feature matching loss), 그리고 양자화기에서 나오는 commitment loss의 합으로 이루어져 있습니다.

시간 도메인의 재구성 손실은 원본과 재구성된 파형(waveform) 사이의 L1 손실로 다음과 같이 정의됩니다.

<br>
\begin{equation}
\ell\_t (x, \hat{x}) = \lVert x - \hat{x} \rVert\_1
\end{equation}
<br>

주파수 도메인의 재구성 손실로는 다음과 같은 다중 스케일 스펙트럼 손실(multi-scale spectrum loss)을 사용합니다.

<br>
\begin{equation}
\ell\_s (x, \hat{x}) = \frac{1}{\vert \alpha \vert \cdot \vert s \vert} \sum\_{\alpha\_i \in \alpha} \sum\_{i \in e} \lVert \mathcal{S}\_i (x) - \mathcal{S}\_i (\hat{x}) \rVert\_1 + \alpha\_i \lVert \mathcal{S}\_i (x) - \mathcal{S}\_i (\hat{x}) \rVert\_2
\end{equation}
<br>

이때 $\small \mathcal{S}\_i$ 는 64개 주파수 구간의 스펙트로그램이고 STFT의 윈도우 크기는 $\small 2^i$, 홉 길이(hop length)는 $\small e=5, \ldots, 11$ 일 때 $\small 2^i / 4$ 입니다. L1과 L2 손실 사이의 비율을 결정하는 $\small \alpha\_i$ 의 값은 $\small 1$ 을 사용합니다.

생성자의 적대적 손실은 $\small \ell_g(\hat{x}) = \frac{1}{K} \sum\_k \max (0, 1 - D\_k (\hat{x}))$ 이고 $\small K$ 는 판별자의 개수입니다. 판별자의 적대적 손실은 $\small L_d(x, \hat{x}) = \frac{1}{K} \sum\_{k=1}^K \max (0, 1 - D\_k(x)) + \max (0, 1 + D\_k(\hat{x}))$ 입니다. 판별자의 중간 층들에서 나오는 특징 매칭 손실은 다음 식과 같이 정의됩니다.

<br>
\begin{equation}
\ell\_{feat} (x, \hat{x}) = \frac{1}{KL} \sum\_{k=1}^K \sum\_{l=1}^L \frac{\lVert D\_k^l (x) - D\_k^l (\hat{x}) \rVert\_1}{\text{mean} (\lVert D\_k^l (x) \rVert\_1)}
\end{equation}
<br>

여기서 $\small D_k$ 는 각각의 판별자이고 $\small L$ 은 판별자 안의 층 개수입니다.

RVQ의 commitment loss는 인코더에만 적용되고 다음 식과 같이 정의됩니다.

<br>
\begin{equation}
\ell\_{w}  = \lVert z\_c - q\_c(z\_c) \rVert\_2^2
\end{equation}
<br>

이때 $\small C$ 는 코드북 개수이고 $\small z_c$ 는 현재의 잔차, $\small q_c(z_c)$ 는 현재의 코드북에서 배정된 가장 가까운 코드북 벡터입니다.

전체 생성자의 손실은 다음과 같이 계수 $\lambda$에 의해 결정된 비율로 합쳐집니다.

<br>
\begin{equation}
L\_G = \lambda\_t \cdot \ell\_t (x, \hat{x}) + \lambda\_s \cdot \ell\_s (x, \hat{x}) + \lambda\_g \cdot \ell\_g (\hat{x}) + \lambda\_{feat} \cdot \ell\_{feat} (x, \hat{x}) + \lambda\_w \cdot \ell\_w (w)
\end{equation}
<br>

각각의 손실들 사이의 그래디언트 스케일 차이는 학습을 불안정하게 만들 수 있습니다. 따라서 EnCodec에서는 손실들 사이의 균형을 맞추기 위한 balancer의 사용을 제안합니다. 각 손실의 그래디언트 $\small g_i = \frac{\partial{l_i}}{\partial{\hat{x}}}$ 에 대해서 원래 학습의 역전파(backpropagation)는 $\small \sum_i \lambda\_i g\_i$ 를 통해 이루어집니다. 이 대신에 가장 최근 배치에 대한 $\small g_i$ 의 이동 평균(moving average)을 $\small \langle \lVert g_i \rVert\_{2} \rangle\_{\beta}$ 에 대한 $\small \tilde{g}\_i$ 를 다음과 같이 정의합니다.

<br>
\begin{equation}
\tilde{g}\_i = \frac{\lambda\_i}{\sum\_j \lambda\_j} \cdot \frac{g\_i}{\langle \lVert g\_i \rVert\_2 \rangle\_{\beta}}
\end{equation}
<br>

그리고 $\sum\_i \tilde{g}\_i$ 를 통해 역전파하면 그래디언트들 사이의 스케일 균형을 맞출 수 있습니다. 이 방법은 commitment loss를 제외하고 $\small \hat{x}$ 에 의해 결정되는 손실들에 대해서만 적용됩니다.

<br><br>

## 실험

실험에 사용하는 데이터셋의 종류는 클린 음성(clean speech), 노이지 음성(noisy speech), 음악, 그리고 일반 오디오입니다. 클린 음성은 DNS Challenge 4, [(Harishchandra Dubey et al., 2022)](https://ieeexplore.ieee.org/abstract/document/9747230) 노이지 음성은 Common Voice [(Rosana Ardila et al., 2019)](https://arxiv.org/abs/1912.06670) 데이터셋을 사용하고 음악은 Jamendo [(Dmitry Bogdanov et al., 2019)](https://repositori.upf.edu/handle/10230/42015) 데이터셋, 그리고 일반 오디오는 AudioSet과 [(Jort F Gemmeke et al., 2017)](https://ieeexplore.ieee.org/abstract/document/7952261) FSD50K를 [(Eduardo Fonseca et al., 2021)](https://ieeexplore.ieee.org/abstract/document/9645159) 사용합니다.


베이스라인으로는 Opus [(Jean-Marc Valin et al., 2012)](https://www.rfc-editor.org/rfc/rfc6716.html), EVS [(Martin Dietz et al., 2015)](https://ieeexplore.ieee.org/abstract/document/7179063), 그리고 SoundStream을 사용합니다. SoundStream은 구글의 Lyra-v2에서 공개된 공식적인 소스 코드와 EnCodec 저자들이 다시 구현하여 약간 성능이 높아진 두 가지 버전을 모두 사용합니다. 실험 결과에 오디오 샘플은 [데모 샘플 웹페이지](https://ai.honu.io/papers/encodec/samples.html)에서 들어볼 수 있습니다.

### Evaluation Methods

주관적인 평가 지표로는 MUSHRA 프로토콜을 [(BS Series, 2014)](https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.1534-3-201510-I!!PDF-E.pdf) 사용하고 정량적인 지표로는 VISQOL과 [(Andrew Hines et al., 2012)](https://ieeexplore.ieee.org/abstract/document/6309421) SI-SNR을 [(Yi Luo et al., 2019)](https://ieeexplore.ieee.org/abstract/document/8707065) 사용합니다.

### Results

스트리밍용 설정에서 다양한 비트레이트에 따른 주관적 평가 결과는 아래 그림에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/2qtWjWt/mushra.png" alt="mushra" border="0">
</p>

또한 데이터셋 종류에 따라 구분한 실험 결과도 아래 표에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/WHsGwBm/result-categories.png" alt="result-categories" border="0">
</p>

전체적으로 EnCodec이 가장 높은 성능을 보여주고 특히 비슷한 방법의 SoundStream 보다도 우위에 있습니다. 또한 트랜스포머 언어 모델을 사용하여 엔트로피 코딩을 했을 때에는 비트레이트를 25~40% 정도 더 줄일 수 있습니다.

### The Effect of Discriminators Setup

EnCodec의 판별자는 MS-STFT라는 점에서 multi-scale discriminator (MSD)이지만 mono-STFT인 SoundStream과 차이가 있습니다. 또한 Hifi-GAN에서 [(Jungil Kong et al., 2020)](https://proceedings.neurips.cc/paper_files/paper/2020/hash/c5d736809766d46260d816d8dbc9eb44-Abstract.html) 사용하는 multi-period discriminator (MPD)의 방법도 생성된 오디오의 품질을 향상시키는 데 도움이 되는 것으로 알려져 있습니다. 이러한 다양한 판별자 설정에 대한 실험 결과는 아래 표에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/vHmdX5j/discriminator-ablation.png" alt="discriminator-ablation" border="0">
</p>

판별자 설정이 성능에 주는 영향이 꽤 있는 것을 알 수 있습니다. SoundStream에서 사용하는 MSD+Mono-STFT와 EnCodec의 MS-STFT를 비교해보면 MUSHRA 점수 차이가 상당히 큰 편입니다.

### The Effect of Streamable Modeling

스트리밍과 비스트리밍용 설정의 차이는 아래 표에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/t3Gkqmg/streamable-ablation.png" alt="streamable-ablation" border="0">
</p>

예상 가능한 결과로 스트리밍용 설정에서 성능이 약간 떨어집니다. 하지만 EnCodec에서 하락 폭이 크지는 않고 다른 베이스라인들의 스트리밍용 설정보다 EnCodec의 비스트리밍용 설정이 여전히 훨씬 우세한 성능을 나타냅니다.

### The Effect of the Balancer

EnCodec에서 제안하는 balancer의 영향은 아래 표의 실험 결과에서 볼 수 있습니다.

<p align="center">
<img src="https://i.ibb.co/kJyYwdr/balancer-ablation.png" alt="balancer-ablation" border="0">
</p>

Jamando 음악 데이터셋에 대한 실험 결과이고 balancer 유무에 따른 성능 차이가 큰 편입니다. 하지만 이 결과는 RVQ가 아닌 DiffQ [(Alexandre Défossez et al., 2021)](https://arxiv.org/abs/2104.09987) 방법을 통한 양자화를 적용했을 때의 결과입니다. 이러한 설정에서 실험한 이유에 대해서는 논문에 설명되어 있지 않습니다.

### Alternative Quantizers

EnCodec의 양자화 방법으로 각각 RVQ와 DiffQ를 적용했을 때의 비교 실험 결과는 아래 표에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/BP4HfMq/quantization-ablation.png" alt="quantization-ablation" border="0">
</p>

RVQ가 좀 더 우세한 성능을 보여주고 여기서 비교한 SoundStream은 EnCodec 논문 저자들이 다시 구현한 버전입니다.

### Stereo Evaluation

앞에서 보여준 실험 결과들은 모두 모노 오디오에 대한 것입니다. 48 kHz의 스테레오 오디오에 대한 실험 결과는 아래 표에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/zrgYz1h/stereo-result.png" alt="stereo-result" border="0">
</p>

Opus와 MP3만을 비교군으로 사용하긴 했지만 역시 EnCodec이 가장 우세한 성능을 나타냅니다.

### Latency and Computation Time

EnCodec의 지연 시간을 분석하기 위해 real time factor를 정의하여 측정합니다. 여기서 정의한 real time factor는 오디오 신호의 길이와 처리 시간 사이의 비율로 real time factor 값이 1보다 크면 실시간보다 빠른 것입니다. SoundStream과 EnCodec에 대한 실험 결과는 아래 표에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/KVNPW3q/real-time-factor.png" alt="real-time-factor" border="0">
</p>

EnCodec은 SoundStream보다는 속도가 느리긴 하지만 그래도 24 kHz 기준으로 실시간보다 약 10배 빠르기 때문에 모델을 실제로 활용하는 것에 시간 측면에서의 문제는 없는 수준입니다.

## Reference

[Alexandre Défossez, Jade Copet, Gabriel Synnaeve and Yossi Adi. High Fidelity Neural Audio Compression. In TMLR, 2023.](https://openreview.net/forum?id=ivCd8z8zR2)

[Official Source Code of EnCodec](https://github.com/facebookresearch/encodec)