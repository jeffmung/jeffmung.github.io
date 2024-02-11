---
layout: post
title: "[논문 리뷰] Adversarial Audio Synthesis"
image: https://i.ibb.co/8xJ0sCK/thumbnail.png
date: 2024-02-08
tags: 
categories: Paper-Review
use_math: true
---

<br><br>

## 논문 개요

2014년 GAN이 발표된 이후 이미지 생성에 이를 활용하는 많은 연구들이 진행되었지만 오디오 분야에서는 이 논문 이전까지 GAN을 효과적으로 적용한 연구가 거의 없었습니다. 이 논문에서는 이미지 분야에서 업샘플링을 이용하여 성공적인 결과를 보여준 DCGAN을 [(Redford et al., 2016)](https://arxiv.org/abs/1511.06434) 기반으로 하여 수 초 짜리 짧은 소리를 생성하는 두 가지 모델 WaveGAN과 SpecGAN을 제안합니다. WaveGAN은 시간 도메인에서 오디오 파형을 생성하는 방식이고, SpecGAN은 주파수 도메인에서 스펙트로그램을 이미지로 생성한 뒤 Griffin-Lim 알고리즘으로 대응하는 오디오 신호를 복원하는 방식입니다.

이러한 GAN 기반 모델은 WaveNET과 [(van den Oord et al., 2016)](https://arxiv.org/abs/1609.03499) SampleRNN [(Soroush Mehri et al., 2016)](https://openreview.net/forum?id=SkxKPDv5xl) 같은 자기회귀적(autoregressive) 방식의 모델들과 비교하여 속도가 빠르고 적은 양의 데이터로도 상대적으로 좋은 품질의 오디오를 생성할 수 있다는 장점을 보여줍니다. 하지만 긴 음악 생성이나 TTS 등의 태스크에서도 확실한 우위를 보이는 것은 아니기 때문에 WaveGAN 논문도 주로 드럼이나 새 소리 같은 짧은 사운드 이펙트 생성에 대한 실용적인 활용도나 비지도 학습인 GAN을 처음으로 적용한 구조를 설계했다는 점에 초점을 맞추고 있습니다.

<br><br>

## GAN Preliminaries

GAN은 알고 있는 사전 분포(prior distribution) $\small P_z$로부터 잠재 벡터(latent vector) $\small z \in Z$를 샘플링하여 이를 데이터 공간 $\small \mathcal{X}$로 매핑하는 법을 학습하는 생성 모델입니다. Discriminator $\small D : \mathcal{X} \rightarrow [0, 1]$는 샘플 $\small x$가 실제 데이터인지 생성된 데이터인지 구분하고, generator $\small G : \mathcal{Z} \rightarrow \mathcal{X}$는 discriminator를 속여서 생성된 샘플이 실제 데이터라고 판별되도록 학습됩니다.

오리지널 GAN 논문에서는 [(Goodfellow et al., 2014)](https://proceedings.neurips.cc/paper_files/paper/2014/hash/5ca3e9b122f61f8f06494c97b1afccf3-Abstract.html) 아래의 가치 함수를 최대화시키도록 학습됩니다. 이 식은 데이터셋의 분포 $\small P_X$와 generator로부터 생성된 샘플들의 분포 $\small P_G$ 사이의 Jensen-Shannon divergence를 최소화하는 것과 같은 의미를 갖습니다.

<br>
<center> $ V(D, G) = \mathbb{E}_{x \sim P_X} [\log D(x)] + \mathbb{E}_{z \sim P_Z} [\log (1 - D(G(z)))] $ </center>
<br>

이 방법의 문제점은 catastrophic failure에 취약하고 학습시키가 매우 어렵다는 것입니다. WGAN은 [(Arjovsky et al., 2017)](https://proceedings.mlr.press/v70/arjovsky17a.html) 이러한 단점을 극복하기 위한 모델로 아래와 같이 데이터셋과 생성된 샘플들의 분포 사이의 Wasserstein-1 distance를 최소화시키는 것을 제안합니다.

<br>
<center> $ W(P_X, P_G) = \sup_{\lVert f \rVert_L \leq 1} \mathbb{E}_{x \sim P_X} [f(x)] - \mathbb{E}_{x \sim P_G} [f(x)] $ </center>
<br>

이때 $\small \lVert f \rVert_{L} \leq 1 $는 함수 $\small f$가 1-Lipschwitz 조건을 만족한다는 의미입니다. Wasserstein distance를 최소화하기 위해서는 아래와 같은 가치 함수를 최대화시키도록 discriminator와 generator를 학습시킵니다.

<br>
<center> $ V_{\text{WGAN}}(D_w, G) = \mathbb{E}_{x \sim P_X} [D_w(x)] - \mathbb{E}_{z \sim P_Z} [D_w(G(z))] $ </center>
<br>

여기서 $\small D_w$는 Wasserstein distance를 계산할 수 있도록 1-Lipschwitz 조건을 만족하면서 샘플이 얼마나 진짜 같은지 점수를 매겨주는 critic 함수의 역할을 합니다. WGAN 논문에서는 $\small D_w$가 1-Lipschitz가 되도록 하기 위해 가중치 클리핑(weight clipping)을 하는 방법을 제안하고, WGAN-GP는 [(Gulrajani et al., 2017)](https://proceedings.neurips.cc/paper_files/paper/2017/hash/892c3b1c6dccd52936e27cbd0ff683d6-Abstract.html) 이를 그래디언트 페널티(gradient penalty)로 대체합니다.

WaveGAN과 SpecGAN은 기본적으로 WGAN-GP의 방식으로 모델을 학습시킵니다.

<br><br>

## Intrinsic Differencies between Audio and Images

이미지와 오디오 데이터가 어떤 특징적인 차이를 가지는지 principal component analysis(PCA)를 통해 볼 수 있습니다. 아래 그림은 각각 이미지의 $\small 5 \times 5$ 패치와 음성 신호의 조각에 대해 PCA를 적용하여 8개의 principal component를 시각화한 것입니다.

<p align="center">
    <img src="https://i.ibb.co/nLzbptW/pca.png" alt="model" border="0">
</p>

이미지의 principal component는 주로 선의 방향이나 강도 등의 특징을 포착하는 반면 오디오의 경우에는 각 주파수 성분의 주기적인 특징을 포착하는 것을 볼 수 있습니다. 오디오 신호 내에는 이러한 주기적인 특징들이 존재하기 때문에 모델이 넓은 범위에서의 특징을 표현할 수 있도록 설계되어야 합니다. 예를 들면 WaveNet은 이러한 오디오 데이터의 특성을 고려하여 수용 영역(receptive field)을 효율적으로 증가시킬 수 있는 dilated convolution을 사용했습니다.

<br><br>

## WaveGAN Architecture

WaveGAN의 모델 구조는 DCGAN에 기반하고 있습니다. 2차원의 이미지에 대해 디자인된 모델인 DCGAN을 1차원의 오디오에 적용할 수 있도록 바꾼 형태입니다. DCGAN의 generator는 전치 컨볼루션(transposed convolution)을 이용하여 저차원의 특징맵(feature map)을 고차원의 이미지로 업샘플합니다. 오리지널 DCGAN은 $\small 5 \times 5$ 필터를 가지고 스트라이드 2의 전치 컨볼루션을 하여 매 층(layer)마다 크기가 2배씩 업샘플되는데, WaveGAN에서는 $\small 25$ 길이의 필터와 스트라이드 4의 전치 컨볼루션을 통해 매 층마다 4배씩 업샘플합니다. 아래의 그림은 각각 DCGAN과 WaveGAN generator의 전치 컨볼루션을 묘사한 것입니다.

<p align="center">
    <img src="https://i.ibb.co/TH4tzwr/transposed-convolution.png" alt="transposed-convolution" border="0">
</p>

이때 샘플 사이즈는 패딩을 통해 맞춰줍니다. 매 층마다 4배씩 증가하여 최종적으로는 16384개의 샘플, 즉 16 kHz의 오디오 신호에 대해 약 1초 정도 되는 길이를 만듭니다. 그 외에도 배치 정규화(batch normalization)을 제외하고 16 bit의 데이터를 32 bit의 부동 소수점(floating point)으로 바꿔주는 등의 세부적인 디자인 요소들을 적용합니다. 더 구체적인 하이퍼파라미터와 모델 구조는 논문에 나와 있습니다.

<br><br>

## Phase Shuffle

전치 컨볼루션에 의해 업샘플을 하는 생성 모델들은 체커보드 아티팩트(checkerboard artifact)를 만드는 단점이 있는 것으로 알려져 있습니다. 일정한 간격으로 계산된 값들이 중첩되기 때문에 체스판 모양의 주기적인 패턴이 생기게 되는데 실제 이미지에서는 이러한 주기성을 가진 패턴들이 잘 나타나지 않으므로 discriminator가 이러한 패턴을 보면 쉽게 가짜라고 판별할 수 있습니다. 하지만 오디오 신호는 기본적으로 주기성을 갖기 때문에 이러한 노이즈를 실제의 자연스러운 소리를 구성하는 주파수와 잘 구분해낼 수 있어야 합니다.

Discriminator가 이를 구분해낼 수 있는 하나의 쉬운 방법은 위상(phase) 정보를 이용하는 것입니다. 체커보드 아티팩트는 항상 특정한 위상에서만 생성되기 때문에 단순히 특정 위상의 아티팩트 주파수가 있으면 가짜 샘플이라고 결정하는 것입니다. 이렇게 되면 GAN 모델이 의도대로 학습이 잘 되지 않을 수 있기 때문에 논문에서는 해결책으로 phase shuffle 방법을 제안합니다.

Phase shuffle은 discriminator에서 각 층의 출력을 다음 층에 넣기 전에 몇 칸씩 옆으로 이동시켜주는 것입니다. 몇 칸씩 이동할지는 하이퍼파라미터 $\small n$에 따라 $\small [-n, n]$에서 균일하게(uniform) 샘플링한 값으로 결정합니다. 이동시킨 뒤 비는 샘플들은 대칭이 되도록 채워줍니다. 아래의 그림은 $\small n=1$일 때의 phase shuffle 예시입니다.

<p align="center">
    <img src="https://i.ibb.co/F5SNbbj/phase-shuffle.png" alt="phase-shuffle" border="0">
</p>

직관적으로 말하면 phase shuffle에 의해 위상이 무작위로 이동되면서 체커보드 아티팩트가 만드는 특정한 위상 정보를 없애주고, discriminator는 이러한 아티팩트 정보에 의존하지 않고 샘플을 판별하는 법을 학습하게 됩니다.

<br><br>

## SpecGAN: Generating Semi-invertible Spectrograms

SpecGAN은 WaveGAN과 같은 접근법을 시간 도메인의 파형 대신 주파수 도메인의 스펙트로그램에 적용한 모델입니다. WaveGAN이 $\small 16384$개의 1차원 샘플을 생성하는 것에 대응하여 SpecGAN은 $\small 128 \times 128$ 크기의 2차원 스펙트로그램을 생성합니다. 이 스펙트로그램은 128개의 주파수 구간(frequency bin)과 128개의 프레임으로 이루어져 있습니다.

구체적으로는 먼저 오디오 신호에 8 ms의 스트라이드, 16 ms의 프레임 사이즈로 STFT를 적용합니다. 주파수 구간은 0부터 8 kHz까지 선형적으로 128개 구간을 나누고, 강도(magnitude)는 로그 스케일로 바꿔줍니다. 각각의 프레임 내에서는 주파수에 대해서 표준 정규화(standard normalization)을 해주고 $\small 3 \sigma$까지 클리핑하여 $\small [-1, 1]$ 사이로 스케일을 재조정합니다.

이렇게 스펙트로그램으로 변환된 데이터셋에 대해서 DCGAN 모델을 학습시킵니다. 그 뒤 생성된 스펙트로그램을 다시 선형 강도로 변형시키고 Griffin-Lim 알고리즘을 적용하여 16384개의 오디오 샘플을 만듭니다.

<br><br>

## 실험

평가에 사용된 데이터셋은 Speech Commands Zero Through Nine (SC09) [(Warden, 2018)](https://arxiv.org/abs/1804.03209), Drum sound effects, Bird vocalizations [(Boesman, 2018)](https://xeno-canto.org/contributor/OOECIWCSWV), Piano, 그리고 Large vocab speech (TIMIT)의 [(Garofolo et al., 1993)](https://catalog.ldc.upenn.edu/LDC93s1) 다섯 개입니다. SC09에는 다양한 목소리로 0부터 9 사이의 숫자를 말한 음성들이 들어 있습니다. 각각의 음성은 1초 길이이고 각각의 숫자들이 말해진 순서는 무작위입니다. 데이터셋에 대한 정보는 논문에 더 자세하게 나와 있습니다.

실험의 비교군으로는 WaveNET, SampleRNN, 그리고 parametric speech synthesizer를 [(Buchner, 2017)](https://www.kaggle.com/jbuchner/synthetic-speech-commands-dataset) 사용했습니다. 실험 결과의 데모 사운드는 [여기에서](https://chrisdonahue.com/wavegan_examples/) 들어볼 수 있습니다. 

### Inception Score

GAN 기반의 모델들은 정량적인 지표로 평가하기가 어렵습니다. Inception score는 [(Salimans et al., 2016)](https://arxiv.org/abs/1606.03498) 생성된 샘플의 다양성(diversity)과 구별하기 쉬운 정도(discriminability)를 기준으로 하는 정량적인 평가 지표입니다. 모델이 다양한 레이블의 샘플을 골고루 생성하고 각각의 생성된 샘플들이 특정한 클래스로 확연하게 구별될 수 있으면 높은 inception score를 받습니다.

샘플을 $\small x$, 레이블을 $\small y$라고 할 때, 다양성을 나타내는 분포는 주변 분포(marginal distribution) $\small P(y)$입니다. $\small P(y)$가 넓은(broad) 형태를 가지면 다양한 레이블을 가진 샘플들이 골고루 생성된다고 할 수 있습니다. 구별하기 쉬운 정도를 나타내는 분포는 $\small P(y \vert x)$입니다. $\small P(y \vert x)$가 날카로운(sharp) 형태를 띠면 각각의 샘플들이 특정한 클래스로 잘 구별된다고 할 수 있습니다.

Inception score는 이 두 분포의 KL divergence를 이용하여 다음 식과 같이 정의됩니다.

<br>
<center> $ \exp(\mathbb{E}_x D_{\text{KL}}(P(y | x) \Vert P(y))) $ </center>
<br>

Inception score는 레이블이 있는 SC09 데이터셋에 대한 평가에 활용되었습니다. Inception score를 계산하기 위해서 분류 모델(classifier)을 별도로 학습시킵니다. 분류 모델은 CNN을 기반으로 하고 멜 스펙트로그램으로 변환된 데이터셋에 대하여 각 샘플을 10개의 클래스로 분류합니다. 학습된 분류 모델은 테스트 셋에 대하여 93%의 accuracy를 보였습니다.

### Nearest Neighbor Comparisons

Inception score는 두 가지 경우에 대해 성능이 안 좋은 모델에 높은 점수를 줄 수 있습니다. 첫 번째는 모델이 각각의 클래스에 대해 한 가지의 샘플만 균일하게 생성하는 경우입니다. 두 번째는 모델이 트레인 셋에 오버피팅되어 분류 모델이 잘 학습한 샘플들만 생성하는 경우입니다.

이러한 두 원인으로 inception score가 높게 나오는 것을 식별하기 위해 두 가지 추가적인 지표를 사용합니다. 첫 번째 지표인 $\small \lvert D \rvert_{\text{self}}$는 1000개의 생성된 샘플들에 대해서 각각의 샘플의 nearest neighbor과의 평균 유클리디안 거리를 측정합니다. 높은 $\small \lvert D \rvert_{\text{self}}$ 값은 샘플들 간의 다양성이 높다는 것을 나타냅니다. 이때 거리는 분류 모델을 학습한 것과 같은 주파수 도메인의 표현에 대해 계산됩니다.

두 번째 지표인 $\small \lvert D \rvert_{\text{train}}$은 1000개의 생성된 샘플들에 대해 실제 트레인 셋에 있는 nearest neighbor과의 평균 유클리디안 거리를 측정합니다. 만약 모델이 트레인 셋에 있는 데이터를 그대로 생성한다면 $\small \lvert D \rvert_{\text{train}}$ 값은 $\small 0$이 됩니다.

### Qualitative Human Judgement

정성적인 평가는 사람이 생성된 샘플들을 듣고 각각의 클래스로 분류하는 정확도와 소리의 퀄리티, 알아듣기 쉬운 정도(ease of intelligibility), 그리고 발화자의 다양성에 대해 1부터 5점까지의 점수를 부여하는 것으로 이루어집니다. 정확도는 3000개의 샘플에 대해서, 나머지 MOS 테스트는 300개의 샘플에 대해서 평가했습니다.

### Results and Discussion

전체 실험 결과는 아래 표에 나와 있습니다. SampleRNN과 WaveNET의 평가 결과는 현저하게 떨어져서 표에서 제외되어 있습니다. 논문의 부록에 이 결과들도 나와 있는데, 모델의 절대적인 성능 차이라기 보다는 성격이 다르기 때문에 이 태스크에 대해 비교하는 것이 큰 의미가 없다는 점을 분명하게 명시하고 있습니다.

<p align="center">
    <img src="https://i.ibb.co/HdmqHHD/result.png" alt="result" border="0">
</p>

Inception score와 accuracy는 SpecGAN이 WaveGAN보다 높고 MOS 테스트는 WaveGAN이 더 좋은 결과를 보여줍니다. 논문에서는 SpecGAN이 Griffin-Lim 알고리즘으로 스펙트로그램을 신호로 복원하면서 생기는 손실이 원인일 것이라고 설명합니다. WaveGAN에 적용한 nearest neighbor와 post-processing은 체커보드 아티팩트 등의 노이즈를 제거하기 위한 시도인데 아래 섹션에서 더 자세하게 다루겠습니다. 결과적으로는 두 방법 모두 별 효과가 없는 것이 보입니다.

SC09 외의 다른 네 가지 데이터셋에 대해서는 평가 지표를 사용한 결과가 나와 있지 않고 WaveGAN이 더 나은 성능을 보여줬다고만 언급하고 있습니다. 아래의 그림은 다른 데이터셋을 포함하여 WaveGAN과 SpecGAN을 비교한 스펙트로그램 예시를 나타낸 것입니다.

<p align="center">
    <img src="https://i.ibb.co/gzdCfBh/spectrogram-example.png" alt="spectrogram-example" border="0">
</p>

<br><br>

## Understanding and Mitigating Artifacts in Generated Audio

전치 컨볼루션으로 인해 생기는 체커보드 아티팩트는 오디오 생성 모델의 성능을 저하시킬 수 있습니다. 이러한 아티팩트를 확인하기 위해 WaveGAN의 첫 번째 컨볼루션 층에 임펄스 신호를 통과시켜 나온 임펄스 응답(impulse response)를 아래 그림에 나타냈습니다.

<p align="center">
    <img src="https://i.ibb.co/XxgvXNK/artifact-filter.png" alt="artifact-filter" border="0">
</p>

위쪽에 있는 그림을 보면 250 Hz, 1 kHz, 4 kHz 등에 뾰족한 피크가 있습니다. 이것은 각각의 컨볼루션 층의 샘플 레이트에 해당하며, WaveGAN이 실제로 생성한 샘플들을 들어보면 $\small 247 \times 2^n$ Hz에 해당하는 B음에 가까운 노이즈가 들렸다고도 합니다.

### Learned Post-processing Filters

아티팩트를 제거하기 위한 단순한 시도로 generator에 512 길이로 WaveGAN 모델을 구성하는 다른 필터들에 비해 사이즈가 큰 post-processing filter를 추가하여 학습했습니다. 위의 그림에서 아래쪽에 있는 것이 학습된 post-processing filter를 나타낸 것인데 뾰족한 피크에 해당하는 부분을 줄이는 형태로 학습되었니다.

### Upsampling Procedure

전치 컨볼루션 대신 단순히 nearest-neighbor, linear, cubic interpolation을 사용하여 업샘플링 하면 체커보드 아티팩트가 생성되지 않습니다. 아래의 그림은 그 결과를 나타낸 것입니다.

<p align="center">
    <img src="https://i.ibb.co/ZHyFxkg/upsampling.png" alt="upsampling" border="0">
</p>

실험 섹션의 결과를 보면 결국 모든 방법 다 전치 컨볼루션을 사용한 WaveGAN에 비해 좋은 성능을 나타내지 못했고 nearest neighbor는 전치 컨볼루션과 어느 정도 비슷한 소리를 생성했지만 linear와 cubic interpolation의 경우에는 생성된 샘플의 퀄리티가 심하게 떨어졌다고 합니다.

<br><br>

## Reference

[Chris Donahue, Julian McAuley and Miller Puckette. Adversarial Audio Synthesis. In ICLR, 2019.](https://openreview.net/forum?id=ByMVTsR5KQ)

[Pytorch implementation of WaveGAN](https://github.com/mostafaelaraby/wavegan-pytorch)