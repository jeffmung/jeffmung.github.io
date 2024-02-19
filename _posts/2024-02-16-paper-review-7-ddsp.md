---
layout: post
title: "[논문 리뷰] DDSP: Differentiable Digital Signal Processing"
image: https://i.ibb.co/nkKS7pc/thumbnail.png
date: 2024-02-16
tags: 
categories: Paper-Review
use_math: true
---

<br><br>

## 논문 개요

DDSP는 구글 리서치에서 개발한 오디오 합성 라이브러리입니다. 이 논문에서는 보코더와 신디사이저 같은 고전적인 방법의 장점인 도메인 지식과 해석 가능한 모듈의 활용, 그리고 딥러닝 모델의 장점인 강력한 표현력과 엔드투엔드 학습을 결합한 방법을 제시합니다. 그 결과로 DDSP는 높은 품질(fidelity)의 오디오를 생성하고 음높이(pitch), 음량(loudness), 음색(timbre), 리버브를 독립적으로 제어할 수 있도록 합니다.

<br><br>

## Challenges of Neural Audio Synthesis

신경망을 이용해서 오디오를 합성하는 모델은 방식에 따라 크게 시간 도메인에서 직접 파형을 생성하는 모델, 주파수 도메인에서 스펙트로그램의 형태로 푸리에 계수(Fourier coefficient)를 생성하는 모델, 한 번에 하나의 샘플씩 자기회귀적(autoregressive)으로 파형을 생성하는 모델로 구분할 수 있습니다. 첫 번째에 해당하는 모델로는 SING [(Alexandre Defossez et al., 2018)](https://proceedings.neurips.cc/paper/2018/hash/56dc0997d871e9177069bb472574eb29-Abstract.html), WaveGAN [(Chris Donahue et al., 2019)](https://openreview.net/forum?id=ByMVTsR5KQ) 등이 있고 두 번째는 Tacotron [(Yuxuan Wang et al., 2017)](https://www.isca-archive.org/interspeech_2017/wang17n_interspeech.html), GANSynth [(Jesse Engel et al., 2019)](https://openreview.net/forum?id=H1xQVn09FX), 세 번째는 WaveNET [(Aaron van den Oord et al., 2016)](http://arxiv.org/abs/1609.03499), SampleRNN [(Soroush Mehri et al., 2016)](https://openreview.net/forum?id=SkxKPDv5xl) 등이 해당합니다.

<p align="center">
    <img src="https://i.ibb.co/MDFxnWF/neural-audio-synthesis.png" alt="neural-audio-synthesis" border="0">
</p>

시간 도메인에서 프레임을 중첩시켜 생성하는 스트라이드 컨볼루션 모델은 오디오 파형의 주기와 프레임의 주기가 달라서 생기는 위상 정렬 문제를 갖습니다. 오디오 신호를 구성하는 주파수가 다양하기 때문에 신경망의 필터가 모든 주파수에서 프레임마다 달라지는 위상들의 모든 조합을 다 학습하기에는 한계가 있습니다. 위의 그림에서 첫 번째가 이러한 한계를 나타냅니다.

푸리에 기반 모델은 STFT를 적용할 때 프레임 크기와 신호의 주기가 정수 배로 정확하게 나누어지지 않기 때문에 생기는 스펙트럼 누출(spectral leakage) 문제가 있습니다. 위 그림에서 가운데가 이에 해당합니다.

마지막으로 자기회귀적 모델은 한 번에 하나의 샘플씩만 생성하기 때문에 이러한 문제에서는 자유롭지만 모델 크기, 학습 데이터의 양, 시간 측면에서 단점이 있고 사람의 지각을 반영한 spectral loss 같은 손실 함수를 쓰기가 어려워 비효율적인 문제가 있습니다. 예를 들어 위의 그림에서 맨 오른쪽에 있는 세 개의 파형은 모두 사람이 들을 때는 동일하게 들리지만 자기회귀적 모델에서는 서로 다른 손실 값을 나타냅니다.

<br><br>

## Oscillator Models

보코더나 신디사이저는 오실레이터를 통해 직접 신호를 생성합니다. 이러한 모델들은 도메인 지식을 활용하여 해석 가능한 파라미터들을 조절해서 합성 알고리즘에 사용합니다. 사람이 직접 경험과 지식에 기반해 파라미터들을 튜닝해줘야 한다는 단점이 있으며, 신경망을 통해 추출된 합성 파라미터들을 모델링하는 연구들도 있지만 이러한 파라미터들을 조절하는 알고리즘이 미분 불가능하여 그래디언트가 흐를 수 없기 때문에 한계가 있습니다.

<br><br>

##  Spectral Modeling Synthesis

DDSP는 일종의 Spectral Modeling Synthesis (SMS)의 [(Xavier Serra and Julius Smith, 1990)](https://www.jstor.org/stable/3680788) 미분 가능한 버전이라고 할 수 있습니다. SMS는 사인파들을 합산하는 가산 신디사이저(additive synthesizer)와 화이트 노이즈를 필터링하는 감산 신디사이저(subtractive synthesizer)를 결합하여 소리를 생성해냅니다. 이러한 방식의 모델들 중에서 DDSP는 Harmonic plus Noise 모델을 [(James W Beauchamp, 2007)](https://link.springer.com/book/10.1007/978-0-387-32576-7) 사용합니다. 이 모델은 사인파들을 기본 주파수(fundamental frequency)의 정수 배에 해당하는 주파수들로 제한합니다.

<br><br>

## DDSP Components

DDSP는 오실레이터, 엔벨로프(envelope), 필터 등으로 구성되어 있습니다. 이러한 요소들은 디지털 신호 처리(digital signal processing, DSP)에서 흔히 사용하는 함수들인데 DDSP에서는 이것들을 모두 미분 가능한 피드포워드 함수로 구현하여 역전파(backpropagation)를 통한 엔드투엔드 학습이 가능하도록 합니다.

DDSP 요소들을 통합하여 모델을 만들 때 이론적으로는 GAN, VAE, flow 등 다양한 형태의 확률적인(stochastic) 생성 모델들로 구현하는 것이 가능하지만 논문에서는 기본적으로 결정론적인(deterministic) 오토인코더의 형태로 설계합니다. DDSP 오토인코더의 전체 구조는 다음 그림에 표현되어 있습니다.

<p align="center">
    <img src="https://i.ibb.co/jvNhxN7/ddsp-autoencoder.png" alt="ddsp-autoencoder" border="0">
</p>

빨간색은 신경망 구조에 해당하고 초록색은 잠재 표현(latent representation), 노란색은 결정론적인 신디사이저와 이펙트를 나타냅니다.

<br><br>

## Harmonic Oscillator / Additive Synthesizer

오실레이터가 이산(discrete) 타임스텝 $\small n$에 대해서 신호 $\small x(n)$을 출력하는 것은 다음 식과 같이 표현됩니다.

<br>
<center> $ x(n) = \sum_{k=1}^{K} A_{k}(n) \sin (\phi_{k}(n)) $ </center>
<br>

여기서 $\small A_{k}(n)$은 $\small k$ 번째 사인파의 시간에 따라 변하는(time-varying) 진폭이고 $\small \phi_{k}(n)$은 그것의 순간 위상(instantaneous phase)입니다. 사인파를 원을 회전하는 점이라고 생각하면 어떠한 시간에서의 순간적인 각도가 이 순간 위상에 해당하는 것입니다. 이때 각속도에 해당하는 것은 순간 주파수입니다. 위상 $\small \phi_{k}(n)$는 다음 식과 같이 순간 주파수 $\small f_{k}(n)$을 시간에 대해 적분하여 얻어집니다.

<br>
<center> $ \phi_{k}(n) = 2\pi \sum_{m=0}^{n} f_{k}(m) + \phi_{0, k} $ </center>
<br>

$\small \phi_{0, k}$는 초기 위상인데 임의의 값이거나 고정된 값일수도 있고 학습될 수도 있습니다.

하모닉 오실레이터에서는 모든 사인파의 주파수가 기본 주파수 $\small f_{0}(n)$의 정수 배입니다. 즉, $\small f_{k} (n) = k f_0 (n)$입니다. 따라서 하모닉 오실레이터의 출력은 시간에 따라 변하는 기본 주파수 $\small f_0 (n)$와 하모닉 진폭 $\small A_k (n)$에 완전히 매개변수화(parameterize)될 수 있습니다. 해석 가능성(interpretability)을 높이기 위해 하모닉 진폭은 추가적으로 다음과 같이 분해합니다.

<br>
<center> $ A_k (n) = A(n) c_k (n) $ </center>
<br>

여기서 $\small A(n)$은 음량을 조절하는 전역(global) 진폭이고 $\small c_k (n)$은 각 하모닉들의 정규화된 분포입니다. 따라서 $\small \sum_{k=0}^{K} c_k (n) = 1$이고 $\small c_k (n) \geq 0$입니다.

진폭과 하모닉 분포는 위의 DDSP 전체 구조 그림에서 **Harmonic Audio**로 들어가는 **Decoder**의 출력에 해당합니다. 전역 진폭 $\small A(n)$의 차원(dimension)이 1이고 하모닉 분포 $\small c_k (n)$의 차원은 사인파의 개수와 같은데 기본적으로 101의 값을 사용합니다. 구현할 때에는 102 차원의 벡터를 디코더에서 출력하여 첫 번째 인덱스의 값은 $\small A(n)$, 두 번째부터 끝까지의 값에는 소프트맥스를 적용하여 $\small c_k (n)$이 되게 합니다. 또한 진폭과 하모닉 분포를 양수로 제한하기 위해서 다음과 같이 변형된 시그모이드를 비선형성을 위한 활성화 함수로 사용합니다.

<br>
<center> $ y = 2.0 \cdot \text{sigmoid}(x)^{\log 10} + 10^{-7} $ </center>
<br>

하모닉 오실레이터가 소리를 생성하는 과정은 아래 그림에 요약되어 있습니다.

<p align="center">
    <img src="https://i.ibb.co/dpyLVMk/harmonic-oscillator.png" alt="harmonic-oscillator" border="0">
</p>

먼저 시간에 따라 변하는 기본 주파수 $\small f_0 (n)$를 정수배 한 구성 주파수 $\small f_k (n)$들을 시간에 따라 적분하여 순간 위상 $\small \phi_k (n)$과 사인파 $\small \sin(\phi_k (n))$을 만들고 전역 진폭 $\small A(n)$을 곱해줍니다. 그리고 각 사인파들마다 하모닉 분포 $\small c_k(n)$을 곱해서 합산해줍니다. 이렇게 만들어진 신호는 위의 DDSP 전체 구조 그림에서 **Harmonic Audio**의 출력에 해당합니다.

<br><br>

## Envelopes

위에서 설명한 오실레이터 수식들은 오디오 샘플 레이트의 해상도로 시간에 따라 변하는 진폭과 주파수에 대해 계산됩니다. 하지만 DDSP의 신경망, 즉 전체 구조 그림에서 **Encoder**와 **Decoder**에 해당하는 부분은 프레임 단위의 해상도로 동작합니다. 예를 들어 **Harmonic Audio**로 들어가는 **Decoder**의 출력 텐서 모양은 (프레임 개수, 채널 개수=102)가 됩니다.

따라서 프레임 해상도의 벡터를 오디오 샘플 레이트 해상도로 업샘플링하는 것이 필요합니다. 순간 주파수에 대해서는 단순하게 이중 선형 보간(bilinear interpolation)으로 업샘플링 하는 것이 실험적으로 적절했다고 합니다. 하지만 전역 진폭과 하모닉 분포는 아티팩트를 방지하기 위해 스무딩(smoothing) 해주는 것이 필요합니다.

이를 위해서 각 프레임의 진폭 값에 Hamming 윈도우를 곱해서 중첩하여 더해주는 방식으로 스무딩된 진폭 엔벨로프를 만들어줍니다. 논문에서는 hop 크기 64, 윈도우 프레임 크기 128의 50% 중첩을 사용합니다. 예를 들어 진폭의 프레임 개수가 250이면 250개의 진폭값이 곱해진 크기 128의 윈도우들을 만들어서 64의 hop으로 시간에 따라 더해주는 것입니다. 업샘플링된 시퀀스 양 끝의 중첩되지 않은 부분, 즉 Hamming 윈도우에 의해 시작 쪽에서는 진폭이 0부터 올라가고 끝 쪽에서는 0으로 내려가는 부분은 잘라내줍니다.

<br><br>

## Filtered Noise / Subtractive Synthesizer

DDSP에서 기본적으로 사용하는 harmonic plus noise 모델에서 감산 신디사이저에 해당하는 부분은 화이트 노이즈를 필터링 하는 것으로 소리를 생성합니다. 논문에서는 주파수 샘플링 방법을 사용해서 필터의 임펄스 반응(impulse response, IR)의 주파수 도메인 전달 함수를 신경망이 출력할 수 있도록 모델을 디자인합니다. 이 필터는 시간에 따라 변하는 선형 유한 임펄스 반응(linear time variant finite impulse response, LTV-FIR) 필터가 됩니다.

구체적으로 $\small l$ 번째 프레임에 대한 신경망의 출력은 벡터 $\small H_l$이 되고 $\small h_l = \text{IDFT}(H_l)$입니다. $\small l$ 번째 프레임의 오디오 신호 $\small x_l$에 대하여 먼저 주파수 도메인에서 프레임 단위의 컨볼루션 $\small Y_l = H_l X_l$을 수행합니다. 이때 $\small X_l = \text{DFT}(x_l)$이고 $\small Y_l = \text{DFT}(y_l)$입니다. 그 뒤 IDFT를 적용하여 시간 도메인으로 변환한 $y_l = \text{IDFT}(Y_l)$을 얻고 처음 DFT를 한 hop 크기와 동일하게 프레임들을 중첩하여 더해줍니다.

이 신경망의 출력 $\small H_l$은 DDSP 전체 구조 그림에서 **Filtered Noise**로 들어가는 **Decoder**의 출력이 됩니다. 실제로는 출력을 그대로 $\small H_l$로 사용하지는 않고 Hann 윈도우를 주파수 도메인으로 변환하여 신경망의 출력에 적용한 것을 $\small H_l$로 합니다. 또한 먼저 IR을 위상이 0인 대칭 형태(symmetric form)으로 옮겨서 윈도우를 적용하고 다시 인과 형태(causal form)로 이동시켜(shift) 줍니다.

필터가 적용되는 노이즈 신호인 $\small x_l$은 $\small [-1, 1]$에서 균일(uniform)하게 샘플링합니다. 필터의 크기, 즉 벡터 $\small H_l$의 차원은 65를 기본값으로 사용합니다.

<br><br>

## Reverb: Long Impulse Response

룸 리버브(room reverbation)는 일반적으로 합성 알고리즘에서 내재적으로 모델링됩니다. 하지만 DDSP에서는 룸 어쿠스틱을 따로 합성 후(post-synthesis) 컨볼루션 스텝으로 분리하여 해석 가능성을 향상시킵니다.

실제와 같은 룸 어쿠스틱의 IR은 수 초에 해당하는 길이이기 때문에 수 만에서 수십 만 타임스텝의 커널 사이즈를 사용해야 합니다. 특히 컨볼루션 연산은 $\small \mathcal{O}(n^3)$ 스케일이기 때문에 이러한 큰 커널 사이즈를 구현하여 연산하는 것은 현실적으로 불가능합니다. 따라서 리버브를 주파수 도메인에서 곱하는 것으로 신경망을 구현하여 $\small \mathcal{O}(n \log n)$ 스케일로 축소시키고 마지막에 IDFT를 적용하여 시간 도메인으로 되돌려줍니다.

<br><br>

## DDSP Autoencoder

일반적인 오토인코더는 인코더 신경망 $\small f_{\text{enc}}(\cdot)$가 입력 $\small x$를 잠재 표현 $\small z=f_{\text{enc}}(x)$로 매핑하고 디코더 신경망 $\small f_{\text{dec}}(\cdot)$가 직접적으로 입력을 재구성하여(reconstruct) $\small \hat{x} = f_{\text{dec}}(z)$가 되도록 합니다. DDSP 오토인코더도 기본적으로 인코더와 디코더를 사용하고 최종적으로 입력을 재구성하는 것을 목적으로 하지만 분해(decompose)된 잠재 표현과 DDSP 구성 요소들을 사용한다는 점이 다릅니다.

### Encoder

인코더는 세 종류가 있습니다. $\small f$-인코더는 기본 주파수 $\small f(t)$를, $\small l$-인코더는 음량 $\small l(t)$를, $\small z$-인코더는 나머지 정보를 담은 벡터 $\small z(t)$를 출력합니다.

$\small f$-인코더는 지도학습 버전과 비지도학습 버전으로 나누어서 구현합니다. 지도학습 버전에서는 사전학습된 CREPE [(Jong Wook Kim et al., 2018)](https://arxiv.org/abs/1802.06182) 음높이 탐지 모델을 사용하여 오디오 신호의 실제 기본 주파수를 추출합니다. CREPE는 CNN을 기반으로 하여 360개 구간으로 나눠진 주파수에 대한 확률 분포를 출력함으로써 음높이를 탐지하는 모델입니다. 이때 CREPE의 그래디언트는 학습 도중에 흐르지 않도록 프리즈 합니다.

비지도학습 버전에서는 Resnet으로 구현한 신경망에 오디오 신호의 로그 멜 스펙트로그램을 입력으로 넣어줘서 파라미터가 학습되게 합니다. 이 신경망은 128개 구간으로 나눠진 주파수 값에 대한 정규화된 확률 분포를 출력하고 각각의 주파수 값에 대한 가중합으로 최종 기본 주파수 값을 얻습니다.

$\small l$-인코더는 A-weighting을 적용한 파워 스펙트럼의 RMS 값으로 오디오 신호에서 직접적으로 음량을 계산합니다. A-weighting은 사람의 청각 특성에 맞게 로그 스케일로 높은 주파수에서 더 높은 가중치를 갖게 하는 방법입니다. 추출된 음량 시퀀스 벡터는 데이터셋의 평균과 표준편차로 정규화됩니다.

$\small z$-인코더는 오디오 신호에서 첫 30개의 MFCC를 계산하고 GRU에 통과시켜 임베딩 벡터 $\small z(t)$를 만듭니다. $\small z$-인코더는 실험에 따라 선택적으로 사용되고 그 구조는 아래 그림에 요약되어 있습니다.

<p align="center">
    <img src="https://i.ibb.co/BV4kcKp/z-encoder.png" alt="z-encoder" border="0">
</p>

$\small f(t), l(t), z(t)$는 모두 프레임 해상도에서 시간에 따라 변하는 벡터들입니다. 기본적으로는 모두 같은 타임스텝을 갖도록 하이퍼파라미터를 설정해줍니다. 논문에서는 4초 짜리 오디오 신호에 대해서 250개의 타임스텝이 되도록 프레임 크기와 hop 크기를 설정합니다.

### Decoder

디코더에서 먼저 $\small f(t), l(t), z(t)$는 각각 분리된 MLP를 통과한 뒤 채널 차원으로 연결(concatenate)됩니다. 이것이 GRU를 통과한 뒤 $\small f(t)$와 $\small l(t)$의 MLP 출력과 연결되어 최종 MLP를 통과합니다. 마지막으로는 분리된 선형 층을 거쳐서 각각 **Harmonic Audio**와 **Filtered Noise**로 들어가는 프레임 단위의 시퀀스 벡터를 만듭니다. 디코더의 구조는 아래 그림에 묘사되어 있습니다.

<p align="center">
    <img src="https://i.ibb.co/sjf8LNh/decoder.png" alt="decoder" border="0">
</p>

**Harmonic Audio**에는 디코더를 통과하지 않은 $\small f(t)$가 디코더의 출력과 함께 들어갑니다. 디코더에 사용되는 MLP의 구조는 아래 그림과 같습니다.

<p align="center">
    <img src="https://i.ibb.co/GvCHwq8/decoder-mlp.png" alt="decoder-mlp" border="0">
</p>

### Model Size

DDSP 모델의 파라미터는 GANSynth나 WaveRNN [(Lamtharn Hantrakul et al., 2019)](https://archives.ismir.net/ismir2019/paper/000063.pdf), WaveNet autoencoder [(Jesse Engel et al., 2017)](https://proceedings.mlr.press/v70/engel17a.html)에 비해 훨씬 적은 수의 파라미터를 갖습니다. 더 작은 모델로도 실험을 했을 때 생성되는 오디오의 현실성이 좀 더 떨어지긴 했지만 그래도 높은 품질을 가졌다고 합니다. 모델 크기 비교는 아래 표에 나와 있습니다.

<p align="center">
    <img src="https://i.ibb.co/qmf9Tw3/model-size.png" alt="model-size" border="0">
</p>

<br><br>

## Multi-Scale Spectral Loss

학습의 손실 함수로는 spectral loss를 기반으로 한 multi-scale spectral loss를 제안합니다. 먼저 원래 오디오와 합성된 오디오에 대하여 스펙트로그램의 크기(magnitude) $\small S_i$와 $\small hat{S}_i$를 각각 계산합니다. 그 뒤 FFT 크기 $\small i$에 대한 손실 함수를 다음과 같이 정의합니다.

<br>
<center> $ L_i = \lVert S_i - \hat{S}_i \rVert_{1} + \alpha \lVert \log S_i - \log \hat{S}_i \rVert_{1} $ </center>
<br>

$\small \alpha$는 가중치 항으로 논문의 실험에서는 1.0의 값을 사용했습니다. 전체 손실 함수는 여러 개의 $\small i$에 대한 합으로 $\small \sum_{i} L_i$가 됩니다. 논문의 실험에서는 FFT 크기로 (2048, 1024, 512, 256, 128, 64)를 사용하고 STFT의 중첩은 75%로 설정했습니다. 이러한 손실 함수 설계는 여러 다른 공간-시간적(spatial-temporal) 해상도에서의 실제 오디오와 합성된 오디오의 차이를 모두 다룰 수 있게 해줍니다.

<br><br>

## Datasets

실험에 사용한 데이터셋은 NSynth와 [(Jesse Engel et al., 2017)](https://proceedings.mlr.press/v70/engel17a.html) Solo violin입니다. NSynth는 어쿠스틱 악기와 24-84 범위의 음높이에 해당하는 70,379개의 샘플로 이루어진 일부 데이터셋만 사용하고 80/20으로 트레인/테스트 스플릿을 합니다. NSynth 데이터셋에 대한 실험에서는 $\small z(t)$ 인코더를 사용하고 특정한 경우를 제외하고는 리버브 모듈을 사용하지 않습니다.

Solo violin 데이터셋은 13분 짜리 바이올린 솔로 연주로 이루어진 데이터셋입니다. 연주자는 한 명이고 동일한 공간 환경을 가지고 있습니다. NSynth와 동일하게 16 kHz의 샘플 레이트로 각각의 샘플이 4초 길이가 되도록 나누어져 있습니다. 이 실험에서는 $\small z(t)$ 인코더를 사용하지 않고 리버브 모듈을 추가합니다. 리버브의 IR은 4초 길이에 해당하고 전부 동일한 공간 환경에서 녹음된 데이터셋이기 때문에 모든 데이터 샘플에 대해서 동일한 파라미터 값을 갖도록 학습됩니다.

<br><br>

## 실험

실험에 대한 데모 샘플들은 구글 마젠타의 [DDSP 프로젝트 페이지](https://magenta.tensorflow.org/ddsp)에서 들어볼 수 있습니다.

### High-Fidelity Synthesis

Solo violin 데이터셋의 한 샘플에 대한 분해와 재구성 과정이 아래 그림에 나타나 있습니다.

<p align="center">
    <img src="https://i.ibb.co/R3SdBLT/solo-violin-reconstruction.png" alt="solo-violin-reconstruction" border="0">
</p>

합성된 오디오가 원래와 거의 차이가 없고 각각의 추출된 구성 요소들이 의미 있는 값들을 가지고 있는 것을 볼 수 있습니다. NSynth 데이터셋에 대한 정량적인 비교 결과는 아래 표에 나와 있습니다.

<p align="center">
    <img src="https://i.ibb.co/Bq2Zyrc/nsynth-result.png" alt="nsynth-result" border="0">
</p>

Loudness $\small (L_1)$은 원래 오디오와 합성된 오디오에서 각각 추출된 음량 벡터 사이의 L1 거리입니다. 이 음량에 대한 L1 거리는 학습의 목적 함수로 역전파 되지 않기 때문에 평가 지표로 사용하기에 적절합니다. F0 $\small (L_1)$은 같은 방법으로 기본 주파수에 대해 MIDI 공간에서 계산된 값입니다. 음높이 탐지에 대한 신뢰도가 0.85 이상인 구간에서만 계산되었습니다. F0 Outliers는 신뢰도가 0.85 아래인 샘플들의 비율입니다. 이러한 샘플들은 대부분 음이나 하모닉 요소들이 없는 노이즈에 해당하고 아웃라이어 값이 낮을 수록 더 좋은 성능의 모델이 됩니다.

비지도학습 버전의 DDSP는 오디오 신호로부터 실제값 없이 F0 정보를 추출하는 법을 학습해야 하는데 지도학습 버전의 DDSP 만큼은 아니지만 WaveRNN에 비해서 더 좋은 성능을 보여줍니다. 하지만 이 모델을 학습시킬 때에는 CREPE의 다섯 번째 맥스 풀링 층의 출력에 대한 합성된 오디오와 실제 데이터의 L1 거리를 손실 함수로 추가하여 학습했다고 합니다. 이렇게 되면 CREPE에서 추출되는 특징(feature)에 주로 음높이를 탐지하기 위한 정보가 담겨 있을 것이기 때문에 비지도학습보다는 지도학습에 더 가까운 것 같다는 생각입니다.

### Independent Control of Loudness and Pitch

DDSP는 해석 가능한 구조를 목표로 설계되었기 때문에 각각의 요소들에 대해 독립적으로 제어가 가능한지 보여주는 실험 결과가 중요합니다. 먼저 음량, 음높이, 음색에 대한 보간 실험 결과가 아래 그림에 나와 있습니다.

<p align="center">
    <img src="https://i.ibb.co/k2c5NLN/interpolation.png" alt="interpolation" border="0">
</p>

실선은 원래 데이터, 점선은 합성된 오디오이고 두 데이터 사이의 $\small f(t), l(t), z(t)$에 해당하는 벡터를 보간하여 생성한 결과입니다. 그림에도 표현되어 있지만 데모 샘플을 들어보는 것이 결과를 이해하는 데에 더 도움이 됩니다.

또한 학습할 때 데이터셋에서 보지 못한 음높이에 대한 외삽 실험 결과도 있습니다. 아래 그림은 Solo violin 데이터셋에서 보지 못한 더 낮은 음높이로 오디오를 생성한 결과입니다. 실제로 들어보면 첼로와 같은 소리가 납니다.

<p align="center">
    <img src="https://i.ibb.co/BTXV4wJ/extrapolation.png" alt="extrapolation" border="0">
</p>

### Dereverberation and Acoustic Transfer

아래 그림의 왼쪽은 Solo violin 데이터셋에 대해 학습된 모델에서 마지막 리버브 모듈을 통과하기 전의 오디오를 사용하여 리버브를 제거한 결과입니다.

<p align="center">
    <img src="https://i.ibb.co/VHPdk2w/reverb-result.png" alt="reverb-result" border="0">
</p>

또한 오른쪽 그림은 Solo violin 데이터셋에서 학습된 리버브 모듈을 노래하는 오디오에 적용하여 비슷한 리버브 효과를 적용한 것입니다. 시각화된 스펙트로그램에서도 차이가 잘 나타나지만 역시 직접 데모 샘플을 들어보는 것이 더 좋습니다.

### Timbre Transfer

아래 그림은 Solo violin에 대해 학습된 DDSP 모델에 노래하는 오디오에서 추출된 기본 주파수와 음량 특성을 넣어줘서 음색 전이를 한 결과입니다. 데모 샘플을 들어보면 노래할 때 생기는 숨소리나 떨림 같은 미묘한 특징들도 바이올린의 음색으로 나타나는 것을 알 수 있습니다.

<p align="center">
    <img src="https://i.ibb.co/2WS7FGT/timbre-transfer.png" alt="timbre-transfer" border="0">
</p>

<br><br>

## Reference

[Jesse Engel, Lamtharn Hantrakul, Chenjie Gu and Adam Roberts. DDSP: Differentiable Digital Signal Processing. In ICLR, 2020.](https://openreview.net/forum?id=B1x1ma4tDr)

[Official source code of DDSP](https://github.com/magenta/ddsp)