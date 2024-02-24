---
layout: post
title: "[논문 리뷰] SING: Symbol-to-Instrument Neural Generator"
image: https://i.ibb.co/5G78rC0/thumbnail.png
date: 2024-02-03
tags: 
categories: paper-review
use_math: true
---

<br><br>

## 논문 개요

SING은 Facebook AI Research에서 개발한 음향 합성 모델로 2018년 NeurIPS에 발표되었습니다. 이 모델은 악기, 음높이(pitch), 속도(velocity)가 레이블로 주어진 데이터셋에 대해 학습한 뒤 주어진 조건에 해당하는 악기 소리를 생성합니다.

이전의 대표적인 음악 생성 모델인 WaveNet과 [(van den Oord et al., 2016)](https://arxiv.org/abs/1609.03499) SampleRNN은 [(Soroush Mehri et al., 2016)](https://openreview.net/forum?id=SkxKPDv5xl) 자기회귀적(autoregressive) 방식을 취하기 때문에 학습과 생성 속도가 느린 단점이 있습니다. 또한 오디오 샘플을 카테고리형 분포로 다루기 때문에 현실적으로 연산이 가능하려면 정보의 손실을 감수하더라도 16 bit 샘플을 256개 카테고리의 8 bit로 만드는 것과 같은 압축이 필요합니다.

반면 SING은 오디오 샘플을 연속적인 값으로 처리하며 프레임 단위로 1024 개의 샘플들을 한번에 생성합니다. 손실함수로는 생성된 파형(waveform)에 STFT를 적용하여 얻어진 로그 스펙트로그램에 대한  spectral loss를 새롭게 정의합니다. 모델 구조는 오토인코더를 사전학습(pre-train) 시킨 뒤, 학습된 인코더로 나온 임베딩을 타겟으로 하여 오디오 생성을 위한 LSTM과 디코더를 파인튜닝하는 형태입니다. 아래의 그림에 전체 모델 구조가 요약되어 있습니다.

<p align="center">
    <img src="https://i.ibb.co/hFFDb23/model.png" alt="model" border="0">
</p>

<br><br>

## The Spectral Loss for Waveform Synthesis

이전의 음향 합성 모델에 주로 사용되던 분류 손실(classification loss)의 단점은 양자화로 인한 해상도의 한계입니다. SING은 각각의 오디오 샘플을 연속적인 값으로 예측하고 주파수 도메인에서 타겟 신호와의 거리를 계산합니다.

### Mean Square Regression on the Waveform

SING이 사용하는 spectral loss를 살펴보기에 앞서 고려할 수 있는 다른 단순한 손실함수를 생각해보겠습니다. 시간 도메인에서 예측된 파형과 타겟 파형의 MSE를 직접적으로 계산하는 것으로 식은 다음과 같습니다.

<br>
<center> $ \text{L}_{\text{wav}}(x, \hat{x}) := \lVert x - \hat{x} \rVert^2 $ </center>
<br>

MSE는 약간의 값 차이로도 큰 에러를 발생시키기 때문에 로그 형태로 인지되는 사람의 청각 특성에 잘 부합하지 않다는 것을 직관적으로 알 수 있습니다. 논문에서는 이러한 MSE를 비교군으로 사용해서 spectral loss와의 성능 차이를 비교합니다.

### Spectral Loss

시간 도메인에서 바로 MSE를 계산하는 대신, STFT를 적용한 뒤 주파수 도메인에서 절대값의 로그값에 대해 차이를 계산할 수 있습니다. 로그 스펙트로그램은 다음과 같은 식으로 계산됩니다.

<br>
<center> $ l(x) := \log (\epsilon + |\text{STFT}[x]|^2) $ </center>
<br>

이때 $\small \epsilon$은 $\small 0$에 가까운 낮은 에너지를 갖는 주파수 성분들이 합쳐져서 노이즈로 작용하는 것을 방지하기 위한 역할입니다. $\small \epsilon$ 값이 너무 크면 오차가 커지기 때문에 적당한 값으로 설정해야 하는데 논문에서는 실험적으로 선택한 $\small \epsilon = 1$을 사용했습니다. STFT는 프레임 크기 1024와 hop length 256으로 75%가 겹쳐지게 적용하였습니다. 스펙트로그램은 값들이 넓은 범위에서 분포하는 특성을 가지고 있기 때문에 spectral loss로는 MSE 대신에 다음과 같이 L1 norm을 사용합니다.

<br>
<center> $ \text{L}_{\text{stft}, 1} (x, \hat{x}) := \lVert l(x) - l(\hat{x}) \rVert_1 $ </center>
<br>

Spectral loss를 구성하는 STFT와 Fourier coefficient에 대한 절대값 연산은 모두 미분 가능하기 때문에 엔드-투-엔드로 역전파(backpropagation)를 통해 모델을 학습시킬 수 있습니다.

### Non Unicity of the Waveform Representation

우리가 소리를 지각할 때에는 전체 파동의 모양이 같기만 하면 타임스텝 0에서 어떤 값을 갖는지에 따라서 영향을 받지 않고 같은 소리로 인식합니다. 따라서 같은 악기로 같은 음높이의 음을 연주하더라도 타임스텝 0에서의 값은 매번 달라질 수 있습니다. 이를 논문에서는 위상(phase)이라고 표현하고 있지만 신호 내의 harmonic partial 혹은 프레임 간의 위상과 헷갈릴 수 있으므로 phase보다는 위상 오프셋(phase offset)이 더 맞는 표현으로 보입니다.

만약 시간 도메인에서의 파형에 대한 MSE 손실을 사용한다면, 모델은 학습 데이터 내의 파형과 비슷한 모양의 파형을 생성하기 위해 각각의 음높이마다 오프셋 정보도 기억해서 학습하게 됩니다. 즉, 파형이 타임스텝 0에서 어떤 값을 가질지도 고려하여 학습합니다. 따라서 이러한 오프셋 값을 기억하기 위해 추가적인 자원을 사용하게 되는 단점이 있습니다. 또한 학습 데이터에 없는 음높이에 대해서는 모델이 이러한 오프셋에 대한 정보를 예측할 수 없기 때문에 성능 저하로 이어질 수 있습니다.

Spectral loss의 경우에는 주파수 도메인에서 복소수 Fourier coefficient의 절대값을 사용하기 때문에 모델이 생성하는 파형의 오프셋이나 데이터셋에서 오프셋의 변화에 민감하지 않습니다. 이러한 점은 학습 효율성과 학습 데이터에서 보지 못한 조건에서의 안정성 측면에서 장점으로 작용합니다.

하지만 이와 관련된 문제들도 발생할 수 있습니다. 예를 들어 STFT를 적용하여 만들어진 프레임들 간에 위상이 서로 달라도 spectral loss 계산에는 고려되지 않기 때문에 잠재적인 에러를 발생시킬 수 있습니다. 논문의 실험 결과에서는 이로 인한 치명적인 문제는 발생하지 않았는데, 본문에서는 프레임을 중첩하기(overlap) 때문에 프레임 간에도 어느 정도 위상이 맞는 것이 결국 spectral loss를 최소화 하기 위한 해가 되기 때문이라고 추정합니다.

사실 이 부분에서 이론적으로는 위상을 고려하지 않는 spectral loss의 장점보다도 단점이 명확해 보이는데 로그 스펙트로그램의 절대값만을 손실함수 계산에 사용하는 것이 아니라 절대값을 취하기 전의 위상 정보도 손실함수에 같이 포함시키는 방법을 채택하면 결과가 어떨지 궁금해집니다.

<br><br>

## Model Architecture

SING 모델은 인코더, 디코더, 그리고 LSTM sequence generator로 구성되어 있습니다. 인코더와 디코더는 오토인코더를 구성하여 신호를 중첩된 프레임 단위로 잠재 공간에 임베딩하고 다시 원래의 신호로 복원합니다. 이때 입력은 학습 데이터셋에 있는 오디오 신호의 파형 시퀀스입니다.

LSTM sequence generator는 원핫 인코딩된 속도 $\small V$, 악기 $\small I$, 음높이 $\small P$와 타임스텝 $\small T$에 대한 임베딩 벡터 $\small (u_V, v_I, w_P, z_T)$를 입력으로 받아 중첩된 프레임 단위로 잠재 벡터 시퀀스를 생성합니다. 타임스텝 $\small T$는 프레임 단위로 현재 프레임이 몇 번째인지를 나타내는 값이며 $\small u_V \in \mathbb{R}^2$, $\small v_I \in \mathbb{R}^{16}$, $\small w_P \in \mathbb{R}^8$, $\small z_T \in \mathbb{R}^{4}$는 연결되어(concatenated) 입력으로 들어갑니다.

LSTM sequence generator에 의해 임베딩된 벡터 시퀀스들은 오토인코더의 디코더를 통해 신호 파형으로 디코딩됩니다. 맨 위의 논문 개요 섹션에 있는 모델 구조 그림은 SING의 LSTM sequence generator와 디코더 부분을 나타낸 것입니다.

<br><br>

## LSTM Sequence Generator

Sequence generator는 은닉 상태 차원(hidden state dimension)이 1024이고 층(layer)이 3개인 LSTM입니다. LSTM의 입력은 자기회귀적으로 들어가는 것이 아니라 연결된 임베딩 벡터 $\small (u_V, v_I, w_P, z_T)$만이 사용됩니다. 논문에서는 이전 스텝의 출력이 다음 입력으로 임베딩 벡터와 같이 연결되어 들어가는 자기회귀적 모델도 실험해봤지만 성능이 더 나빴다고 합니다.

LSTM의 출력 시퀀스는 하나의 완전연결층(fully-connected layer)을 거쳐서 잠재 벡터 시퀀스 $\small s(V, I, P)_T \in \mathbb{R}^D$가 됩니다 $(\small \forall 1 \leq T \leq N)$. 이때 $\small N$은 전체 프레임 개수입니다.

<br><br>

## Convolutional Decoder

잠재 벡터 시퀀스 $\small s(V, I, P)$는 컨볼루션 층(convolutional layer)을 통해 파형으로 디코딩 됩니다. 첫번째 층은 커널 사이즈가 $\small 9$이고 스트라이드가 $\small 1$이며, 두 번째와 세 번째 층은 커널 사이즈가 $\small 1$인 $\small 1 \times 1$ 컨볼루션입니다.

마지막 층은 스트라이드가 $\small 256$이고 커널 사이즈가 $\small 1024$인 전치 컨볼루션(transposed convolution)입니다. 따라서 최종 출력은 길이가 $\small 1024$인 프레임이 $\small 256$의 간격을 두고 중첩되어 있는 형태가 됩니다. 신호의 전체 길이는 패딩을 통해서 맞춰줍니다.

최종 파형이 중첩되어 있는 프레임의 형태로 출력되기 때문에 왜곡(artifact)을 줄여주기 위해 마지막 층의 컨볼루션 필터의 가중치(weight)에는 squared Hann window를 곱해줍니다.

<br><br>

## Training Details

전체 학습은 오토인코더의 사전학습, LSTM의 초기 학습, 그리고 LSTM과 디코더의 파인튜닝 세 단계로 이루어집니다. 세부적인 모델 디자인과 하이퍼파라미터는 논문에 더 자세히 나와 있습니다.

### Initialization with an Autoencoder

오토인코더의 인코더는 디코더를 반대로 뒤집은 모양입니다. 입력 신호는 squared Hann window가 적용된 커널 사이즈 $\small 1024$와 스트라이드 $\small 256$의 컨볼루션 층을 시작으로 세 개의 $\small 1 \times 1$ 컨볼루션 층을 지납니다.

모델을 학습시킬 때 먼저 인코더와 디코더를 같이 사전학습 시킵니다. 이때 손실함수는 원래 신호와 복원된(reconstructed) 신호의 MSE를 사용합니다.

### LSTM Training

오토인코더의 사전학습이 끝나면 인코더에 의해 임베딩된 잠재 벡터 시퀀스를 타겟으로 하여 LSTM을 학습시킵니다. 이때는 인코더를 동결시키고(freeze) LSTM만 그래디언트가 흐르게 합니다.

손실함수로는 인코더의 출력 $\small e(x)$와 LSTM의 출력 $\small s(V, I, P)$ 사이의 MSE를 사용합니다. 이 단계의 LSTM 학습에는 truncated backpropagation through time을 적용합니다.

### End-to-end Fine Tuning

인코더에 의해 초기 학습된 LSTM은 디코더와 결합되어 엔드투엔드로 파인튜닝됩니다. 파인튜닝 단계에서 인코더는 사용되지 않고 LSTM에는 truncated backpropagation through time이 적용되지 않습니다.

손실함수로는 타겟 신호와 LSTM-디코더에 의해 생성된 신호에 모두 STFT를 적용하여 얻어진 로그 스펙트로그램 사이의 L1 loss, 즉 논문에서 정의한 spectral loss를 사용합니다.

LSTM을 인코더에 대해 먼저 초기 학습시키지 않고 바로 디코더와 결합하여 학습시킨 경우에는 모델이 제대로 수렴하지 않았다고 합니다.

<br><br>

## 실험

실험에 사용한 데이터셋은 NSynth입니다 [(Jesse Engel et al., 2017)](https://proceedings.mlr.press/v70/engel17a.html). NSynth는 각각의 악기로 일정한 음을 4초 동안 지속되게 연주한 오디오 신호들로 이루어져 있습니다. 

각각의 신호는 서로 다른 1006개의 악기를 나타내는 $\small I \in \text{\{0, ..., 1005\}}$, 5개의 속도를 나타내는 $\small V \in \text{\{0, ..., 4\}}$, 121개의 음높이를 나타내는 $\small P \in \text{\{0, ..., 120\}}$를 인덱스로 하여 분류됩니다. 신호의 샘플 레이트는 16,000 Hz로 각각의 4초 짜리 샘플은 벡터 $\small x_{V, I, P} \in [-1, 1]^{64000}$으로 표현됩니다.

각각의 악기들에서 10%의 음높이에 대한 신호는 테스트셋으로 분리하여 사용합니다. NSynth 데이터셋 내에는 같은 조합의 악기, 음높이를 갖는 신호는 중복되지 않기 때문에 이렇게 하면 학습할 때 보지 못한 악기와 음높이 조합에 대한 일반화 능력을 평가할 수 있습니다.

실험에 대한 데모 오디오 샘플들은 [Facebook Research 웹페이지](https://research.facebook.com/publications/sing-symbol-to-instrument-neural-generator/)에서 들어볼 수 있다고 논문에 나와 있지만 현재는 접속이 안되어 확인할 수가 없습니다.

### Generalization Through Pitch Completion

전체 실험 결과는 아래 표에 나와 있습니다. 테스트셋에 대한 수치를 통해 학습할 때 보지 못했던 악기와 음높이 조합에 대한 일반화 능력을 평가할 수 있습니다.

<p align="center">
    <img src="https://i.ibb.co/vs3XrfY/result.png" alt="result" border="0">
</p>

먼저 사전학습된 오토인코더만 사용한 경우와 SING 전체 모델을 사용한 경우를 비교해 볼 수 있습니다. LSTM이 사전학습된 인코더에 의해 생성된 임베딩을 타겟으로 학습되기 때문에 오토인코더만 사용한 것은 LSTM가 완벽하게 인코더를 모방하도록 학습된 최대 성능과 같습니다. 오토인코더만 사용할 때에는 waveform loss와 spectral loss 모두 테스트셋에 대해서도 트레인셋과 비슷한 성능을 보여줍니다. 이 경우에는 입력으로 오디오 신호가 들어가기 때문에 실제로 생성 모델의 용도로 사용될 수는 없습니다.

LSTM sequence generator를 사용한 SING 전체 모델의 경우에는 오토인코더에 비해 성능이 감소하긴 하지만 그래도 spectral loss를 사용했을 때 waveform loss에 비해 성능 감소 정도가 훨씬 작습니다. 또한 시간 임베딩 $x_T$를 사용하는 것이 상당한 영향을 미칩니다.

<p align="center">
    <img src="https://i.ibb.co/JCp2ryp/rainbowgram.png" alt="rainbowgram" border="0">
</p>

위의 그림은 테스트셋에 대한 rainbowgram으로 선의 강도는 파워 스펙트럼의 로그 절대값을 나타내고 색은 위상을 나타냅니다. 스펙트로그램과 마찬가지로 가로축은 시간, 세로축은 주파수입니다. 그림의 왼쪽부터 오른쪽까지 각각 실제값(ground truth), WaveNet autoencoder [(Jesse Engel et al., 2017)](https://proceedings.mlr.press/v70/engel17a.html), SING with spectral loss, SING with waveform loss, 그리고 SING without the time embedding에 해당합니다.

네 번째 그림의 waveform loss의 경우에는 테스트셋의 처음 보는 음높이에 대해 위상 오프셋을 예측할 수가 없으므로 위상이 아예 맞지 않는 것이 보입니다. Spectral loss를 사용했을 때에는 위상을 꽤 정확하게 맞추고 WaveNet autoencoder보다도 나은 성능을 보여줍니다.

### Evaluation of Perceptual Quality: Mean Opinion Score

MOS 테스트는 100개의 테스트셋 샘플에 대해 진행되었습니다. 실험 참가자는 60명이며 오디오 샘플이 얼마나 자연스럽게 들리는지 1점부터 5점까지 평가합니다. 비교대상인 WaveNet autoencoder는 해당 논문 저자들이 공개한 사전학습된 모델을 사용했는데 이 경우에는 테스트셋을 나누는 방법이 다르므로 테스트 샘플들도 모델을 학습할 때 사용되었습니다. 결과는 아래 표에 나와 있습니다.

<p align="center">
    <img src="https://i.ibb.co/y5F5Mpp/mos-test.png" alt="mos-test" border="0">
</p>

SING이 WaveNet autoencoder에 비해 좋은 성능을 보여주고 실제값과도 격차가 크지 않습니다. MOS 테스트 결과 외에 속도와 메모리 측면에서 비교한 결과도 나와 있는데 SING은 자기회귀적 방식을 사용하지 않기 때문에 상당한 이점을 가지고 있습니다.

### ABX Similarity Measure

생성된 오디오 샘플의 자연스러움을 평가하는 것만으로는 모델이 정말로 주어진 악기, 음높이, 속도 조건에 맞게 소리를 생성했는지 판단할 수 없습니다. 따라서 ABX 테스트도 진행하였는데 실험 참가자들이 원래 샘플과 모델들이 생성한 샘플을 각각 듣고 어떤 모델이 원래와 더 비슷한 소리를 만들었는지 평가하는 것입니다.

10명의 참가자가 100개의 샘플에 대해 테스트를 진행했고, 69.7%의 참가자가 WaveNet autoencoder에 비해 SING이 더 낫다고 평가하였습니다.

<br><br>

## Reference

[Alexandre Défossez, Neil Zeghidour, Nicolas Usunier, Léon Bottou and Francis Bach. SING: Symbol-to-Instrument Neural Generator. In NeurIPS, 2018.](https://proceedings.neurips.cc/paper/2018/hash/56dc0997d871e9177069bb472574eb29-Abstract.html)

[Official source code of SING](https://github.com/facebookresearch/SING)