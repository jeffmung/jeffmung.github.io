---
layout: post
title: "[논문 리뷰] MIDI-DDSP: Detailed Control of Musical Performance via Hierarchical Modeling"
image: https://i.ibb.co/gD3dznW/thumbnail.png
date: 2024-02-23
tags: 
categories: paper-review
use_math: true
---

<br><br>

## 논문 개요
<!-- excerpt-start -->
MIDI-DDSP는 DDSP를 [(Jesse Engel et al., 2020)](https://openreview.net/forum?id=B1x1ma4tDr) 기반으로 미디 노트 시퀀스로부터 다양한 악기 소리와 연주의 표현을 반영한 실제에 가까운 음악을 생성해내는 모델입니다. DDSP의 장점인 음높이, 음량 등의 특징들을 미분 가능한 연산들로 오디오 합성에 활용하여 사용자가 제어할 수 있도록 한다는 것에 더해서 비브라토, 크레센도 등의 연주 표현에 관련된 특징들도 제어하는 것이 가능합니다. 논문에서는 아래 그림과 같이 기존 연구들이 제어 가능성과 자연스러운 소리 둘 중 한 가지에 주로 집중했다면 MIDI-DDSP는 두 가지 관점 모두에서 높은 성능을 보여준다는 것을 강조합니다.

<p align="center">
<img src="https://i.ibb.co/ykcWzzB/mididdsp-strengths.png" alt="mididdsp-strengths" border="0">
</p>

이러한 강점을 나타낼 수 있도록 MIDI-DDSP 모델은 계층적인 구조로 설계되어 있습니다. 가장 아래 모델은 DDSP로 오디오로부터 음높이, 진폭, 하모닉 분포, 노이즈 등의 소리 합성에 필요한 파라미터들을 추론하고 반대로 다시 소리를 합성하는 오토인코더입니다. 중간 레벨에는 합성 파라미터와 연주 표현 특징들 사이에서 추출과 생성을 학습하는 모델이 있습니다. 가장 위 레벨에는 음표(note)로부터 연주 표현을 생성해내는 방법을 학습하는 모델이 있습니다. 이러한 계층 구조는 아래 그림에 표현되어 있습니다.

<p align="center">
<img src="https://i.ibb.co/b7LRGLm/hierearchy.png" alt="hierearchy" border="0">
</p>

<br><br>

## DDSP Synthesis and Inference

맨 아래 레벨에서 사용하는 DDSP의 구조는 기본적으로는 기존 논문의 모델과 동일하고 약간의 개선이 더해졌습니다. 기존 DDSP 모델에 대한 자세한 설명은 [DDSP 논문 리뷰 포스트](https://jeffmung.github.io/2024/02/16/paper-review-7-ddsp/)의 링크로 대체합니다.

MIDI-DDSP 논문에서는 미분 가능하게 연산되어 하모닉 플러스 노이즈 모델에 입력으로 들어가는 기본 주파수(fundamental frequency) $\small f_0(t) \in \mathbb{R}^{1 \times t}$, 전역(global) 진폭 $\small a(t) \in \mathbb{R}^{1 \times t}$, 하모닉 분포 $\small \boldsymbol{h}(t) \in \mathbb{R}^{60 \times t}$, 그리고 노이즈 필터 크기(magnitude) $\small \boldsymbol{\eta}(t) \in \mathbb{R}^{65 \times t}$ 를 합성 파라미터(synthesis parameter)라고 명명합니다. 기존 논문과 마찬가지로 변형된 시그모이드를 사용하고 프레임 크기는 64, 샘플 레이트는 16000 Hz로 설정합니다.

또한 리버브 모듈을 사용하는데 서로 다른 악기에 대해서는 다른 리버브 모듈 파라미터를 사용합니다. 리버브 모듈의 학습 가능한 임펄스 반응은 48000개의 샘플 포인트를 갖도록 설정되어 있는데 후반부는 음색에 거의 영향을 주지 않으면서 소리의 잔향만 길게 남기기 때문에 다음과 같이 16000 샘플 이후에는 지수적으로 감소하도록 제한합니다.

<br>
\begin{equation}
\left.
\begin{aligned}
\text{IR}^{\prime}(t) &= \text{IR}(t), \qquad\qquad\qquad & 0 \leq t \leq 16000 \newline
\text{IR}^{\prime}(t) &= \text{IR}(t) \cdot (-4(t-16000)), & 16000 < t \leq 48000
\end{aligned}
\quad \right\\}
\end{equation}
<br>

여기서 $\small \text{IR}(t)$ 는 기존 임펄스 반응이고 $\small \text{IR}^{\prime}(t)$ 는 지수 감소가 적용된 후의 임펄스 반응입니다.

기존 DDSP에서 가장 크게 바뀐 점은 CREPE로 [(Jong Wook Kim et al., 2018)](https://ieeexplore.ieee.org/abstract/document/8461329/) 기본 주파수를 추정하고 A-weighting으로 음량을 추출하는 것에 더해서 로그 멜 스펙트로그램으로부터 CNN으로 추가적인 정보를 추출해낸다는 것입니다. MFCC로부터 추가 정보를 추출하는 기존 모델의 $\small z$-인코더는 사용하지 않습니다. 그리고 다중 악기 설정의 실험에서는 64차원의 악기 임베딩이 채널 축으로 같이 연결(concatenate)되어 입력으로 들어갑니다. 이러한 DDSP 추론 모델 구조는 아래 그림에 요약되어 있습니다.

<p align="center">
<img src="https://i.ibb.co/gj1Ct8z/ddsp-inference.png" alt="ddsp-inference" border="0">
</p>

아래 표는 멜 스펙트로그램에서 특징을 추출하는 CNN의 세부 설계입니다. 멜 스펙트로그램은 64개의 주파수 구간으로 나누어져 있는데 CNN 블록과 밀집층(dense layer)을 거쳐서 주파수 축의 64차원이 256차원으로 매핑됩니다. 시간 축으로는 1000개의 프레임이 그대로 유지됩니다.

<p align="center">
<img src="https://i.ibb.co/7NrNg1p/logmelspec-cnn.png" alt="logmelspec-cnn" border="0">
</p>

또한 기존 DDSP에서는 기본 주파수와 음량이 분리된 MLP를 통과하는 것과 달리 MIDI-DDSP에서는 주파수와 음량이 채널 축으로 연결되어 2차원이 된 뒤 하나의 밀집층을 통과하여 256차원으로 매핑됩니다. 그 뒤 다른 특징들과 연결되어 통과하는 bi-directional LSTM은 256의 은닉 차원을 갖습니다. 이는 기존 DDSP가 uni-directional GRU를 사용하는 것과도 약간 다른 점입니다.

Bi-directional LSTM에서 나온 출력은 세 개의 서로 다른 밀집층을 거쳐서 각각 $\small a(t) \in \mathbb{R}^{1 \times t}$, $\small \boldsymbol{h}(t) \in \mathbb{R}^{60 \times t}$, 그리고 $\small \boldsymbol{\eta}(t) \in \mathbb{R}^{65 \times t}$ 가 됩니다.

<br><br>

## Expression Controls

연주 표현에 대한 제어 요소들(expression controls)은 6가지 종류를 정의하여 사용합니다. 이 요소들은 합성 파라미터 $\small \boldsymbol{s}(t) = (f_0 (t), a(t), \boldsymbol{h}(t), \boldsymbol{\eta}(t))$ 로부터 도메인 지식을 반영하여 정의된 함수들을 통해 계산됩니다.

표현 제어 요소들은 음표 단위로 추출됩니다. 예를 들어 $\small i$ 번째 음표 $\small \boldsymbol{n}\_i$ 가 프레임 $\small T_{\text{on},i}$ 에서 시작되고 프레임 $\small T_{\text{off},i}$ 에서 끝나면, 그 음표의 지속 시간은 $\small T_{n,i} = T_{\text{off},i} - T_{\text{on},i}$ 이고 음표가 활성화 되어 있는 프레임은 $\small \tau_i \in \[ T_{\text{on},i}, T_{\text{off},i} \]$ 라고 정의합니다. 그러면 $\small \boldsymbol{n}_i$ 에 대한 표현 제어 요소들은 $\small \boldsymbol{s}(\tau_i)$ 로부터 추출되는 것입니다.

모든 표현 제어 요소 계산에서 전역 진폭 $\small a(\tau)$ 와 노이즈 크기 $\small \boldsymbol{\eta}(\tau)$ 는 다음과 같이 dB 스케일로 변환되어 사용됩니다.

<br>
\begin{equation}
a^{\prime}(\tau) = 20 \log_{10} a(\tau), \qquad\qquad \boldsymbol{\eta}^{\prime} (\tau) = 20 \log_{10} \boldsymbol{\eta} (\tau)
\end{equation}
<br>

### Volume

볼륨은 각 음표의 길이에 대한 평균 진폭으로 구합니다.

<br>
\begin{equation}
\frac{1}{T_n} = \sum_{i=1}^{T_n} a^{\prime}(i)
\end{equation}
<br>

### Volume Fluctuation

볼륨 변동(fluctuation)은 각 음표 내에서 진폭의 표준 편차를 계산하여 구합니다. 아래 식에서 $\small \bar{a^{\prime}}(\tau)$ 는 각 음표 내에서의 평균 진폭을 의미합니다.

<br>
\begin{equation}
\sqrt{\frac{1}{T_n} \sum_{i=1}^{T_n} (a^{\prime}(i) - \bar{a^{\prime}}(\tau))^2}
\end{equation}
<br>

### Volum Peak Position

볼륨 피크 위치는 각 음표 내에서 가장 높은 진폭 값이 나타나는 프레임 위치로 얻어집니다. 이 값은 시간 축에서 $\small \[ 0, 1 \]$ 범위 내로 정규화됩니다. 예를 들어 볼륨 피크 위치가 0이면 디크레센도, 1이면 크레센도를 나타냅니다.

<br>
\begin{equation}
\frac{1}{T_n} \text{argmax}_i a^{\prime}(i) \qquad \forall i \in \[ T_n \]
\end{equation}
<br>

### Vibrato

비브라토는 기본 주파수 시퀀스에 DFT를 적용하여 다음과 같이 계산됩니다.

<br>
\begin{equation}
\max_i \mathcal{F}\\{ f_0(t) \\}_i
\end{equation}
<br>

여기서 $\small \mathcal{F}\\{ \cdot \\}$ 은 DFT 함수를 의미합니다. 계산된 비브라토 값이 3에서 9 Hz 사이이고 길이가 200 ms 보다 큰 경우에만 사용됩니다. 그 외에는 비브라토 값을 0으로 설정합니다. DFT를 적용할 때 $\small f_0(t)$ 는 1000 프레임 길이로 제로 패딩 됩니다.

### Brightness

밝기는 하모닉 분포의 스펙트럴 센트로이드로 정의됩니다. 이 값은 주파수 구간 번호를 기준으로 계산되고 시간에 대한 평균값을 사용합니다.

<br>
\begin{equation}
\frac{1}{T_n} \sum_{i}^{T_n} \sum_{k=1}^{\vert \boldsymbol{h} \vert} k \cdot \boldsymbol{h}^k (i)
\end{equation}
<br>

위 식에서 $\small \boldsymbol{h}^k (i)$ 는 $\small k$ 번째 하모닉 분포를 의미하고 $\small \vert \boldsymbol{h} \vert =60 $ 는 하모닉 분포의 총 개수를 나타냅니다.

### Attack Noise

어택 노이즈는 음표의 시작 부분에서 발생하는 노이즈의 양을 나타냅니다. 많은 악기들은 음표 초반의 몇 밀리초 내에서 많은 양의 노이즈가 발생하고 이는 각 악기의 음색을 구분하는 중요한 특징 중 하나입니다. 이 값은 다음 식과 같이 얻어집니다.

<br>
\begin{equation}
\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^{\vert \boldsymbol{\eta} \vert} {\boldsymbol{\eta}^{\prime}}^{k} (i)
\end{equation}
<br>

여기서 $\small N$은 처음 몇 개의 프레임 안에서 어택 노이즈를 계산할지 결정하는 값이고 40 ms에 해당하는 $\small N=10$ 을 기본값으로 사용합니다. $\small {\boldsymbol{\eta}^{\prime}}^{k} (i)$ 는 $\small i$ 번째 프레임에서 dB 스케일의 노이즈 크기를 나타내며 $\small \vert \boldsymbol{\eta} \vert = 65$ 는 노이즈 필터 개수입니다.

모든 표현 제어 요소들은 $\small \[ 0.0, 1.0 \]$ 사이의 값으로 정규화된 다음 6차원의 벡터로 연결됩니다. 이 값들은 각 노트의 프레임 길이 $\small T_n$ 만큼 반복되어 전체 길이는 다시 $\small \boldsymbol{s}(t)$ 의 길이와 같아집니다.

<br><br>

## Note Sequence

6차원의 표현 제어 요소 시퀀스는 합성 파라미터를 예측하기 전 추가로 음표의 음높이, 음표 시작, 음표 종료, 그리고 음표 내의 위치 인코딩 시퀀스들과 연결됩니다. 연결되는 추가 시퀀스들은 각각 스칼라 값을 갖는 시퀀스이기 때문에 최종적으로는 10차원이 됩니다. 이 10차원 시퀀스를 조건 시퀀스(conditioning sequence)라고 명명하고 **Synthesis Generator**의 입력으로 들어가서 합성 파라미터 $\small \boldsymbol{s}(t)$ 를 예측하는 데 사용됩니다.

음표의 음높이는 MIDI 음높이 단위로 표현되며 전체 MIDI 음높이 개수인 127로 나누어져 정규화 되어 0에서 1 사이의 값을 갖습니다. 음표 시작은 각 음표가 시작되는 프레임의 값이 1이고 나머지는 0인 시퀀스입니다. 음표 종료는 반대로 각 음표가 종료될 때의 값이 1이고 나머지는 0입니다.

음표 내의 위치 인코딩은 각 음표의 길이 $\small T_n$ 내에서 몇 번째 프레임인지를 0부터 1 사이의 값으로 나타냅니다. 예를 들어 어떤 음표가 4개 프레임 동안 지속되고 다음 음표는 2개 프레임 동안 지속되면 그 안의 위치 인코딩은 $\small \[ 0.25, 0.5, 0.75, 1.0, 0.5, 1.0 \]$ 이 됩니다. 트랜스포머에서 사용하는 것처럼 사인과 코사인 값으로 계산되는 위치 인코딩을 사용해도 되지만 논문의 기본 설정으로는 이러한 선형 인덱스를 사용합니다.

이러한 음표에 대한 정보들은 학습된 모델이나 정의된 연산을 통해 추출할 수도 있지만 논문의 실험에서는 데이터셋에 실제값(ground truth)이 주어집니다. 즉, MIDI-DDSP의 계층 구조를 나타내는 이 포스트의 두 번째 그림에 나와 있는 **Note Detection**은 따로 학습하거나 계산하는 것이 아니라 사람이 레이블링 한 실제값을 사용하는 것입니다. 물론 반드시 이런 방법으로 제한되는 것은 아니고 논문에서도 후속 연구에서는 음표 채보(note transcription) 모델 등을 사용할 수 있다고 언급하면서 가능성을 열어두고 있습니다.

<br><br>

## Synthesis Generator

**Synthesis Generator**는 표현 제어 요소와 음표 정보를 입력으로 받아 합성 파라미터를 생성합니다. 구체적으로 입력은 위 섹션에서 설명한 10차원의 조건 시퀀스와 64차원의 악기 임베딩이며, 모델은 GAN 구조로 설계되어 있습니다. Generator에서 합성 파라미터 $\small \boldsymbol{s}(t) = (f_0 (t), a(t), \boldsymbol{h}(t), \boldsymbol{\eta}(t))$ 를 생성하고 그 출력은 discriminator를 통과하여 GAN 손실 함수 계산에 사용됩니다. 아래 그림에 **Synthesis Generator** 모델 구조가 표현되어 있습니다.

<p align="center">
<img src="https://i.ibb.co/vkhxxS6/synthesis-generator.png" alt="synthesis-generator" border="0">
</p>

입력으로 들어가는 조건 시퀀스는 밀집층(dense layer)을 통해 256차원으로 매핑된 뒤 64차원의 악기 임베딩과 연결됩니다. 합성 파라미터 중 $\small f_0(t)$ 는 자기회귀적(autoregressive) RNN을 통해 생성되고 $\small a(t), \boldsymbol{h}(t), \boldsymbol{\eta}(t)$ 은 1D dilated CNN을 통해 생성됩니다.

### $\small f_0$ Generation using the Autoregressive RNN

이 신경망의 목표는 MIDI 음표, 비브라토, 악기 등의 정보가 포함되어 있는 입력 시퀀스를 받아 오디오 합성에 사용될 기본 주파수 $\small f_0$ 시퀀스를 생성하는 것입니다. 출력의 형식은 카테고리형 분포이고 0.01 semitone 단위로 MIDI 음표 번호에 대한 $\small f_0$ 의 오프셋을 $\small \[ -1.00, 1.00 \]$ 범위에서 예측합니다. 즉, 총 카테고리의 수는 201개입니다.

예를 들어 어떤 프레임의 음표가 MIDI A4에 해당하는 $\small f_0^{\text{A4}} = 69$ 의 값을 갖고 있고 해당 오디오 신호의 $\small f_0$ 실제값을 MIDI 단위로 나타낸 것이 $\small f_0^{\text{GT}} = 69.2$ 라고 하겠습니다. 자기회귀적 RNN은 $\small f_0^{\text{GT}} - f_0^{\text{A4}} = 0.20$ 을 예측해야 합니다. 예측된 오프셋은 원래 입력의 MIDI 번호에 더해져 MIDI 단위의 $\small f_0$ 예측값이 됩니다. 이를 이용해 **DDSP Synthesis**에서 오디오를 합성할 때에는 $\small f(n) = 440 \cdot 2^{(n - 69)/12}$ 공식을 통해 Hz 단위로 환산됩니다. 여기서 $\small f(n)$ 은 Hz 단위의 주파수이고 $\small n$ 은 0.01 semitone 단위의 MIDI 번호입니다.

RNN은 구체적으로 한 층 짜리 Bi-LSTM과 두 층짜리 GRU로 구성되어 있습니다. 각 신경망들은 모두 256의 은닉 차원을 갖고 있습니다. Bi-LSTM이 인코더, GRU가 디코더의 역할을 하여 Bi-LSTM에서 추출된 512차원의 문맥(context) 벡터가 201차원의 원핫 벡터인 이전 타임스텝의 최종 출력과 연결되어 다음 GRU 입력으로 들어가는 자기회귀적 방식으로 동작합니다. 학습 시에는 실제값을 사용하여 교사 강요(teacher forcing) 하고 추론 시에는 $\small p=0.95$의 top-p 샘플링을 사용하여 주파수 곡선에 비현실적인 급격한 변화가 생기지 않도록 방지합니다.

### Generating the Rest of the Synthesis Parameters

1D dilated CNN의 입력으로는 256차원의 조건 시퀀스, 64차원의 악기 임베딩, 64차원으로 임베딩 된 $\small f_0(t)$ 가 연결되어 들어갑니다. 이 신경망은 4개의 1D dilated CNN 스택으로 구성되어 있습니다. 그 구조는 아래 표에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/znyLkfD/dilated-cnn.png" alt="dilated-cnn" border="0">
</p>

출력 시퀀스는 126차원으로 각각 1차원의 $\small a(t)$, 60차원의 $\small \boldsymbol{h}(t)$, 65차원의 $\small \boldsymbol{\eta}(t)$ 예측값에 해당합니다.

### Discriminator for the Synthesis Generator

Discriminator는 세 개를 사용하는데 각각 다른 스케일로 다운샘플링 된 입력으로부터 학습하여 여러 시간 해상도의 특징들을 추출할 수 있도록 설계되어 있습니다. 다운샘플링은 평균 풀링을 통해 이루어집니다. 예를 들어 프레임 개수가 1000개인 입력이 들어가면 첫 번째 discriminator는 그대로 길이 1000의 입력을 사용하고 두 번째 discriminator의 입력은 평균 풀링을 한 번 하여 길이가 500이 됩니다. 세 번째 discriminator는 평균 풀링을 한 번 더 한 250 길이의 입력이 들어갑니다.

각각의 discriminator는 4개의 블록으로 구성되어 있습니다. 각 블록에서 추출된 256차원 특징 맵(feature map)은 따로 저장되어 손실 함수 계산에 사용됩니다. 마지막에는 선형(linear) 층을 통해 256차원이 1차원으로 매핑되어 실제값이면 1, 예측값이면 0인 스칼라 값이 되도록 합니다. Discrminator 구조는 아래 그림에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/dWqL1Yg/discriminator.png" alt="discriminator" border="0">
</p>

### Synthesis Generator Losses

**Synthesis Generator** 학습의 손실 함수는 예측된 합성 파라미터로부터 합성된 오디오에 대한 멀티스케일 스펙트럼 손실(multi-scale spectral loss) $\small \mathcal{L}\_{spec}$, 예측된 $\small f_0$ 에 대한 크로스 엔트로피 손실 $\small \mathcal{L}_{CE(f_0)}$, LSGAN 손실 $\small \mathcal{L}\_{lsgan}$, 그리고 특징 맵 손실 $\small \mathcal{L}\_{fm}$ 로 구성되어 있습니다. 각각의 손실 함수에 대한 비율은 기본값으로 $\small \alpha=1$ 과 $\small \gamma=10$ 을 사용합니다.

<br>
\begin{equation}
\mathcal{L} = \mathcal{L}\_{spec} + \mathcal{L}_{CE(f_0)} + \alpha (\mathcal{L}\_{lsgan} + \gamma \mathcal{L}\_{fm})
\end{equation}
<br>

$\small \mathcal{L}_{spec}$ 은 예측된 합성 파라미터로부터 합성된 오디오와 실제값 오디오 사이의 재구성 손실입니다. 기존 DDSP 논문에서 사용하는 손실함수와 동일하며 다음 식과 같습니다.

<br>
\begin{align}
\mathcal{L}\_{spec}^{(i)} &= \lVert S\_i - \hat{S}\_i \rVert\_1 + \beta \lVert \log S\_i - \log \hat{S}\_i \rVert\_1 \newline
\newline
\mathcal{L}\_{spec} &= \sum\_i \mathcal{L}\_{spec}^{(i)} \qquad \forall i \in \\{ 2048, 1024, 512, 256, 128, 64 \\}
\end{align}
<br>

이떄 $\small S_i$ 와 $\small \hat{S}_i$ 는 각각 실제값과 예측값 오디오에 STFT를 적용하여 얻은 크기(magnitude) 스펙트로그램이며 $\small i$ 는 FFT 크기입니다.

$\small \mathcal{L}_{CE(f_0)}$ 는 $\small f_0$ 를 예측하는 RNN의 학습에 사용되며 다음 식과 같습니다.

<br>
\begin{equation}
\mathcal{L}\_{CE(f\_0)} = - \sum\_i {f\_0}\_i \log \hat{f\_0}\_i
\end{equation}
<br>

**Synthesis Generator**를 학습시키기 위한 LSGAN [(Xudong Mao et al., 2017)](https://arxiv.org/abs/1611.04076) 손실은 다음과 같습니다.

<br>
\begin{equation}
\mathcal{L}\_{lsgan} = \mathbb{E}\_c \left[ \sum\_k \lVert D\_k (\hat{\boldsymbol{s}}, \boldsymbol{c}) - 1 \rVert\_2 \right]
\end{equation}
<br>

이때 $\small k = [1, 2, 3]$ 는 $\small k$ 번째 discriminator를 의미하고 $\small \hat{\boldsymbol{s}}$ 는 예측된 합성 파라미터, $\small \boldsymbol{c}$ 는 조건 시퀀스입니다. Discriminator 학습을 위한 LSGAN 목적 함수는 다음과 같습니다.

<br>
\begin{equation}
\min\_{D\_k} \mathbb{E} [ \lVert D\_k (\boldsymbol{s}, \boldsymbol{c}) - 1 \rVert\_2 + \lVert D\_k (\hat{\boldsymbol{s}, \boldsymbol{c}}) \rVert\_2 ], \qquad \forall k
\end{equation}
<br>

특징 맵 손실은 각각의 discriminator 블록에서 따로 저장해 놓았던 특징 맵들을 이용하여 다음과 같이 계산합니다.

<br>
\begin{equation}
\mathcal{L}\_{fm} = \mathbb{E}\_{\boldsymbol{s}, \boldsymbol{c}} \left[ \sum\_{i=1}^4 \frac{1}{N\_i} \lVert D\_k^{(i)} (\boldsymbol{s}, \boldsymbol{c}) - D\_k^{(i)} (\hat{\boldsymbol{s}}, \boldsymbol{c}) \rVert\_1 \right]
\end{equation}
<br>

여기서 $\small D\_k^{(i)}$ 는 $\small k$ 번째 discriminator의 $\small i$ 번째 블록을 의미하며 $\small N\_i = 256$ 는 $\small i$ 번째 블록의 특징맵 차원 크기입니다.

**Synthesis Generator**의 신경망들 중에서 $\small f\_0$ 를 예측하는 RNN은 크로스 엔트로피 손실만을 이용해서 학습됩니다. 따라서 RNN에서 그래디언트가 끊기기 때문에 공식 소스 코드에서는 학습 시에 다른 네트워크의 입력으로 들어가는 $\small f\_0$ 는 모두 예측값이 아닌 실제값을 사용하기도 합니다. 예를 들어 1D dilated CNN, discriminator, 그리고 멀티스케일 스펙트럼 손실을 계산하기 위한 **DDSP Synthesizer**의 입력으로 추론 시에는 예측된 $\small f\_0$ 가 들어가야 하지만 학습 시에는 데이터셋에서 CREPE를 이용하여 추출된 $\small f\_0$ 실제값을 넣어줍니다.

<br><br>

## On the Improvement of Using GAN for the Synthesis Generator

GAN 구조를 사용하는 이유는 "over-smoothing" 문제를 방지하기 위한 목적입니다. 재구성 손실만을 사용하면 하모닉 분포가 균일하고 부드럽게 퍼져 있는 것이 전반적으로 손실을 감소시킵니다. 따라서 GAN 손실을 추가하여 하모닉 분포가 다양하게 생성되도록 장려하는 것입니다. 이에 대한 실험 결과는 아래 그림에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/WDcHt8c/gan-oversmoothing.png" alt="gan-oversmoothing" border="0">
</p>

GAN을 제외했을 때 테스트 샘플에 대한 하모닉 분포가 골고루 퍼져 있는 "over-smoothing" 문제가 발생하는 것을 볼 수 있습니다.

<br><br>

## Expression Generator

**Expression Generator**는 MIDI 음높이와 길이, 그리고 악기 임베딩을 입력으로 받아 연주 표현 요소들을 생성합니다. 모델 구조는 아래 그림에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/jwxnbn1/expression-generator.png" alt="expression-generator" border="0">
</p>

음높이와 길이, 악기는 각각 64차원으로 매핑됩니다. 이 시퀀스들은 192차원으로 연결되어 문맥 벡터를 추출하는 Bi-GRU와 자기회귀적 GRU를 통해 6차원의 연주 표현 요소 시퀀스들을 생성합니다. 그림에는 나와 있지 않지만 마지막에는 두 층 짜리 MLP가 있습니다.

이 네트워크들을 학습시킬 때에는 데이터 증강이 적용됩니다. 음높이는 $\small \\{ -3, -2, -1, 0, 1, 2, 3 \\}$ semitone 내에서 임의로 이동시키고 길이에는 $\small \\{ 0.9, 0.95, 1, 1.05, 1.1 \\}$ 중의 임의 값을 곱해줍니다.

학습의 손실 함수로는 실제값과 예측된 연주 표현 요소 사이의 MSE를 사용합니다.

<br><br>

## Training Procedures

**Expression Generator**, **Synthesis Generator**, **DDSP Inference** 모듈은 각각 따로 학습됩니다. 아래 그림의 왼쪽 모듈부터 차례대로 학습시키며 각각 10000 스텝, 40000 스텝, 5000 스텝씩 학습됩니다.

<p align="center">
<img src="https://i.ibb.co/52MRbrV/training-procedure.png" alt="training-procedure" border="0">
</p>

<br><br>

## 실험

실험에는 URMP 데이터셋을 [(Bochen Li et al., 2018)](https://ieeexplore.ieee.org/abstract/document/8411155) 사용합니다. 이 데이터셋은 여러 악기들의 솔로 연주들로 이루어져 있고 총 3.75시간 분량의 117개 녹음이 있습니다. 이 중 3시간 짜리 85개 녹음은 트레인 셋으로, 0.75시간 짜리 35개 녹음은 테스트 셋으로 사용합니다. 이때 같은 곡을 다른 악기로 연주한 것이 트레인과 테스트 셋에 겹쳐서 존재하지 않도록 나눠줍니다.

원래 데이터셋은 48 kHz 샘플 레이트를 가지고 있지만 16 kHz로 다운샘플링 해주고 모든 데이터는 50%씩 겹쳐진 4초 길이의 클립으로 나눠서 사용합니다. 실험 결과에 대한 데모 오디오 샘플은 [프로젝트 웹사이트](https://midi-ddsp.github.io/)에서 들어볼 수 있습니다.

### Model Accuracy

MIDI-DDSP의 계층 구조를 이루고 있는 세 개의 모듈들에 대해서 각각 재구성을 얼마나 잘하는지 평가합니다. 전체 결과는 아래 표에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/S3SSGkb/reconstruction.png" alt="reconstruction" border="0">
</p>

**DDSP Inference**는 합성된 오디오와 원래 오디오의 스펙트럼 로스를 비교합니다. MIDI-DDSP에서 멜 스펙트로그램으로부터 특징을 추출하는 CNN이 추가된 것이 더 정확한 재구성에 도움을 주는 것을 알 수 있습니다.

**Synthesis Generator**는 예측된 합성 파라미터로부터 다시 추출된 연주 표현 요소와 실제값 연주 표현 요소 사이의 RMSE로 비교합니다. **Expression Generator**는 연주 표현 요소의 실제값과 예측값 사이의 RMSE로 비교합니다. 이 두 모듈의 비교 대상은 MIDI2Params입니다 [(Rodrigo Castellon et al., 2020)](https://www-cs.stanford.edu/~rjcaste/research/realistic_midi.pdf).

MIDI2Params는 모델에 연주 표현 요소를 예측하는 모듈이 없고 MIDI 음표로부터 합성 파라미터를 직접 생성한다는 점에서 MIDI-DDSP가 이점을 갖고 있습니다.

### Audio Quality Evaluation by Human Listeners

사람 청취 평가는 **DDSP Inference** 모듈만 단독으로 사용한 것, MIDI2Params, 그리고 연결 샘플러(concatenative sampler)인 FluidSynth와 Ableton의 오케스트라 현악기 팩과 비교합니다. 참여자들은 두 모델이 생성한 8초 짜리 오디오 클립을 듣고 5점 Likert 스케일로 더 자연스러운 샘플을 선택합니다.

<p align="center">
<img src="https://i.ibb.co/RQmsRnc/human-evaluation.png" alt="human-evaluation" border="0">
</p>

**DDSP Inference** 단독 모듈은 실제값 오디오로부터 합성 파라미터를 추출하기 때문에 MIDI로부터 연주 표현 요소와 합성 파라미터를 예측하는 MIDI-DDSP의 상한 성능에 해당합니다. 각 모델들의 차이는 아래의 멜 스펙트로그램 그림에서도 뚜렷하게 나타납니다.

<p align="center">
<img src="https://i.ibb.co/GdRXH4S/spectrogram-comparison.png" alt="spectrogram-comparison" border="0">
</p>

MIDI2Params의 경우 다섯 번째 음에 해당하는 소리가 중간에 끊겨 있는 것이 보이는데 합성 파라미터 레벨에서만 모델링 하는 것이 긴 시간 동안의 일관성에 대해 한계를 보인다는 점을 나타냅니다.

### Effects of Note Expression Controls

MIDI-DDSP의 가장 주요한 특징이자 강점 중 하나는 각 음표에 대한 연주 표현을 제어할 수 있다는 것입니다. 이에 대한 평가를 위해 테스트 셋의 샘플에 대해서 각 표현 요소를 0부터 1까지 보간해가며 합성 파라미터를 생성합니다. 그 뒤 생성된 합성 파라미터로부터 연주 표현 요소를 다시 추출합니다. 아래 표에는 이 두 값들 사이의 상관관계를 나타낸 것입니다.

<p align="center">
<img src="https://i.ibb.co/sJKWj6s/interpolation.png" alt="interpolation" border="0">
</p>

볼륨 피크 위치를 제외하고는 대부분 높은 상관 관계로 제어되는 것을 볼 수 있습니다. 아래 그림은 각각의 표현 요소를 제어할 때 어떤 식으로 소리에 변화가 나타나는지 보여줍니다.

<p align="center">
<img src="https://i.ibb.co/CmnkjbZ/expression-control.png" alt="expression-control" border="0">
</p>

예를 들어, 비브라토를 증가시키면 주파수의 진동이 빨라지고 어택 노이즈를 증가시키면 초반부 노이즈가 증가합니다.

### Fine Grained Control or Full End-to-End Generation

MIDI-DDSP는 사용자가 최소한의 제어만 해서 음악을 만들어낼 수도 있고 계층 구조의 각 레벨에서 세밀한 조작을 할 수도 있습니다. 아래 그림은 각 레벨에서 서로 다른 음악적 요소들을 조절한 경우의 예시입니다.

<p align="center">
<img src="https://i.ibb.co/D7WgpDB/level-manipulation.png" alt="level-manipulation" border="0">
</p>

예를 들어 초록색 박스처럼 음표 레벨에서 비브라토를 조절하거나 어택 노이즈와 볼륨을 조절해서 스타카토를 표현할 수 있습니다. 또한 노란 박스처럼 합성 레벨에서 음높이 커브를 조절해서 피치 벤드 효과를 낼 수도 있습니다.

반면 완전 엔드투엔드로 기호 음악(symbolic music) 생성 모델과 결합하여 음악을 처음부터 생성할 수도 있습니다. 아래 그림은 COCONET을 [(Cheng-Zhi Anna Huang et al., 2017)](https://arxiv.org/abs/1903.07227) 이용하여 사중주(quartet) 음악을 자동으로 생성한 예시입니다.

<p align="center">
<img src="https://i.ibb.co/nCjzN52/endtoend-generation.png" alt="endtoend-generation" border="0">
</p>

<br><br>

## Reference

[Yusong Wu, Ethan Manilow, Yi Deng, Rigel Swavely, Kyle Kastner, Tim Cooijmans, Aaron Courville, Cheng-Zhi Anna Huang and Jesse Engel. MIDI-DDSP: Detailed Control of Musical Performance via Hierarchical Modeling. In ICLR, 2022.](https://openreview.net/forum?id=UseMOjWENv)

[Official Source Code of MIDI-DDSP](https://github.com/magenta/midi-ddsp)