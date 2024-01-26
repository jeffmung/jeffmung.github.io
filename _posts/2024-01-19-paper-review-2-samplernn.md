---
layout: post
title: "[논문 리뷰] SampleRNN: An Unconditional End-to-End Neural Audio Generation Model"
image: https://i.ibb.co/cw9zNGn/thumbnail.png
date: 2024-01-19
tags: 
categories: Paper-Review
use_math: true
---

<br><br>

## 논문 개요

RNN은 시계열 데이터를 학습시키고 시퀀스를 생성하기에 적합한 모델입니다. 하지만 오디오 도메인에서는 시간적 해상도(temporal resolution)와 스케일과 관련된 어려움들이 존재합니다. 예를 들어, 일반적으로 오디오 신호의 샘플 레이트가 적어도 16 kHz는 되는데 이러한 신호에서 학습해야 되는 패턴은 아주 작은 스케일로 연속적인 몇 개의 샘플들 사이에도 있을 수 있고, 수백 ~ 수천 ms에 걸쳐서 존재할 수도 있습니다.

게다가 RNN이 한 번에 하나의 샘플씩 생성할 경우 이러한 패턴에 대한 시간적 해상도와 너무 차이나는 스케일로 인해 좋은 성능을 기대하기가 어렵습니다. WaveNet이 [(van den Oord et. al., 2016)](https://arxiv.org/abs/1609.03499) RNN이 아닌 dilated CNN을 사용한 것도 아마 이러한 이유 때문일 것입니다.

이 논문에서는 계층적인 구조로 RNN 모델을 설계하여 이러한 어려움을 극복합니다. 실험 결과에서는 음성과 음악 데이터셋에 대해 학습시켜 WaveNet과 비교했을 때 더 좋은 성능을 보여줍니다. 몬트리올의 Yoshua Bengio 그룹이 연구를 진행하여 ICLR 2017에 발표하였습니다.

<br><br>

## Hierarchical RNN Model for Audio Waveforms

SampleRNN은 파형(waveform) 샘플들의 시퀀스를 이전 타임스텝의 샘플들에 대한 조건부 확률들의 곱으로 모델링합니다. 식으로 나타내면 다음과 같습니다.

<br>
<center> $ p(X) = \prod_{i=0}^{T-1} p(x_{i+1} \vert x_1, \ldots, x_{i}) $ </center>
<br>

이를 위해 사용하는 RNN은 다음 식과 같이 표현됩니다.

<br>
<center> $ h_t = \mathcal{H} (h_{t-1}, x_{i=t}) $ </center>
<center> $ p(x_{i+1} \vert x_1, \ldots, x_{i}) = Softmax (MLP(h_t)) $ </center>
<br>

여기서 $\mathcal{H}$는 GRU나 LSTM 등 RNN 계열 네트워크들의 메모리 셀이고 $h_{t}$는 타임스텝 $t$에서의 은닉 상태(hidden state)입니다. 오디오 신호를 처리할 때의 어려움을 극복하기 위해 SampleRNN은 각각 다른 시간적 해상도에서 동작하는 RNN 모듈들의 계층 구조를 사용합니다.

가장 낮은 모듈은 샘플 레벨에서 각각의 연속된 샘플들을 처리하고 높은 레벨의 모듈들은 더 긴 시간 스케일, 즉 더 낮은 시간적 해상도로 동작합니다. 각각의 모듈들은 바로 아래 레벨의 모듈에 조건을 부여하고, 가장 낮은 레벨의 모듈은 샘플 레벨의 예측을 출력합니다. 전체 모델은 엔드-투-엔드(end-to-end)로 역전파(backpropagation)를 통해 학습됩니다.

<br><br>

## Frame-level Modules

가장 아래 레벨을 제외한 모듈들은 모두 프레임 레벨로 동작합니다. 이때 프레임 $f^{(k)}$는 겹쳐지지 않은(non-overlapping) $FS^{(k)}$ 개의 샘플들로 구성되어 있습니다. $FS$는 Frame Size를 말하며 위첨자 $(k)$는 k번째 레벨의 모듈을 의미합니다.

각 모듈의 RNN은 이전 프레임까지의 입력을 반영한 은닉 상태 $h_{t-1}^{(k)}$과 현재의 입력 $inp_t^{(k)}$를 받아서 메모리를 업데이트합니다. 이때 $t$는 실제 타임스텝이 아니라 각 레벨의 시간적 해상도를 반영한 시간 간격입니다. 예를 들어, 3번째 레벨의 프레임 크기 $FS^{(3)}$이 64이면 $h_{t-1}^{(3)}$과 $h_{t}^{(3)}$ 사이에는 실제로 64개의 타임스텝 차이가 존재하는 것입니다.

입력 $inp$은 가장 위 레벨 $k=K$의 모듈일 때에는 단순히 현재 프레임이고, 중간 레벨 $1 < k < K$의 모듈에 대해서는 바로 위 레벨의 출력 벡터와 현재 프레임의 선형 결합(linear combination)이 됩니다. 이것을 식으로 나타내면 다음과 같습니다.

<br>
<center> $ inp_t = \left\{ \begin{aligned} &W_x f_t^{(k)} + c_t^{(k+1)} \qquad\; 1 < k < K \\ &f_t^{(k=K)} \qquad\qquad\qquad k = K \end{aligned} \right. $ </center>
<br>
<center> $ h_t = \mathcal{H}(h_{t-1}, inp_t) $ </center>
<br>

각 레벨의 모듈들이 서로 다른 시간적 해상도로 동작하기 때문에 위 모듈에서의 출력 벡터 $c$를 바로 아래 모듈의 입력 조건으로 넣어줄 때 업샘플링을 해줘야 됩니다. 위아래 모듈의 시간적 해상도 비율을 $r^{(k)}$라고 하면 위 모듈의 출력이 $r^{(k)}$ 개의 벡터로 확장되어야 하고, 이러한 업샘플링은 아래의 식과 같이 선형 프로젝션(linear projection)을 통해 이루어집니다. 

<br>
<center> $ c_{(t-1)*r + j}^{(k)} = W_j h_t \qquad\qquad 1 \leq j \leq r $ </center>
<br>

이러한 계층적 구조를 표현한 아래의 전체 모델 그림을 보면 이해하는 데 도움이 됩니다. 이 그림은 $K=3$이고 $FS^{(2)}$가 4, $FS^{(3)}$가 16일 때의 예시입니다.

<p align="center">
  <img src="https://i.ibb.co/vhcN3Rt/samplernn-model.png" alt="samplernn model" border="0">
</p>

2번째 레벨과 3번째 레벨의 시간적 해상도 비율 $r^{(3)}$가 4이므로 3번째 레벨의 모듈에서 출력된 벡터는 4개의 벡터로 업샘플링되어 2번째 레벨의 모듈로 들어갑니다. 2번째 모듈의 입력은 위에서 업샘플링된 벡터와 프레임 크기 $FS^{(2)} = 4$의 현재 프레임 입력이 선형 결합된 것입니다.

<br><br>

## Sample-level Module

가장 아래 레벨의 모듈은 한 번에 하나의 샘플 $x_{i+1}$에 대한 확률 분포를 출력합니다. 입력은 상위 레벨의 모듈들과 마찬가지로 프레임 단위로 넣을 수 있지만 이 경우에는 겹쳐진(overlapped) 프레임이 들어갑니다. 예를 들어 위의 그림에서는 첫 번째 레벨의 모듈에서 프레임 크기 $FS^{(1)}$가 4이고, $x_{i+28}, \ldots, x_{i+31}$이 먼저 입력으로 들어가면 다음 입력으로는 $x_{i+29}, \ldots, x_{i+32}$가 들어갑니다.

$FS^{(1)}$의 값이 일반적으로 작기 때문에 굳이 RNN을 사용하지 않고 간단하게 MLP를 사용합니다. 이때 양자화된 정수 값을 갖는 샘플 $x_i$는 임베딩 레이어를 통해 벡터 $e_i$로 임베딩 되고 바로 위의 모듈에서 업샘플링되어 출력된 벡터와 선형 결합되어 MLP에 입력으로 들어갑니다. 이를 식으로 나타내면 다음과 같습니다.

<br>
<center> $ f_{i-1}^{(1)} = flatten([e_{i - FS^{(1)}}, \ldots, e_{i-1}]) $ </center>
<center> $ f_{i}^{(1)} = flatten([e_{i - FS^{(1)} + 1}, \ldots, e_{i}]) $ </center>
<br>
<center> $ inp_i^{(1)} = W_x^{(1)}f_i^{(1)} + c_i^{(2)} $ </center>
<br>

최종 출력은 아래의 식과 같이 softmax를 적용하여 카테고리형 분포에 대한 음의 로그 우도(negative log-likelihood, NLL)를 최소화 하는 손실함수로 학습시킵니다.

<br>
<center> $ p(x_{i+1} \vert x_1, \ldots, x_i) = Softmax(MLP(inp_i^{(1)})) $ </center>
<br>

학습이 끝나고 생성할 때에는 한 번에 하나씩 샘플을 생성하고 자기회귀적(auto-regressive)으로 이전 스텝의 출력을 입력으로 넣어줍니다.

<br><br>

## Output Quantization