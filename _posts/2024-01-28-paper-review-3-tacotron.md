---
layout: post
title: "[논문 리뷰] Tacotron: Towards End-to-End Speech Synthesis"
image: https://i.ibb.co/Hhr8FJ4/thumbnail.png
date: 2024-01-28
tags: 
categories: paper-review
use_math: true
---

<br><br>

## 논문 개요
<!-- excerpt-start -->
Tacotron은 2017년 구글이 개발하고 Interspeech에 발표한 TTS 모델로, 이후로도 여러 버전의 발전된 모델들이 출시되었습니다. 이 [구글 리서치 페이지](https://google.github.io/tacotron/)에 지금까지 개발된 Tacotron을 기반으로 한 변종 모델들의 논문과 각각의 데모 오디오 샘플들이 정리되어 있습니다.

이전까지의 TTS 모델들은 일반적으로 언어적 특성(linguistic feature)을 추출하는 텍스트 프론트엔드, 음향적 특성(acoustic feature)을 예측하는 모델, 오디오 신호를 생성하는 보코더 등 여러 요소들로 이루어진 복잡한 파이프라인을 가지고 있었습니다. 이러한 구성요소들은 독립적으로 학습되어 각각의 에러가 누적되며, 새로운 시스템으로 확장하려면 도메인 지식을 기반으로 한 세심한 엔지니어링이 필요하다는 단점도 있습니다.

반면 Tacotron은 이러한 문제들을 해결하고자 엔드-투-엔드 접근을 채택합니다. 이 모델은 텍스트에서 음성으로 변환하는 데 필요한 복잡한 파이프라인 대신에 <text, audio> 쌍의 데이터를 사용하여 단일 모델에서 학습합니다. 따라서 수작업으로 정확한 특징을 추출하거나 정렬할 필요 없이 대규모의 음성 데이터를 활용할 수 있고 목소리, 언어 등 다양한 속성에 대한 조건부 학습도 가능합니다.

또한 Tacotron은 입력 시퀀스의 전체적인 컨텍스트를 동시에 고려하여 출력을 생성하는 방식이므로 WaveNet이나 [(van den Oord et al., 2016)](https://arxiv.org/abs/1609.03499) SampleRNN [(Soroush Mehri et al., 2016)](https://openreview.net/forum?id=SkxKPDv5xl) 같은 자기회귀적(autoregressive) 모델과 비교했을 때 병렬 처리가 가능하고 학습과 추론 속도가 빠른 장점이 있습니다.

전체적인 모델 구조는 attention 메커니즘과 seq2seq을 [(Ilya Sutskever et al., 2014)](https://proceedings.neurips.cc/paper/2014/hash/a14ac55a4f27472c5d894ec1c3c743d2-Abstract.html) 기반으로 하여 문자(character)를 입력으로 받고 스펙트로그램을 출력으로 생성한 뒤, 후처리(post-processing) 네트워크에서 이를 받아 음성 파형으로 변환해주는 형태입니다. 아래 그림에 전체 구조가 표현되어 있습니다.

<p align="center">
    <img src="https://i.ibb.co/dKgHqNX/architecture.png" alt="architecture" border="0">
</p>

<br><br>

## CBHG Module

Tacotron 모델의 구성요소 중 하나인 CBHG 모듈은 1-D convolutional filter bank, highway networks, bidirectional GRU로 구성되어 있는 블록입니다. 아래 그림이 CBHG 모듈의 구조를 나타낸 것입니다.

<p align="center">
    <img src="https://i.ibb.co/1JBHbk2/cbhg.png" alt="cbhg" border="0">
</p>

먼저 1-D convolutional bank는 $\small K$ 세트의 필터들을 포함합니다. 이때 $\small k$ 번째 세트는 필터 사이즈가 $\small k$입니다 (즉, $\small k=1, 2, \ldots, K$). 예를 들어 $\small K=16$이면, 필터 사이즈가 각각 $\small 1, 2, \ldots, 16$인 $\small 16$개의 1-D 컨볼루션 층(convolutional layer)이 쌓여있지 않고 따로따로 있는 것입니다. 이러한 $\small K$ 세트의 필터들로 이루어진 컨볼루션 층을 통해 입력 시퀀스의 국소적인 맥락(local context) 정보를 추출하는 것이 목적인데 unigram, bigram, ..., K-gram을 모델링하는 것과 같다고 볼 수 있습니다.

각각의 컨볼루션 층 출력은 채널 차원에서 스택된 뒤 시간축을 따라 max pooling 됩니다. Max pooling으로 기대하는 것은 모델이 국소적인 변화에 너무 민감해지지 않도록 국소 불변성(local invariance)을 증가시키는 것이고, 커널 사이즈는 $\small 1$보다 크면서 스트라이드는 $\small 1$이 되게 설정하여 시퀀스 길이는 유지시킵니다.

그리고 마찬가지로 시퀀스 길이가 그대로 유지되도록 커널 사이즈와 패딩을 설정한 몇 개의 컨볼루션 층을 거쳐서 residual connection을 해줍니다. 모든 컨볼루션 층에는 batch normalization을 적용합니다.

그 다음에는 몇 개의 층으로 이루어진 highway network가 [(Rupesh Kumar Srivastava et al., 2015)](https://arxiv.org/abs/1505.00387) 있습니다. Highway network는 간단히 표현하면 MLP를 구성하는 각각의 층마다 입력이 그대로 전달될지, 혹은 비선형 연산을 거쳐서 전달될지를 결정해주는 게이트들을 학습시키는 네트워크입니다. 게이트는 일반적으로 $\small 0 \sim 1$의 값을 출력하도록 학습 가능한 가중치(weight)와 시그모이드 활성 함수로 구성되며, MLP 층의 출력에 게이트 $\small T$의 출력을 곱해주고 병렬적으로 $\small (1 - T)$와 입력을 곱해준 뒤 서로 더하는 방식입니다.

최종적으로 bidirectional GRU를 통해 맥락 정보를 담은 특성(feature)을 출력합니다. CBHG 모듈에서 계속 입력 시퀀스의 길이는 유지되므로 최종 출력은 입력 시퀀스와 같은 길이의 특성 벡터 시퀀스가 됩니다.

<br><br>

## Encoder

인코더의 목적은 문자 시퀀스 입력을 받아서 그 텍스트의 표현(representation)이 추출된 특성 시퀀스를 출력하는 것입니다. 인코더는 임베딩 네트워크, pre-net, CBHG 모듈로 구성되어 있습니다.

입력은 각각의 문자들이 원핫 벡터로 표현된 시퀀스이고, 맨 처음에 임베딩 네트워크를 통해 각각의 문자가 실수 공간의 연속적인 벡터로 대응되도록 임베딩된 후 pre-net으로 들어갑니다. Pre-net은 dropout이 적용된 MLP입니다. 이후 CBHG 모듈을 거쳐서 인코더의 최종 출력이 나옵니다. 이렇게 인코딩된 텍스트의 표현은 이후 attention을 포함한 디코더에서 스펙트로그램으로 변환됩니다.

<br><br>

## Decoder

디코더는 80개 밴드의 mel-scale spectrogram을 타겟으로 학습하고 예측합니다. 또한 한 스텝에 하나씩의 프레임만 예측하는 것이 아니라 여러 개의 겹치지 않는(non-overlapping) 프레임을 예측합니다. 이렇게 한 스텝에 여러 개의 프레임을 한 번에 예측함으로써 학습 속도도 빨라지고 모델의 크기도 줄일 수 있습니다.

Tatotron에 사용하는 디코더의 attention 기법은 Bahdanau attention에 [(Dzmitry Bahdanau et al., 2014)](https://arxiv.org/abs/1409.0473) 가깝습니다. 디코더의 입력은 먼저 pre-net을 통과한 뒤 attention RNN 셀의 은닉 상태(hidden state)가 query, 인코더의 출력 시퀀스가 key가 되어 서로 더해져서 tanh 연산으로 attention score가 계산됩니다. 그 뒤 value와 가중합(weighted sum)을 하여 얻어진 맥락 벡터(context vector)와 attention RNN 셀의 출력을 결합하여(concatenate) decoder RNN 셀의 입력으로 넣습니다. 이때 각 decoder RNN의 각 층마다 residual sum을 해줍니다.

매 스텝마다 decoder RNN의 출력은 $\small r$개 프레임의 mel spectrogram입니다. 첫 스텝의 입력으로 들어가는 \<GO\> 토큰은 모든 값이 $\small 0$인 프레임이고, 이후 스텝에서 학습 시에는 데이터에 있는 이전 스텝의 $\small r$ 번째 프레임의 실제값(ground truth)으로 들어갑니다. 추론(inference) 시에는 이전 스텝에서 예측한 $\small r$개 프레임 중 마지막 프레임이 다음 입력으로 들어갑니다.

<br><br>

## Post-processing Net and Waveform Synthesis

Post-processing net은 디코더의 출력으로 나온 mel spectrogram을 선형 주파수(linear-frequency) 스케일의 스펙트로그램으로 바꿔주는 역할을 합니다. 간단한 구조의 네트워크를 사용해도 되지만 논문에서는 post-processing net으로 CBHG 모듈을 사용합니다. 얻어진 linear spectrogram을 다시 음성 신호로 복원하기 위해서는 Griffin-Lim 알고리즘을 [(Daniel Griffin and Jae Lim, 1984)](https://ieeexplore.ieee.org/document/1164317) 사용합니다.

최종적으로 모델의 손실 함수는 디코더 출력과 mel spectrogram 사이의 L1 loss, 그리고 post-processing network의 출력과 linear spectrogram 사이의 L1 loss를 합한 것을 사용합니다. Tacotron 모델에서 사용한 몇몇 세부적인 테크닉들과 하이퍼파라미터는 논문에 더 자세하게 나와 있습니다.

<br><br>

## 실험

실험에 사용한 데이터셋은 여성 목소리의 24.6 시간짜리 영어 음성입니다. 텍스트 입력은 모두 문자로 변환하여 사용되었습니다. 예를 들어 "16"은 "sixteen"으로 변환되는 것입니다. 실험에 대한 데모 샘플들은 [여기](https://google.github.io/tacotron/publications/tacotron/index.html)에서 들어볼 수 있습니다.

### Ablation Analysis

먼저 인코더와 디코더를 vanilla seq2seq 모델로 사용한 것, 인코더의 CBHG 모듈을 GRU로 대체한 것, 그리고 Tacotron과의 비교 실험을 진행했습니다. 음성 생성의 특성 상 객관적인 평가 지표를 사용하기가 쉽지 않은데 이 실험은 인코더와 디코더 타임스텝 사이의 attention 값을 시각화하여 attention alignment의 정도가 어떤지를 기준으로 판단했습니다. TTS는 텍스트와 음성이 순서대로 대응되기 때문에 attention alignment가 단조롭게 증가하는(monotonic) 형태가 나오는 것이 잘 학습된 경우일 가능성이 높습니다.

<p align="center">
    <img src="https://i.ibb.co/xLHHGsB/attention-alignment.png" alt="attention alignment" border="0">
</p>

위 그림의 (a)를 보면 vanilla seq2seq를 사용한 경우에는 attention alignment가 좋지 않습니다. CBHG 대신 GRU를 사용한 (b)의 경우에도 (c)에 비하면 떨어지는 attention alignment를 보여주는데 실제 샘플을 들어보면 발음을 잘못하는 경우가 종종 있었다고 합니다.

다른 ablation 분석으로는 post-processing net의 유무에 따른 비교가 있습니다. 아래 그림의 왼쪽은 post-processing net을 사용하지 않았을 때, 오른쪽은 사용했을 때의 스펙트로그램입니다. 오른쪽의 스펙트로그램이 harmonic partial들을 더 명확하게 나타내고 있습니다.

<p align="center">
    <img src="https://i.ibb.co/1Q5zrVQ/spectrogram-postnet.png" alt="spectrogram postnet" border="0">
</p>

### Mean Opinion Score Tests

MOS 테스트는 LSTM에 기반한 parametric system과 [(Heiga Zen et al., 2016)](https://arxiv.org/abs/1606.06061) concatenative system을 [(Xavi Gonzalvo et al., 2016)](https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/45564.pdf) 비교군으로 이루어졌습니다. 영어 사용자들이 모집되어 학습에 사용되지 않은 100개의 구문에 대해 5점 만점의 Likert scale 점수로 평가했습니다.

<p align="center">
    <img src="https://i.ibb.co/tcgQNs5/mos-test.png" alt="mos-test" border="0">
</p>

위의 표가 MOS 테스트 결과입니다. Tacotron은 3.82로 꽤 높은 점수를 보여주고 있습니다.

<br><br>

## Reference

[Yuxuan Wang, RJ Skerry-Ryan, Daisy Stanton, Yonghui Wu, Ron J Weiss, Navdeep Jaitly, Zongheng Yang, Ying Xiao, Zhifeng Chen, Samy Bengio, et al. Tacotron: Towards End-to-End Speech Synthesis. In INTERSPEECH, 2017.](https://arxiv.org/abs/1703.10135)

[Pytorch implementation of Tacotron](https://github.com/r9y9/tacotron_pytorch)
