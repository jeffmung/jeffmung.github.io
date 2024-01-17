---
layout: post
title: "[논문 리뷰] WaveNet: A Generative Model for Raw Audio"
image: https://i.ibb.co/PFc9Pnw/thumbnail.png
date: 2024-01-10
tags: 
categories: Paper-Review
use_math: true
---

<br><br>

## 논문 개요

WaveNet은 2016년 구글 딥마인드에서 발표한 오디오 생성 모델입니다. 이 논문의 1저자인 Aaron van den Oord는 이전에 이미지를 생성하기 위한 모델인 PixelRNN과 ([van den Oord et. al., 2016](https://arxiv.org/abs/1601.06759)) PixelCNN을 ([van den Oord et. al., 2016](https://proceedings.neurips.cc/paper_files/paper/2016/hash/b1301141feffabac455e1f90a7de2054-Abstract.html)) 발표하였는데 PixelCNN에서 사용한 dilated causal CNN의 개념을 오디오에도 마찬가지로 적용하여 좋은 결과를 얻어냈습니다.

WaveNet은 text-to-speech(TTS)에 사용되어 텍스트를 음성으로 변환할 수도 있고, 발화자에 대해 조건을 두어 하나의 모델로 여러 목소리를 생성할 수도 있습니다. 또한 음악 데이터로 WaveNet을 학습시켜 음악을 생성하는 것도 가능합니다.

<br><br>

## Probabilistic and Autoregressive Generative Model

WaveNet은 입력으로 들어온 오디오 데이터 시퀀스의 다음에 올 값들을 예측하도록 학습하고, 생성할 때에는 이전 출력값이 다음 입력으로 들어가는 autoregressive 방식으로 동작하는 모델입니다. 다음 값을 예측한다는 것을 좀 더 자세히 말하면, 이전 타임스텝의 샘플들에 조건을 두어 다음 타임스텝의 파형(waveform) 확률 분포를 예측한다는 것입니다. 파형 $\mathbf{x} = \{ x_1, \ldots, x_T \}$의 결합 확률(joint probability)은 다음과 같은 조건부 확률의 곱으로 표현됩니다.

<br>
<center> $ p(\mathbf{x}) = \prod_{l=1}^T p(x_l \vert x_1, \ldots, x_{l-1}) $ </center>
<br>

이러한 조건부 확률분포는 dilated causal CNN으로 모델링되어 네트워크의 출력에 소프트맥스(softmax)를 취한 것이 다음 값 $x_t$에 대한 카테고리형 분포(categorical distribution)로 나옵니다. 예측된 확률과 실제 $x_t$의 log-likelihood를 최대화하도록 하기 위해 손실 함수로는 cross entropy가 사용됩니다.

<br><br>

## Dilated Causal Convolutions