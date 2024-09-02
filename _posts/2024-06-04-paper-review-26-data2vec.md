---
layout: post
title: "data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language [ICML, 2022]"
description: data2vec 논문 리뷰. data2vec은 self-supervised learning을 도메인에 따라 따로 분리하지 않고 동일하게 사용할 수 있도록 개발된 멀티모달 프레임워크입니다.
image: https://i.ibb.co/FDV62JF/thumbnail.png
date: 2024-06-04
tags: 
categories: paper-review
use_math: true
---

## 논문 개요

<!-- excerpt-start -->

data2vec은 자기지도학습(self-supervised learning) 방법을 도메인에 따라 따로 분리하지 않고 동일하게 사용할 수 있도록 개발된 프레임워크입니다. 기존에는 이미지, 스피치, NLP 등 도메인에 따라 자기지도학습 방법이 따로 구분되어서 설계되었지만 data2vec은 문맥화된(contextualized) 잠재 표현을 타겟으로 마스킹하여 예측(masked prediction)하는 통합된 방법을 제안합니다. 실험 결과는 세 가지 도메인에서 모두 data2vec이 기존 방법들에 비해 성능이 우수하다는 것을 보여줍니다.

<br><br>

## Model Architecture

모델 구조는 일반적인 트랜스포머 인코더이며 학습 샘플은 마스킹 되어 인코딩되는 $\small \text{student mode}$ 와 마스킹되지 않고 인코딩되는 $\small \text{teacher mode}$ 두 가지로 나뉘어서 처리됩니다. $\small \text{teacher mode}$ 는 $\small \text{student mode}$ 와 같은 모델을 사용하지만 exponential moving average (EMA)를 취한 파라미터를 사용합니다. $\small \text{student mode}$ 임베딩은 마스킹된 부분에서 $\small \text{teacher mode}$ 의 임베딩을 예측하도록 학습됩니다. 학습 과정은 아래 그림에 요약되어 있습니다.

<p align="center">
<img src="https://i.ibb.co/2KVBYDq/overview.png" alt="overview" border="0">
</p>

CV 태스크에는 ViT와 같이 이미지를 패치로 나눠서 넣습니다. 스피치 태스크에서는 16 kHz의 파형을 1D CNN으로 인코딩하여 50 Hz 표현으로 만듭니다. NLP 태스크에는 텍스트를 유닛으로 전처리하고 임베딩 한 것을 사용합니다. 입력 샘플들을 토큰 시퀀스로 임베딩한 이후에는 일부를 학습된 토큰으로 마스킹하여 트랜스포머에 넣습니다. 

<br><br>

## Training Targets

모델은 마스킹된 샘플의 인코딩이 마스킹되지 않은 샘플의 표현을 예측하도록 학습됩니다. 예측은 마스킹된 타임스텝에 대해서만 수행하며, 예측하는 표현은 특정 타임스텝의 인코딩이지만 트랜스포머의 셀프어텐션으로 인해 다른 정보도 반영된 문맥화된 표현입니다. 이것은 BERT나 MAE 등이 문맥 정보가 없는 특정 위치의 표현을 예측하는 것과 구분되는 중요한 특징입니다.

마스킹 되지 않은 샘플은 $\small \text{student-mode}$ 모델 파라미터 $\small \theta$ 의 EMA를 사용한 $\small \text{teacher-mode}$ 모델로 인코딩 됩니다. $\small \text{teacher-mode}$ 모델의 파라미터 $\small \Delta$ 는 다음과 같이 업데이트 됩니다.

<br>
\begin{equation}
\Delta \leftarrow \tau \Delta + (1 - \tau) \theta
\end{equation}
<br>

업데이트 되는 비율을 결정하는 $\small \tau$ 는 $\small \tau\_0$ 로부터 $\small \tau\_e$ 까지 첫 $\small \tau\_n$ 번의 업데이트 동안 선형적으로 증가하도록 스케쥴링 합니다. 이러한 스케쥴링은 학습 초반에는 많이 업데이트 하다가 모델이 충분히 학습되면 덜 업데이트 하는 것을 의도한 전략입니다. 또한 EMA 업데이트는 트랜스포머 층에만 적용하고 그 아래의 위치 인코더(positional encoder)와 특징 인코더 (feature encoder)는 그냥 $\small \text{teacher network}$ 와 $\small \text{student network}$ 의 파라미터를 공유합니다.

<br><br>

## Targets

학습의 타겟은 $\small \text{teacher network}$ 의 맨 위 $\small K$ 개 블럭들의 출력에 기반하여 만들어집니다. 타겟이 되는 타임스텝은 $\small \text{student-mode}$ 에서 마스킹된 부분입니다. 타임스텝 $\small t$ 에서 블럭 $\small l$ 의 출력을 $\small a_t^l$ 이라 명명하겠습니다. 학습 타겟 $\small y_t$ 는 각각의 블럭에 정규화(normalization)를 적용하여 $\small \hat{a}_t^l$ 을 얻고 다음 식과 같이 맨 위 $\small K$ 개 블럭의 평균으로 만듭니다.

<br>
\begin{equation}
y\_t = \frac{1}{K} \sum\_{l = L - K + 1}^{L} \hat{a}\_t^l
\end{equation}
<br>

타겟을 정규화하는 것은 모델이 모든 타임스텝에 대해 동일한 표현으로 붕괴되는 것을 방지하는 데 도움이 됩니다. 또한 높은 값의 분포를 가지고 있는 특정 층이 타겟 표현에 지배적으로 작용하는 것도 방지해줍니다. 이렇게 문맥화된 타겟 $\small y_t$ 가 주어졌을 때 모델 학습의 목표는 $\small \text{student encoder}$ 의 인코딩의 마스킹된 부분이 $\small \text{student decoder}$ 를 거쳐 타겟을 예측하는 것입니다. 손실함수로는 Smooth L1 loss를 사용합니다.

<br><br>

## Experimental Setup

모델은 크기에 따라 두 개로 나눠서 실험합니다. data2vec Base는 $\small L=12$ 개의 트랜스포머 블럭과 $\small H=768$ 의 은닉 차원(hidden dimension)을 사용하고 data2vec Large는 $\small L=24$ 와 $\small H=1024$ 를 사용합니다.

### Computer Vision

이미지를 위한 트랜스포머는 BEiT를 [(Hangbo Bao et al., 2021)](https://arxiv.org/abs/2106.08254) 기반으로 합니다. 이미지를 $\small 16 \times 16$ 픽셀의 패치들로 나누고 인접한 여러 패치들을 같이 마스킹합니다. 각각의 마스킹 블럭은 최소 16개의 패치들을 포함합니다. 마스킹 비율은 60%를 사용하고 resized image crops, horizontal flipping, color jittering을 임의로 적용합니다. 수정된 이미지는 $\small \text{student-mode}$ 와 $\small \text{teacher-mode}$ 에서 모두 사용합니다.

### Speech Processing

스피치 모델은 fairseq의 [(Myle Ott et al., 2019)](https://arxiv.org/abs/1904.01038) 구현을 기반으로 합니다. 마스킹 전략은 wav2vec 2.0을 [(Alexei Baevski et al., 2020)](https://proceedings.neurips.cc/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html) 따라서 $\small p=0.065$ 의 시작 인덱스를 샘플링하고 뒤따르는 10개의 타임스텝을 마스킹합니다. 결과적으로 전체에서 약 49%의 타임스텝이 마스킹됩니다.

### Natural Language Processing

NLP 태스크 모델은 RoBERTa를 [(Yinhan Liu et al., 2019)](https://arxiv.org/abs/1907.11692) 기반으로 구현합니다. 입력 데이터는 byte-pair 인코딩으로 토큰화되고 15%의 토큰들에 대해 BERT 마스킹 전략을 적용하여 80%는 학습된 마스크 토큰으로, 10%는 임의로 선택된 어휘 토큰으로 대체하고 나머지 10%는 그대로 둡니다.

<br><br>

## Results

### Computer Vision

CV에서 data2vec을 평가하기 위해 ImageNet-1K에 대해 사전학습한 뒤 같은 벤치마크의 레이블 데이터를 사용하여 이미지 분류를 하도록 모델을 파인튜닝합니다. 평가 지표는 top-1 validation accuracy입니다. 아래 표는 data2vec이 다른 방법들을 능가하는 것을 보여줍니다.

<p align="center">
<img src="https://i.ibb.co/MpXf8sJ/cv-result.png" alt="cv-result" border="0">
</p>

### Speech and Audio Processing

스피치 도메인에 대해서는 ASR 태스크의 성능을 평가합니다. 960시간 짜리 Librispeech (LS-960) 데이터셋에 대해 사전학습한 뒤 10분부터 960시간까지 다양한 범위의 레이블 데이터의 ASR을 목표로 모델을 파인튜닝합니다. 아래 표는 data2vec의 성능이 다른 모델들에 비해 우세하다는 것을 보여줍니다.

<p align="center">
<img src="https://i.ibb.co/fvB7KNM/speech-result.png" alt="speech-result" border="0">
</p>

또한 AudioSet 벤치마크의 오디오 이벤트 분류 태스크에 대해서도 실험한 결과가 아래 표에 나와 있습니다. 역시 data2vec이 가장 우수한 성능을 보여줍니다.

<p align="center">
<img src="https://i.ibb.co/qyBRKy0/audioset-result.png" alt="audioset-result" border="0">
</p>

### Natural Language Processing

NLP는 Books Corpus와 English Wikipedia 데이터셋을 사전학습하고 GLUE 벤치마크로 평가합니다. 그 결과는 아래 표에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/LznkWtk/nlp-result.png" alt="nlp-result" border="0">
</p>

data2vec은 RoBERTa 베이스라인을 능가하고 wav2vec 2.0 마스킹 전략을 적용하였을 때에는 성능이 더 증가합니다.

<br><br>

## Ablations

### Layer-average Targets

BYOL과 비교했을 때 data2vec의 가장 큰 차이는 $\small \text{teacher network}$ 의 여러 층에서 나온 표현의 평균을 타겟으로 사용한다는 것입니다. 이 아이디어는 wav2vec 2.0의 맨 위 층이 중간 층보다 다운스트림 태스크에서 더 안 좋은 성능을 나타낸다는 것에서 영향을 받았습니다. 아래 그림은 맨 위부터 $\small K$ 개 층의 평균을 타겟으로 사용했을 때의 결과를 나타냅니다.

<p align="center">
<img src="https://i.ibb.co/bK1b1f3/layer-avg.png" alt="layer-avg" border="0">
</p>

모든 태스크에 대해 여러 층의 평균을 사용하는 것이 맨 위 층 하나만 사용하는 것보다 훨씬 유리하다는 것을 알 수 있습니다.

### Text Contextualization

data2vec의 타겟은 셀프어텐션이 적용되기 때문에 문맥 정보를 담고 있는 표현입니다. 이것은 다른 자기지도학습 방법에서 입력의 일부분을 그대로 복원하거나 예측하도록 학습하는 것과 차별화되는 점입니다. 아래 그림은 셀프어텐션이 적용되는 크기에 따른 실험 결과를 보여줍니다.

<p align="center">
<img src="https://i.ibb.co/SwjWh7r/context-size.png" alt="context-size" border="0">
</p>

문맥화된 타겟을 사용하는 것이 성능 향상에 도움이 된다는 것을 알 수 있습니다.

### Target Feature Type

트랜스포머 블럭 안에 존재하는 여러 층의 출력을 모두 타겟으로 사용할 수 있고 때문에 각각이 성능에 미치는 영향이 다를 수 있습니다. 아래 표는 각각의 층을 타겟으로 사용했을 때의 NLP 태스크 실험 결과를 나타냅니다.

<p align="center">
<img src="https://i.ibb.co/D5pddMD/layer-ablation.png" alt="layer-ablation" border="0">
</p>

FFN의 출력이 가장 좋은 성능을 나타내고 셀프어텐션의 출력은 제대로 기능하지 못합니다. 이는 셀프어텐션이 residual connection 전에 있고 편향이 크기 때문이라고 추측됩니다.

<br><br>

## Reference

[Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu and Michael Auli. data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language. In ICML, 2022.](https://proceedings.mlr.press/v162/baevski22a.html)
