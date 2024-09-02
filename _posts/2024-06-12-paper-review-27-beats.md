---
layout: post
title: "BEATs: Audio Pre-Training with Acoustic Tokenizers [ICML, 2022]"
description: BEATs 논문 리뷰. BEATs는 acoustic tokenizer로 만들어진 descritized token label을 audio SSL 모델이 예측하도록 하는 self-supervised learning 모델입니다.
image: https://i.ibb.co/pP4Mv0D/thumbnail.png
date: 2024-06-12
tags: 
categories: paper-review
use_math: true
---

## 논문 개요

<!-- excerpt-start -->

BEATs는 이산화된(discretized) 토큰 레이블을 예측하도록 하는 방법을 오디오 도메인에 처음으로 적용한 자기지도학습(self-supervised learning) 모델입니다. 오디오는 자연어 처리나 음성과는 달리 명백하게 구분되는 토큰들이 존재하지 않기 때문에 이전의 많은 연구들이 이산적인 토큰 예측 방식을 사용하지 않고 마스킹된 부분을 복원하는 것을 목표로 자기지도학습을 했습니다. 하지만 이렇게 복원 손실(reconstruction loss)로 학습하는 경우 모델이 저차원의 특징들을 맞추는 데 집중해서 고차원의 의미를 담고 있는 추상 특징들을 무시하게 되는 경향이 있습니다.

이 논문에서는 $\small \text{acoustic tokenizer}$ 와 $\small \text{audio SSL}$ 모델이 서로를 번갈아서 학습시키는 파이프라인을 제안합니다. $\small \text{acoustic tokenizer}$ 는 $\small \text{audio SSL}$ 모델에 예측 타겟으로 사용할 수 있는 이산화된 레이블을 제공해주고 반대로 $\small \text{audio SSL}$ 모델은 $\small \text{acoustic tokenizer}$ 가 오디오의 의미적인 특징을 담아서 토큰을 만들 수 있도록 지식 증류(knowledge distillation)의 교사(teacher) 모델로 작용합니다.

많은 오디오 분류 태스크에 대한 실험에서 BEATs는 다른 모델들에 비해 우수한 성능을 보여줍니다. 또한 오디오 도메인에서 이산적인 레이블의 예측을 통해 학습하는 방법을 제안하여 같은 방법이 널리 사용되고 있는 컴퓨터 비전이나 자연어 처리 도메인과의 멀티모달 학습으로도 확장시킬 수 있는 가능성을 열어주는 연구입니다.

<br><br>

## Iterative Audio Pre-training

아래 그림은 BEATs의 반복적인 학습 프레임워크를 보여줍니다.

<p align="center">
<img src="https://i.ibb.co/gF3CPgk/framework.png" alt="framework" border="0">
</p>

각각의 반복(iteration)에서, 레이블이 없는 오디오가 주어졌을 때 $\small \text{acoustic tokenizer}$ 가 이산적인 레이블을 만들어주고  $\small \text{audio SSL}$ 모델은 그것을 사용해서 마스킹된 부분의 레이블을 예측하는 손실로 학습합니다. 모델이 수렴하고 나면 $\small \text{audio SSL}$ 모델을 교사 모델로 사용하여 새로운 $\small \text{acoustic tokenizer}$ 에 대한 지식 증류를 해줍니다.

구체적으로는 오디오 클립이 주어지면 멜스펙트로그램으로 변환한 뒤 일정한 크기의 그리드 패치로 나눠줍니다. 그리고 평평하게 하여(flatten) 패치 시퀀스 $\small \mathbf{X} = \\{ \mathbf{x}\_t \\}\_{t=1}^T$ 를 만듭니다. 그 뒤 $\small \text{audio SSL}$ 모델을 학습시키기 위해 $\small \text{acoustic tokenizer}$ 를 사용해 패치 시퀀스 $\small \mathbf{X}$ 를 패치 레벨의 이산적인 레이블 $\small \hat{Z} = \\{ \hat{z}\_t \\}\_{t=1}^T$ 로 만들어 마스킹 예측 타겟으로 쓸 수 있게 해줍니다.

$\small \text{acoustic tokenizer}$ 를 학습시킬 때에는 $\small \text{audio SSL}$ 모델을 사용하여 패치 시퀀스 $\small \mathbf{X}$ 를 인코딩한 시퀀스 $\small \hat{\mathbf{O}} = \\{ \hat{o}\_t \\}\_{t=1}^T$ 를 지식 증류 타겟으로 쓰도록 합니다. 교사 모델로 사용하는 $\small \text{audio SSL}$ 모델은 자기지도학습으로 사전학습된 것 뿐만 아니라 지도학습으로 파인튜닝된 것도 사용할 수 있습니다.

이렇게 번갈아가며 학습하는 과정을 통해 $\small \text{acoustic tokenizer}$ 는 의미 정보를 많이 담고 있는 $\small \text{audio SSL}$ 의 인코딩을 이산적인 레이블 생성에 반영할 수 있고 $\small \text{audio SSL}$ 모델은 이러한 이산적인 레이블을 활용하는 이점을 얻게 됩니다. 이 과정은 모델이 완전히 수렴할 때까지 반복됩니다.

<br><br>

## Acoustic Tokenizer

첫 번째 반복에서는 $\small \text{acoustic tokenizer}$ 를 학습시키기 위한 교사 모델이 주어지지 않기 때문에 Random-Projection Tokenizer [(Chung-Cheng Chiu et al., 2022)](https://arxiv.org/pdf/2202.01855) 를 사용해 연속적인 오디오 특징을 이산적인 레이블로 클러스터링합니다. 두 번째 반복부터는 직전 반복에서 얻어진 사전학습되거나 파인튜닝된 $\small \text{audio SSL}$ 모델을 지식 증류의 교사 모델로 사용하여 $\small \text{acoustic tokenizer}$ 가 의미 정보를 반영한 이산적인 레이블을 생성할 수 있도록 합니다.

### Cold Start: Random-Projection Tokenizer

Random-Projection Tokenizer의 구조는 아래 그림에 요약되어 있습니다.

<p align="center">
<img src="https://i.ibb.co/RPTn0vm/random-projection-tokenizer.png" alt="random-projection-tokenizer" border="0">
</p>

입력으로 들어간 패치 시퀀스 $\small \mathbf{X} = \\{ \mathbf{x}\_t \\}\_{t=1}^T$ 는 프로젝션 층을 통해 벡터 시퀀스 $\small \mathbf{W}\mathbf{x}\_t$ 로 변환됩니다. 그리고 랜덤하게 초기화된 $\small K$ 개의 벡터 $\small \mathbf{V} = \\{ \mathbf{v}\_i \\}\_{i=1}^K$ 를 포함한 코드북을 이용하여 가장 가까운 코드 벡터에 배정합니다. $\small t$ 번째 패치의 이산적인 레이블은 가장 가까운 벡터의 인덱스 $\small \hat{z}\_t = \arg \min\_i \Vert \mathbf{v}\_i - \mathbf{W}\mathbf{x}\_t \Vert \_2^2$ 로 정의됩니다.

### Iteration: Self-Distilled Tokenizer

두 번째 반복부터의 지식 증류를 통한 $\small \text{acoustic tokenizer}$ 학습 과정은 아래 그림에 요약되어 있습니다.

<p align="center">
<img src="https://i.ibb.co/s1Hv2H1/self-distilled-tokenizer.png" alt="self-distilled-tokenizer" border="0">
</p>

먼저 트랜스포머 기반의 $\small \text{tokenizer encoder}$ 를 사용하여 입력 패치 시퀀스 $\small \mathbf{X} = \\{ \mathbf{x}\_t \\}\_{t=1}^T$ 를 벡터 시퀀스 $\small \mathbf{E} = \\{ \mathbf{e}\_{t=1}^T \\}$ 로 인코딩합니다. 그 뒤 각각의 인코딩된 벡터 $\small \mathbf{e}\_{t=1}^T$ 를 코드북 임베딩 벡터 $\small \mathbf{V} = \\{ \mathbf{v} \\}\_{i=1}^K$ 중 가장 가까운 것으로 배정하여 이산적인 레이블 $\small \hat{z}\_t = \arg \min\_i \Vert \ell\_2(\mathbf{v}\_i) - \ell\_2(\mathbf{e}\_t) \Vert \_2^2$ 를 얻습니다. 여기서 $\small \ell\_2$ 는 L2 정규화를 의미합니다.

이렇게 벡터 양자화를 할 때는 VQ-VAE에서 하는 것과 같이 미분 불가능한 문제를 처리하기 위해 straight-through gradient를 사용하고 코드북 임베딩을 더 안정적으로 학습시키기 위해 exponential moving average를 적용합니다. 코드북 학습에 대한 손실 함수는 다음과 같습니다.

<br>
\begin{align}
\mathcal{L} = \Vert \text{sg} [ \ell\_2(\mathbf{e}\_t) ] - \ell\_2 (\mathbf{v}\_{\hat{z}\_t}) \Vert\_2^2 + \Vert \ell\_2(\mathbf{e}\_t) - \text{sg} [ \ell\_2 (\mathbf{v}\_{\hat{z}\_t}) ] \Vert\_2^2
\end{align}
<br>

양자화된(quantized) 벡터 시퀀스 $\small \mathbf{E}^q = \\{ \mathbf{v}\_{\hat{z}\_t} \\}\_{t=1}^T$ 는 3층 짜리 트랜스포머 구조의 $\small \text{tokenizer estimator}$ 로 들어가서 교사 모델의 출력 $\small \\{ \hat{\mathbf{o}}\_t \\}\_{t=1}^T$ 를 예측하도록 합니다. 예측의 목적 함수는 두 벡터 시퀀스 사이의 코사인 유사도를 최대화하도록 하는 것입니다.

추론(inference) 시에는 $\small \text{tokenizer estimator}$ 를 떼어내고 사전학습된 $\small \text{tokenizer encoder}$ 와 코드북 임베딩을 이용하여 입력 오디오 패치 시퀀스 $\small \mathbf{X} = \\{ \mathbf{x}\_t \\}\_{t=1}^T$ 를 패치 레벨의 이산적인 레이블 $\small \hat{Z} = \\{ \hat{z}\_t \\}\_{t=1}^T$ 로 만듭니다.

<br><br>

## Audio SSL Model

### Backbone

백본 네트워크로는 선형 프로젝션 층과 트랜스포머 인코더 블럭들로 이루어진 ViT 구조를 사용합니다. 먼저 입력 오디오 패치 $\small \mathbf{X} = \\{ \mathbf{x}\_t \\}\_{t=1}^T$ 를 선형 프로젝션 층을 통해 패치 임베딩 $\small \mathbf{E} = \\{ \mathbf{e}\_t \\}\_{t=1}^T$ 로 변환시키고 트랜스포머 인코더를 통해 인코딩된 패치 표현 $\small \mathbf{R} = \\{ \mathbf{r}\_t \\}\_{t=1}^T$ 를 얻습니다.

트랜스포머는 컨볼루션 기반의 상대적 위치 임베딩(relative positional embedding)과 gated relative position bias를 [(Zewen Chi et al., 2022)](https://arxiv.org/pdf/2106.16138) 맨 아래에 사용하고 안정적인 학습을 위해 DeepNorm을 [(Hongyu Wang et al., 2022)](https://ieeexplore.ieee.org/abstract/document/10496231) 적용합니다.

### Pre-Training

아래 그림은 $\small \text{audio SSL}$ 모델의 사전학습 과정을 보여줍니다.

<p align="center">
<img src="https://i.ibb.co/7CtvSM4/pretraining.png" alt="pretraining" border="0">
</p>

먼저 입력 패치 시퀀스 $\small \mathbf{X} = \\{ \mathbf{x}\_t \\}\_{t=1}^T$ 와 이산적인 타겟 레이블 $\small \hat{Z} = \\{ \hat{z}\_t \\}\_{t=1}^T$ 가 주어졌을 떄 임의로 75%의 패치를 마스킹합니다. 이때 마스킹된 위치를 $\small \mathcal{M} = \\{1, \ldots, T \\}^{0.75T}$ 라고 명명합니다. 그리고 마스킹되지 않은 패치 시퀀스 $\small \mathbf{X}^U = \\{\mathbf{x}\_t : t \notin \mathcal{M} \\}\_{t=1}^T$ 를 ViT 인코더에 넣어서 인코딩된 표현 $\small \mathbf{R}^U = \\{\mathbf{r}\_t : t \notin \mathcal{M} \\}\_{t=1}^T$ 를 얻습니다.

마지막으로, 마스킹되지 않은 패치 표현과 마스킹된 패치를 합친 시퀀스 $\small \\{ \mathbf{r}\_t : t \notin \mathcal{M}\\}\_{t=1}^T \cup \\{ \boldsymbol{0} : t \in \mathcal{M}\\}\_{t=1}^T$ 를 $\small \text{label predictor}$ 에 넣어서 이산적인 레이블 $\small Z = \\{z\_t \\}\_{t=1}^T $ 를 예측하도록 합니다. 손실 함수는 다음과 같이 마스킹된 위치의 레이블에 대한 음의 로그우도의 합입니다.

<br>
\begin{equation}
\mathcal{L} = -\sum\_{t\in\mathcal{M}} \log p(\hat{z}\_t \vert \mathbf{X}^U)
\end{equation}
<br>

### Fine-Tuning

$\small \text{audio SSL}$ 모델을 파인튜닝 할 때에는 $\small \text{label predictor}$ 를 떼어내고 태스크에 대한 선형 분류기(linear classifier)를 ViT 인코더의 위에 붙여서 다운스트림 분류 태스크의 레이블을 만들도록 합니다. 이 구조는 아래 그림에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/RDFXYxK/finetuning.png" alt="finetuning" border="0">
</p>

먼저 입력 멜스펙트로그램의 시간과 주파수 도메인에 대하여 spec-augmentation을 [(Daniel S. Park et al., 2019)](https://arxiv.org/abs/1904.08779) 적용하고 패치 시퀀스 $\small \mathbf{X} = \\{\mathbf{x}\_t \\}\_{t=1}^T$ 로 나눕니다. 그리고 사전학습과는 다르게 전체 패치 시퀀스 $\small \mathbf{X}$ 를 ViT 인코더에 넣어서 인코딩된 표현 $\small \mathbf{R} = \\{\mathbf{r}\_t\\}\_{t=1}^T$ 를 얻습니다.

마지막으로, 선형분류기의 출력에 평균 풀링과 소프트맥스를 취해서 카테고리 확률 $\small p(C) = \text{Softmax}(\text{MeanPool}(\mathbf{W}\_c\mathbf{R}))$ 을 얻습니다. 단일 레이블 분류 태스크에 대해서는 크로스 엔트로피 손실을, 다중 레이블 분류 태스크에 대해서는 이진 크로스 엔트로피 손실을 적용합니다.

<br><br>

## Experiments

### Dataset

AudioSet 데이터셋으로 사전학습된 모델들을 여섯 개의 다운스트림 태스크에 대해서 평가합니다. 다운스트림 태스크는 오디오 분류 태스크인 AS-2M, AS-20K, ESC-50과 음성 분류 태스크인 KS1, KS2, 그리고 ER을 사용합니다.

### Backbone

BEATs 모델은 은닉 차원 768, 어텐션 헤드 8개의 12층 짜리 트랜스포머 구조를 갖습니다. 모델 파라미터 수는 약 90M개로 다른 비교 모델들과 비슷한 크기를 갖도록 설계한 것입니다.

### Acoustic Feature

입력 샘플은 128개 주파수 구간의 멜스펙트로그램으로 만들고 $\small 16 \times 16$ 패치로 나눠서 평평하게 만들어 모델에 집어넣습니다.

### Model and Tokenizer Training

BEATs 모델을 AS-2M 데이터셋으로 사전학습할 때 세 번의 반복에 대해 각각 $\small \text{BEATs\_{iter1}}$ , $\small \text{BEATs\_{iter2}}$ , $\small \text{BEATs\_{iter3}}$ 이라고 명명합니다. 또한 세 번째 반복에서 $\small \text{BEATs\_{iter2}}$ 의 $\small \text{audio SSL}$ 모델을 지도학습으로 파인튜닝한 것으로 $\small \text{acoustic tokenizer}$ 를 학습한 모델은 $\small \text{BEATs\_{iter3+}}$ 라고 명명합니다.

### Comparing with the SOTA Single Models

BEATs를 다른 모델들과 비교한 실험 결과는 아래 표에 정리되어 있습니다.

<p align="center">
<img src="https://i.ibb.co/j3KLg99/result-table.png" alt="result-table" border="0">
</p>

IN, AS, LS는 각각 ImageNet, AudioSet, 그리고 LibriSpeech 데이터셋을 의미합니다. CLAP과 CLIP의 결과에서 TA와 TI는 각각 text-audio 쌍과 text-image 쌍을 나타냅니다. 회색으로 쓰여진 결과들은 추가적인 데이터셋으로 지도학습 파인튜닝을 한 모델들입니다.

결과를 보면 BEATs가 첫 번째 반복부터 이미 상당히 높은 성능을 보여주고 반복이 더 진행될 때 추가적인 성능 향상도 나타냅니다.

<br><br>

## Reference

[Sanyuan Chen, Yu Wu, Chengyi Wang, Shujie Liu, Daniel Tompkins, Zhuo Chen, Wanxiang Che, Xiangzhan Yu and Furu Wei. BEATS: Audio Pre-Training with Acoustic Tokenizers. In ICML, 2023.](https://arxiv.org/abs/2212.09058)
