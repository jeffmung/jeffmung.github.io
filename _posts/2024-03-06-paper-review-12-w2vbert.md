---
layout: post
title: "[논문 리뷰] w2v-BERT: Combining Contrastive Learning and Masked Language Modeling for Self-Supervised Speech Pre-Training"
image: https://i.ibb.co/fkZkRLV/thumbnail.png
date: 2024-03-06
tags: 
categories: paper-review
use_math: true
---

<br><br>

## 논문 개요

<!-- excerpt-start -->

w2v-BERT는 레이블이 없는 음성 데이터에 대해 자기지도 학습(self-supervised learning)으로 표현(representation)을 추출하는 모델입니다. 같은 종류의 모델인 wav2vec 2.0과 [(Alexei Baevski et al., 2020)](https://proceedings.neurips.cc/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html) 많은 부분이 비슷하며, BERT에서 [(Jacob Devlin et al., 2019)](https://arxiv.org/abs/1810.04805) 사용하는 마스킹 예측(masked prediction)을 추가하고 트랜스포머를 컨포머(conformer)로 바꿈으로써 향상된 성능을 보여줍니다. wav2vec 2.0에 대한 자세한 설명은 [이전 포스트](https://jeffmung.github.io/2024/03/01/paper-review-11-wav2vec2/)를 참고하면 됩니다.

<br><br>

## Model Architecture

전체적인 모델 구조는 아래 그림에 요약되어 있습니다.

<p align="center">
<img src="https://i.ibb.co/myFtSbb/architecture.png" alt="architecture" border="0">
</p>

### Feature Encoder

특징 인코더(feature encoder)는 오디오 신호를 입력으로 받아 특징을 추출합니다. 오디오 신호 입력은 파형이 될 수도 있지만 이 논문의 실험에서는 로그 멜 스펙트로그램을 사용합니다. 신경망은 스트라이드가 (2, 2)인 컨볼루션 층 두개로 이루어져 있어 입력 시퀀스의 길이가 4배 축소됩니다.

### Contrastive Module

특징 인코더의 출력은 대조 모듈(contrastive module)의 컨포머 블럭과 양자화기(quantizer)로 각각 나누어져 들어갑니다. 컨포머[(Anmol Gulati et al., 2020)](https://arxiv.org/abs/2005.08100)는 CNN과 트랜스포머를 결합한 신경망으로 wav2vec 2.0에서는 트랜스포머가 같은 역할을 합니다. 컨포머 블럭의 입력에는 마스킹이 적용되고 출력은 연속적인 문맥 벡터입니다.

양자화기는 마스킹 되지 않은 입력을 받아 양자화된 벡터와 토큰 아이디를 출력합니다. 양자화된 벡터는 컨포머 블럭에서 출력된 문맥 벡터의 대조 학습 타겟으로 사용되고, 토큰 아이디는 이후에 마스킹 예측의 타겟으로 사용됩니다.

### Masked Prediction Module

대조 모듈의 컨포머 블럭에서 출력된 문맥 벡터는 마스킹 예측 모듈(masked prediction module)로 들어갑니다. 마스킹 예측 모듈은 대조 모듈에서 사용하는 것과 같은 구조의 컨포머 블럭들로 이루어져 있습니다. 여기에서 출력된 상위 레벨의 문맥 벡터는 양자화기에서 만들어진 토큰 아이디를 타겟으로 학습됩니다.

<br><br>

## Pre-training

사전학습(pre-training)에서는 레이블이 없는 데이터를 사용합니다.

### Contrastive Loss

대조 손실(contrastive loss)은 대조 모듈의 컨포머 블럭 출력과 양자화기를 통해 양자화된 특징 인코더의 출력 벡터 사이에서 계산됩니다. 학습 방식은 wav2vec 2.0에서 사용하는 것과 동일하고, 다른 점 한 가지는 마스킹된 타임스텝을 공유되는 학습 가능한 벡터가 아니라 랜덤 벡터로 대체한다는 것입니다. wav2vec 2.0에서 정의된 대조 손실을 $\small \mathcal{L}\_w$ 라 하고 코드북 다양성 손실(diversity loss)을 $\small \mathcal{L}\_d$ 라고 하면 대조 모듈에 대한 손실 함수는 다음과 같이 정의됩니다.

<br>
\begin{equation}
\mathcal{L}\_c = \mathcal{L}\_w + \alpha \cdot \mathcal{L}\_d
\end{equation}
<br>

$\small \alpha$ 는 두 손실 사이의 비율을 결정하는 하이퍼파라미터이고 기본값으로 0.1을 사용합니다.

### Masked Prediction Loss

대조 모듈의 컨포머 블럭에서 출력된 문맥 벡터는 마스킹 예측 모듈로 들어가서 최종 문맥 벡터를 생성합니다. 이 최종 문맥 벡터는 소프트맥스 층을 거쳐서 마스킹 예측 태스크에 사용됩니다. 손실 함수는 마스킹된 타임스텝의 문맥 벡터가 양자화기에서 만들어진 같은 타임스텝의 토큰 아이디를 예측하도록 하는 크로스 엔트로피(cross-entropy) 손실입니다. 이 손실을 $\small \mathcal{L}\_m$ 이라고 명명하면 w2v-BERT의 전체 손실 함수는 다음과 같습니다.

<br>
\begin{equation}
\mathcal{L}\_p = \beta \cdot \mathcal{L}\_c + \gamma \cdot \mathcal{L}\_m
\end{equation}
<br>

계수 $\small \beta$ 와 $\small \gamma$ 는 기본값을 1로 설정합니다.

<br><br>

## Fine-tuning

파인튜닝은 레이블이 있는 데이터에 대해 수행됩니다. 이 논문에서는 음성 인식(speech recognition) 태스크를 목표로 하고 사전학습된 모델의 마스킹 예측 모듈에서 나온 최종 문맥 벡터가 디코더에 연결되어 오디오에 대응되는 텍스트를 생성합니다. 기본 디코더로는 두 층 짜리 LSTM을 사용합니다. 특징 인코더를 제외한 사전학습된 모델과 디코더의 파라미터들을 파인튜닝하고, 일반적으로 음성 인식에서 성능 향상에 도움이 되는 몇 가지 테크닉들을 사용합니다.

먼저 기본적으로 SpecAugment의 [(Daniel S. Park et al., 2019)](https://arxiv.org/abs/1904.08779) 방법으로 마스킹을 적용하고 언어 모델 융합(language model fusion)을 [(Caglar Gulcehre et al., 2015)](https://arxiv.org/abs/1503.03535) 사용합니다. 언어 모델 융합의 기본 설정으로 사용하는 얕은 융합(shallow fusion)은 LSTM 디코더와 트랜스포머 언어 모델에서 나온 각각의 후보 문자들에 대한 소프트맥스 확률을 가중합 하는 방법입니다.

또한 Noisy Student Training에서 [(Daniel S. Park et al., 2020)](https://arxiv.org/pdf/2005.09629.pdf) 사용하는 자가학습(self-training)의 적용 유무에 따라 실험 결과를 비교합니다. 간단히 말하면 파인튜닝된 모델로 레이블이 없는 데이터셋에 레이블링을 하고, 그 데이터를 기존 데이터셋과 혼합하여 다시 파인튜닝 하는 것을 반복하는 방법입니다.

<br><br>

## 실험

레이블이 없는 데이터셋으로는 Libri-Light unlab-60k를 [(Jacob Kahn et al., 2020)](https://ieeexplore.ieee.org/abstract/document/9052942/) 사용하고 레이블이 있는 데이터셋으로는 LibriSpeech [(Vassil Panayotov et al., 2015)](https://ieeexplore.ieee.org/abstract/document/7178964) 960hr와 100hr 데이터셋을 사용합니다. 정량적인 평가 지표로는 word error rate(WER)를 사용합니다.

모델은 크기에 따라 w2v-BERT XL과 w2v-BERT XXL라고 명명한 두 종류를 사용합니다. 두 모델은 구조적으로는 컨포머 블럭의 수만 다르고 학습에 사용되는 세부적인 하이퍼파라미터에 약간 차이가 있습니다.

### Main Results

Libri-Light unlab-60k와 LibriSpeech 960hr 데이터셋에 대한 실험 결과는 아래 표에 정리되어 있습니다.

<p align="center">
<img src="https://i.ibb.co/t8ndvwM/main-results.png" alt="main-results" border="0">
</p>

먼저 언어 모델을 사용하는 HuBERT와 [(Wei-Ning Hsu et al., 2021)](https://ieeexplore.ieee.org/abstract/document/9585401?casa_token=rKpeKtDGu1UAAAAA:0stGrDQivqXS_FOezIWsclR417C2gEY78G_xzoWnSkPxqKdwW1z1V4XLskpFj5aSXZc1gzSY1t8) 결과를 비교해보면 w2v-BERT가 자가학습과 언어 모델을 사용하지 않았을 때에도 이미 비슷한 성능을 보여줍니다. 자가학습과 언어 모델을 적용한 w2v-BERT는 전체적으로 성능이 가장 우수합니다.

w2v-Conformer는 wav2vec 2.0에서 트랜스포머를 컨포머로 바꾼 모델입니다. 이 모델과 w2v-BERT의 가장 큰 차이는 사전학습에서의 마스킹 예측입니다. 두 모델의 결과를 비교해보면 대조 학습과 마스킹 예측을 같이 사용하는 것이 성능 향상에 도움이 된다는 것을 알 수 있습니다.

### Necessity of Contrastive Module

만약 대조 모듈 없이 마스킹 예측 모듈만 존재한다면 양자화기가 마스킹 예측 손실로만 학습될 것입니다. 이렇게 되면 양자화기가 마스킹된 타임스텝의 벡터를 모두 같은 코드북 벡터로 배정하는 무의미한 해(trivial solution)를 만들도록 학습될 가능성이 있습니다. 이 경우 마스킹 예측 손실이 매우 작아지더라도 모델이 유용한 표현을 학습하지 못하게 됩니다.

이러한 직관을 입증하기 위해 대조 모듈 없이 w2v-BERT XL를 학습시키는 실험을 진행했습니다. 24개의 컨포머 블럭이 모두 마스킹 예측 모듈을 구성하고 전체 손실 함수로는 마스킹 예측 손실과 코드북 다양성 손실만 사용합니다. 아래 그림은 왼쪽부터 학습 스텝에 따른 마스킹 예측 손실, 마스킹 예측 정확도, 그리고 코드북 다양성 손실을 그래프로 나타낸 것입니다.

<p align="center">
<img src="https://i.ibb.co/Kyj6h7P/contrastive-ablation.png" alt="contrastive-ablation" border="0">
</p>

모든 그래프는 대조 모듈이 없을 때 모델이 제대로 학습되지 않는다는 것을 보여줍니다. 특히 맨 오른쪽의 다양성 손실 값을 보면 직관적으로 예상한 대로 코드북의 엔트로피가 매우 낮아지는 코드 붕괴(code collapse) 현상이 일어나는 것을 알 수 있습니다.

### Impact of Contrastive Module's Capacity

대조 모듈의 크기가 음성 인식 태스크 성능에 미치는 영향을 보기 위한 실험도 진행했습니다. 대조 모듈과 마스킹 예측 모듈의 컨포머 블럭 수의 합을 24로 고정하고 대조 모듈에 $\small n$ 개의 컨포머 블럭이 있을 때의 모델을 $\small C_n$ 이라고 명명합니다. 예를 들어 $\small C_4$ 는 컨포머 블럭이 대조 모듈에 4개, 마스킹 예측 모듈에 20개 있는 모델입니다. 아래 표는 그 결과를 정리한 것입니다.

<p align="center">
<img src="https://i.ibb.co/TcX57x9/contrastive-size.png" alt="contrastive-size" border="0">
</p>

w2v-BERT XL 모델이 기본값으로 사용하는 $\small C_{12}$ 가 가장 좋은 성능을 나타내고 대조 모듈의 크기가 너무 작거나 큰 경우에는 성능에 악영향을 미치는 것을 볼 수 있습니다. 하지만 성능 차이가 크지는 않고 모든 경우에 다 wav2vec 2.0의 성능을 능가합니다.

### Results on Voice Search Traffic

w2v-BERT가 현실에서의 음성 신호에 대해서도 잘 작동하는지 보기 위해 Google의 Voice Search에 대해서도 실험을 진행합니다. 이 음성 데이터셋은 사람이 기기에게 어떤 기능을 수행하라고 명령하는 실제 음성들로 이루어져 있으므로 배경 노이즈나 침묵들이 포함되어 있고 평균 길이도 약 5초 정도로 짧기 때문에 난이도가 높습니다. 실험 결과는 아래 표에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/Hgz6wLq/voice-search.png" alt="voice-search" border="0">
</p>

w2v-Conformer와 비교했을 때 w2v-BERT가 더 우수한 성능을 보여줍니다.

<br><br>

## Reference

[Yu-An Chung, Yu Zhang, Wei Han, Chung-Cheng Chiu, James Qin, Ruoming Pang and Yonghui Wu. w2v-BERT: Combining Contrastive Learning and Masked Language Modeling for Self-Supervised Speech Pre-Training. In ASRU, 2021.](https://ieeexplore.ieee.org/abstract/document/9688253)
