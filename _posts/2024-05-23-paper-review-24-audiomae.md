---
layout: post
title: "Masked Autoencoders that Listen [NeurIPS, 2022]"
description: Audio-MAE는 Masked Autoencoder를 오디오 스펙트로그램의 특성에 맞게 변형하여 오디오 도메인에서의 성능을 높인 자기지도학습 모델입니다.
image: https://i.ibb.co/S3F8dh0/thumbnail.png
date: 2024-05-23
tags: 
categories: paper-review
use_math: true
---

## 논문 개요

<!-- excerpt-start -->

Audio-MAE는 Masked Autoencoder (MAE)를 오디오 스펙트로그램의 특성에 맞게 변형하여 오디오 도메인에서의 성능을 높인 자기지도학습(self-supervised learning) 모델입니다. 오디오 분류 태스크에서 우수한 성능을 보여줬습니다.

<br><br>

## Audio Masked Autoencoders (Audio-MAE)

Audio-MAE의 전체 모델 구조는 아래 그림에 요약되어 있습니다.

<p align="center">
<img src="https://i.ibb.co/4W9dBrc/architecture.png" alt="architecture" border="0">
</p>

### Spectrogram Patch Embeddings

모델의 입력은 멜스펙트로그램이며 서로 겹치지 않는 그리드 패치로 나눠집니다. 패치들은 평평해진(flattened) 뒤 선형 프로젝션(linear projection)으로 임베딩됩니다. 임베딩된 패치에는 고정된 사인 위치 임베딩이 더해집니다.

### Masking Strategies

아래 그림과 같이 다양한 마스킹 전략을 사용했을 때의 성능을 비교합니다.

<p align="center">
<img src="https://i.ibb.co/6ydJSrF/masking.png" alt="masking" border="0">
</p>

경험적으로는 사전학습에는 Unstructured 마스킹을 높은 비율로 사용하고 파인튜닝 때는 Time+frequency 마스킹을 낮은 비율로 사용하는 것이 가장 우수한 성능을 나타냅니다.

### Encoder

인코더로는 12 층의 ViT-Base (ViT-B) [(Alexey Dosovitskiy et al., 2021)](https://openreview.net/forum?id=YicbFdNTTy&utm_campaign=f86497ed3a-EMAIL_CAMPAIGN_2019_04_24_03_18_COPY_01&utm_medium=email&utm_source=Deep%20Learning%20Weekly&utm_term=0_384567b42d-f86497ed3a-72965345) 트랜스포머를 사용합니다. 계산량을 줄이기 위해 마스킹 되지 않은 패치들만 인코더에 들어갑니다.

### Decoder with Local Attention

디코더도 마찬가지로 트랜스포머입니다. 인코딩된 패치들과 학습 가능한 마스크 토큰이 원래 순서대로 다시 배열된 뒤 고정된 사인 위치 임베딩을 더해서 디코더에 넣어줍니다. 디코더의 마지막에는 선형 층이 있어 원래의 스펙트로그램 패치를 타겟으로 재구성하고 예측하도록 합니다.

이미지 도메인에서는 트랜스포머 디코더가 일반적으로 전역 셀프 어텐션을 사용하지만 오디오에서는 국소적인 정보가 중요합니다. 따라서 스펙트로그램 패치들을 국소적인 윈도우로 나눠서 어텐션을 계산하는 방법을 사용합니다.

첫 번째로 Shifted window 방법은 인접한 트랜스포머 디코더 층에서 윈도우를 50%씩 이동시키는 것입니다. 윈도우의 이동은 아래 그림과 같이 왼쪽 위 방향으로 이루어집니다.

<p align="center">
<img src="https://i.ibb.co/89CwmxZ/shift.png" alt="shift" border="0">
</p>

두 번째로 Hybrid window 방법은 마지막 몇 층에서는 전역 어텐션을 사용하고 나머지에서는 국소 어텐션을 사용합니다. 이 방법은 마지막 재구성 층이 전역 정보를 포함하게 만들어줍니다.

### Objective

Audio-MAE의 목적은 마스킹된 부분의 예측된 스펙트로그램과 원래 입력 사이의 MSE를 최소화하는 것입니다.

### Fine-tuning for Downstream Tasks

파인튜닝 단계에서는 디코더를 제외하고 인코더만 학습합니다. 파인튜닝 시에도 마스킹을 적용하는데 마찬가지로 마스킹된 부분은 별도의 마스크 토큰으로 두는 것이 아니라 입력에서 아예 제외시킵니다. 마스킹 되지 않은 패치들로부터 인코딩된 벡터 시퀀스는 평균 풀링된 뒤 선형 층을 통해 분류 태스크에 파인튜닝됩니다.

<br><br>

## 실험

실험은 다양한 오디오 분류 태스크에 대해서 진행합니다.

### Datasets and Tasks

AudioSet (AS-2M, AS-20K)은 2백만개의 10초 짜리 유튜브 클립이 포함된 오디오 분류 데이터셋입니다. 527 종류의 오디오 이벤트 레이블이 클립마다 달려 있고 여러 개의 레이블을 갖는 것도 가능합니다. AS-2M 실험에는 200만개의 unbalanced 세트와 2만개의 balanced 세트를 합쳐서 사전학습과 파인튜닝에 모두 사용하고 AS-20K 실험에는 사전학습에 AS-2M을 사용하고 파인튜닝에 balanced 세트를 사용합니다.

Environemental Sound Classification (ESC-50)은 2000개의 5초 짜리 환경 소리 녹음을 포함하고 있는 오디오 분류 데이터셋입니다. 50개의 클래스가 존재합니다.

Speech Commands (SPC-2, SPC-1)는 키워드 발견 태스크입니다. 1초 짜리 녹음이 약 10만 개 있고 키워드가 말해지는지 감지하는 것이 목표입니다.
VoxCeleb (SID)는 발화자를 분류하는 태스크로 1251명의 발화자가 말한 15만 개의 녹음을 포함합니다.

### Implementation Details

입력 데이터는 16k 샘플 레이트의 파형을 128 Kaldi-compatible 멜 밴드로 변환한 멜스펙트로그램을 사용합니다. 예를 들어 Audioset의 10초 짜리 녹음에 대해서 전처리된 스펙트로그램은 (1, 1024, 128) 차원의 모양을 갖습니다.

패치 임베딩은 (16, 16) 크기의 커널과 스트라이드의 컨볼루션으로 구현합니다. 따라서 각각의 패치들은 서로 겹치지 않습니다. 기본적으로 사전학습에는 0.8의 랜덤 마스킹 비율을 사용하고 파인튜닝에는 시간과 주파수에 대해 각각 0.3의 마스킹 비율을 사용합니다.

### Ablations and Model Properties

아래 그래프는 사전학습과 파인튜닝 시의 마스킹 전략들을 비교한 결과를 보여줍니다.

<p align="center">
<img src="https://i.ibb.co/FJLQ2p3/strategies-result.png" alt="strategies-result" border="0">
</p>

사전학습에서는 unstructured 마스킹과 0.8의 높은 비율이 가장 좋은 성능을 나타냅니다. 반면 파인튜닝에서는 time+frequency 마스킹과 상대적으로 낮은 비율인 0.3일 때가 가장 우수합니다. 높은 마스킹 비율로 structured 마스킹을 사용하면 난이도가 너무 높아지기 때문에 성능이 좋지 못한 것으로 추정됩니다.

### Impact of Patch Size and Stride

아래 표는 패치 크기와 스트라이드에 따른 결과를 보여줍니다.

<p align="center">
<img src="https://i.ibb.co/59GGmHK/patch-result.png" alt="patch-result" border="0">
</p>

패치 크기는 (16, 16)인 것이 적합하고 패치가 중첩될 때 성능에는 변화가 없지만 연산 속도가 훨씬 느려지는 것을 알 수 있습니다.

### Encoder

예상 가능한 것이지만 인코더 모델 사이즈가 커질수록 성능이 증가하고 계산과 메모리 비용도 같이 증가합니다. ViT-B가 적절하다는 것을 아래 표에서 볼 수 있습니다.

<p align="center">
<img src="https://i.ibb.co/7ykyQ6T/enc-result.png" alt="enc-result" border="0">
</p>

### Decoder

디코더에서는 어텐션 종류를 여러 가지 비교하고 결과는 아래 표에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/zRmvkCj/attention-result.png" alt="attention-result" border="0">
</p>

국소 어텐션이 가장 좋은 성능을 보여주고 하이브리드도 전역 어텐션보다는 우세합니다.

### Pre-training Data and Setup

사전학습 데이터셋 크기와 학습량에 대한 비교 실험 결과는 아래 표에서 볼 수 있습니다.

<p align="center">
<img src="https://i.ibb.co/56jn4ZZ/pretrain-result.png" alt="pretrain-result" border="0">
</p>

### Out-of-domain Pre-training on ImageNet

아래 표는 ImageNet 데이터셋에 대해 사전학습을 했을 때 오디오 분류 태스크에서의 성능을 보여줍니다.

<p align="center">
<img src="https://i.ibb.co/1qM6BQ9/imagenet-result.png" alt="imagenet-result" border="0">
</p>

일반적인 이미지와는 다른 오디오 도메인의 특성 때문에 ImageNet에서의 사전 학습이 별로 도움이 되지 못한다는 것을 알 수 있습니다.

### Comparison with the State-of-the-art

아래 표는 Audio-MAE를 다른 모델들과 비교한 전체 결과를 보여줍니다.

<p align="center">
<img src="https://i.ibb.co/0YYMTyd/result-table.png" alt="result-table" border="0">
</p>

AudioSet 데이터셋으로 사전학습된 Audio-MAE는 사전학습된 다른 자기지도학습 모델들과 사전학습 없이 지도학습된 모델들을 모두 포함하여 가장 우수한 성능을 나타냅니다. 또한 ImageNet에 지도학습으로 사전학습된 모델들과 비교했을 때에도 더 우세합니다.

### Visualization Examples by Audio-MAE Decoder

아래 그림은 AudioSet-2M eval 데이터셋에서의 디코더 재구성 결과를 보여줍니다.

<p align="center">
<img src="https://i.ibb.co/X8T9cTy/examples.png" alt="examples" border="0">
</p>

Audio-MAE가 다양한 마스킹 전략들에 대해서 합리적인 재구성을 한다는 것을 알 수 있습니다.

<br><br>

## Reference

[Po-Yao Huang, Hu Xu, Juncheng Li, Alexei Baevski, Michael Auli, Wojciech Galuba, Florian Metze and Christoph Feichtenhofer. Masked Autoencoders that Listen. In NeurIPS, 2022.](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b89d5e209990b19e33b418e14f323998-Abstract-Conference.html)

[Official Source Code of Audio-MAE](https://github.com/facebookresearch/AudioMAE)