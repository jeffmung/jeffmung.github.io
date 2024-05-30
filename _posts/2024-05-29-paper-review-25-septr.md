---
layout: post
title: "SepTr: Separable Transformer for Audio Spectrogram Processing [Interspeech, 2022]"
description: SepTr은 ViT를 오디오 도메인의 스펙트로그램에 적용할 때 시간과 주파수 차원의 어텐션 계산을 따로 하도록 설계한 모델입니다.
image: https://i.ibb.co/zSRtV73/thumbnail.png
date: 2024-05-29
tags: 
categories: paper-review
use_math: true
---

## 논문 개요

<!-- excerpt-start -->

SepTr은 Vision Transformer (ViT)를 오디오 도메인의 스펙트로그램에 적용할 때 시간과 주파수 차원의 어텐션 계산을 따로 하도록 설계한 모델입니다. 일반적인 이미지와 다르게 스펙트로그램에서는 시간과 주파수 축이 서로 구분되는 특징을 가지고 있기 때문에 이러한 분리된 트랜스포머 디자인이 더 적합하다고 추측할 수 있는데, 실험적으로도 다양한 오디오 분류 태스크에서 기존의 ViT보다 향상된 성능을 논문에서 보여줍니다. 이러한 설계는 또한 트랜스포머의 파라미터 수도 줄일 수 있기 때문에 속도 측면에서도 이점이 있습니다.

<br><br>

## Data Representation

모델에 들어가는 입력은 STFT로 변환된 멜스펙트로그램입니다.

<br><br>

## Overview of the Architecture

SepTr은 두 개의 연속된 트랜스포머 구조로 이루어져 있습니다. 각각의 트랜스포머는 서로 다른 축으로 작동하는데 시간과 주파수 축의 트랜스포머 사이의 순서는 고정될 필요가 없습니다. 아래의 그림은 둘 중 한 가지 예시로 시간 축의 트랜스포머가 먼저 있는 모델 구조를 나타냅니다.

<p align="center">
<img src="https://i.ibb.co/PT0DfL5/architecture.png" alt="architecture" border="0">
</p>

트랜스포머 블락이 L번 반복되고 마지막에는 mean \[CLS\] 토큰이 MLP를 통과하여 최종 출력을 만듭니다.

<br><br>

## Tokenization and Linear Projection

$\small k \cdot p \in \mathbb{N}$ 개의 주파수 구간과 $\small n \cdot p \in \mathbb{N}$ 개의 시간 프레임으로 이루어진 스펙트로그램 입력은 크기 $\small p \times p$ 의 정사각형 패치(토큰) $\small k \cdot n$ 개로 나눠집니다. 이 논문의 실험에서는 패치 크기를 $\small 1 \times 1$ 로 설정해서 가장 작은 레벨에서 셀프 어텐션이 계산될 수 있도록 합니다.

토큰들은 선형 프로젝션 층(linear projection layer)을 통해 $\small d$ 차원 벡터들로 변환됩니다. 프로젝션 층의 출력은 $\small T \in \mathbb{R}^{k \times n \times d}$ 의 텐서이고 $\small \mathbf{T}_{i, j} \in \mathbb{R}^d$ , $\small \forall i \in \\{1, 2, \ldots, k\\}$ , $\small \forall j \in \\{1, 2, \ldots, n\\}$ 는 프로젝트된 토큰을 나타냅니다.

<br><br>

## Vertical Transformer

프로젝트된 토큰들을 시간 축으로 나누어서 $\small n$ 개의 배치 $\small \mathbf{T}\_{:, j} = [\mathbf{T}\_{1, j}, \mathbf{T}\_{2, j}, \ldots, \mathbf{T}\_{k, j}] \in \mathbb{R}^{k \times d}$ 로 만듭니다. 각각의 배치는 $\small k$ 개의 토큰들로 이루어져 있습니다. 모든 배치의 맨 앞에는 복제된 $\small n$ 개의 클래스 토큰 $\small \mathbf{T}\_{\text{[CLS]}} \in \mathbb{R}^d$ 을 추가합니다. 배치들은 학습 가능한 길이 $\small k$ 의 위치 임베딩(positional embedding)이 더해진 뒤 $\small \text{Vertical Transformer}$ 에 들어갑니다.

<br><br>

## Horizontal Transformer

$\small \text{Vertical Transformer}$ 의 출력은 텐서 $\small \mathbf{T}^V \in \mathbb{R}^{n \times k \times d}$ 와 클래스 토큰에 평균 풀링(average pooling)을 적용한 $\small \hat{\mathbf{T}}^V_{[\text{CLS}]}$ 로 분리됩니다. $\small \mathbf{T}^V$ 는 주파수 축으로 나눠져서 $\small k$ 개의 샘플 $\small \mathbf{T}^V\_{i, :} = [\mathbf{T}^V\_{i, 1}, \mathbf{T}^V\_{i, 2}, \ldots, \mathbf{T}^V\_{i, n}] \in \mathbb{R}^{n \times d}$ 를 포함한 배치가 됩니다.

클래스 토큰 $\small \hat{\mathbf{T}}^V_{[\text{CLS}]}$ 은 $\small k$ 개로 복제되어 각각의 샘플 $\small \mathbf{T}^V\_{i, :}$ 의 맨 앞에 추가됩니다. 배치는 학습 가능한 위치 임베딩이 더해진 뒤 $\small \text{Horizontal Transformer}$ 로 들어갑니다.

<br><br>

## Experiments

### Datasets

오디오 분류 태스크를 위한 세 가지 데이터셋을 실험에 사용합니다. ESC-50 데이터셋은 50개의 클래스를 가진 2000개의 5초 짜리 일반적인 소리 이벤트 샘플들로 이루어져 있습니다. Speech Command V2 (SCV2)는 105,829개의 1초짜리 발화들로 이루어져 있고 키워드가 들어있는지 탐지하는 태스크를 위한 데이터셋입니다. CREMA-D는 멀티모달 데이터셋으로 91명의 배우가 정해진 12개의 문장을 말하는 7442개의 비디오에서 감정을 분류하는 것이 목표입니다.

### Evaluation Setup

모든 실험에서 평가 지표는 분류 정확도(accuracy)입니다. 또한 모든 실험에 대해서 noise perturbation, time shifting, speed perturbation, mix-up, 그리고 SpecAugment의 데이터 증강 기법이 사용됩니다. SepTr 모델은 패치 크기를 $\small 1 \times 1$ 크기로 설정하지만 비교 모델인 ViT는 $\small 8 \times 8$ 로 설정합니다. ViT에는 메모리 한계 때문에 $\small 1 \times 1$ 의 패치 크기를 적용할 수 없습니다.

### Ablation Study

먼저 CREMA-D 데이터셋에 대해서 $\small \text{Vertical Transformer}$ 와 $\small \text{Horizontal Transformer}$ 를 각각 단일로 사용했을 때와 둘 다 사용했을 때를 비교합니다. 그 결과는 아래 표에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/RB8BRBm/cremad-result.png" alt="cremad-result" border="0">
</p>

각각 단일로 사용한 SepTr-V와 Septr-H는 ViT보다도 성능이 떨어지고 둘 다 같이 사용한 SepTr-HV와 SepTr-VH는 순서에 상관없이 성능 향상이 있는 것을 볼 수 있습니다.

### Effectiveness

아래 표에는 다른 두 데이터셋에 대한 결과가 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/SrhnFmQ/result-all.png" alt="result-all" border="0">
</p>

모든 데이터셋에 대해서 SepTr이 가장 우수한 성능을 보여줍니다.

### Efficiency

아래 그림은 입력 스펙트로그램 크기에 대한 모델 파라미터 수의 변화를 보여줍니다.

<p align="center">
<img src="https://i.ibb.co/0B3xLCy/n-parameters.png" alt="n-parameters" border="0">
</p>

ViT와 비교했을 때 SepTr은 입력 크기가 커져도 파라미터 수 증가 정도가 매우 적습니다. 예를 들어 $\small 512 \times 512$ 크기의 입력에 대해 SepTr은 9.4M 개의 파라미터를 갖는 반면 ViT는 75.7M 개의 파라미터를 갖습니다.

<br><br>

## Reference

[Nicolae-Catalin Ristea, Radu Tudor Ionescu and Fahad Shahbaz Khan. SepTr: Separable Transformer for Audio Spectrogram Processing. In Interspeech, 2022.](https://www.isca-archive.org/interspeech_2022/ristea22_interspeech.html)

[Official Source Code of SepTr](https://github.com/ristea/septr)
