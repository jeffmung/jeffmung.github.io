---
layout: post
title: "Improving Synthesizer Programming from Variational Autoencoders Latent Space [DAFx, 2021]"
description: 이 논문은 디지털 신디사이저를 사용하여 원하는 소리를 입력으로 받았을 때 그 소리를 낼 수 있는 신디사이저 파라미터를 추정하는 신경망 모델을 제안합니다.
image: https://i.ibb.co/Jy0QGvy/thumbnail.png
date: 2024-04-16
tags: 
categories: paper-review
use_math: true
---

## 논문 개요

<!-- excerpt-start -->

이 논문은 디지털 신디사이저를 사용하여 원하는 소리를 입력으로 받았을 때 그 소리를 낼 수 있는 신디사이저 파라미터를 추정하는 신경망 모델을 제안합니다. 모델의 구조는 주로 VAE와 Normalizing Flows에 기반하고 있으며 오픈소스 FM 신디사이저인 Dexed를 기준으로 개발되었습니다. 같은 주제의 다른 연구들과 차별화 되는 가장 큰 특징은 미디 노트의 음높이와 속도에 영향을 받는 신디사이저 파라미터들을 고려하여 여러 개의 미디 노트에 대한 소리를 학습에 사용한다는 점입니다.

<br><br>

## General Architecture

모델은 먼저 스펙트로그램 $\small \mathbf{x}$ 를 입력으로 받아 CNN 기반의 VAE 인코더로 잠재 분포(latent distribution) $\small \mathbf{z}_0$ 를 만들고 Flow 기반 모델을 통해 같은 차원의 잠재 벡터 $\small $\mathbf{z}\_K$ 로 변환합니다. 파라미터 $\small \mathbf{v}$ 를 예측하는 모델은 역시 Flow 기반으로 변환 $\small \mathbf{v} = U(\mathbf{z}\_K)$ 를 수행합니다. 또한 VAE 디코더 $\small p\_{\theta} (\mathbf{z}\_0 \vert \mathbf{x})$ 도 학습에 사용됩니다. 전체 모델 구조는 아래 그림에 요약되어 있습니다.

<p align="center">
<img src="https://i.ibb.co/c3gxdfP/architecture.png" alt="architecture" border="0">
</p>

<br><br>

## Latent Space Normalizing Flows

VAE의 사후 분포(posterior distribution)를 모델링하는 $\small q_{\phi} (\mathbf{z} \vert \mathbf{x})$ 와 사전 분포 $\small p_{\theta} (\mathbf{z})$, 디코더 $\small p_{\theta} (\mathbf{x} \vert \mathbf{z}) 에 대해서 ELBO에 의한 학습의 손실 함수는 다음과 같습니다.

<br>
\begin{equation}
\mathcal{L}\_{\theta, \phi}(\mathbf{x}) = - \mathbb{E}\_{q\_{\phi}(\mathbf{z} \vert \mathbf{x})}[\log p\_{\theta}(\mathbf{x} \vert \mathbf{z})] + \mathbb{E}\_{q\_{\phi}(\mathbf{z} \vert \mathbf{x})}[\log q\_{\phi}(\mathbf{z} \vert \mathbf{x}) - \log p\_{\theta}(\mathbf{z})]
\end{equation}
<br>

Normalizing Flows는 간단한 분포 (e.g. 가우시안 분포) $\small \mathbf{z}_0$ 와 추정하고 싶은 복잡한 분포 $\small \mathbf{z}_K$ 사이에서 일련의 가역적인 변환 $\small T\_1, \ldots, T\_K$ 을 모델링합니다. 이때 $\small T\_{k+1}(\mathbf{z}\_k) = \mathbf{z}\_{k+1}$ 입니다. $\small T_k$ 의 Jacobian matrix $\small J_k$ 를 활용하여 Flow 변환의 관계를 다음 식과 같이 나타낼 수 있습니다.

<br>
\begin{equation}
\log p(\mathbf{z}\_K) = \log p(\mathbf{z}\_0) - \sum\_{k=1}^K \log \vert \det J\_{T\_k} (\mathbf{z}\_{k-1}) \vert
\end{equation}
<br>

이 논문의 모델에서는 VAE의 CNN 인코더 출력 $\small \mathbf{z}_0$ 와 디코더 입력 $\small \mathbf{z}_K$ 사이에 Flow 모델로 결정론적인(deterministic) 변환이 이루어지므로 위의 VAE 손실을 Normalizing Flows의 식과 결합하여 다음과 같이 표현할 수 있습니다.

<br>
\begin{equation}
\mathcal{L}\_{\theta, \phi, \psi}(\mathbf{x}) = \mathbb{E}\_{q\_{\phi}(\mathbf{z\_0} \vert \mathbf{x})} \left[ - \log p\_{\theta}(\mathbf{x} \vert \mathbf{z}\_K) + \log q\_{\phi} (\mathbf{z}\_0 \vert \mathbf{x}) - \sum\_{k=1}^K \log \vert \det J\_{T\_{k}} (\mathbf{z}\_{k-1}) \vert - \log p\_{\theta} (\mathbf{z\_K})
\end{equation}
<br>

<br><br>

## Synthesizer Parameters Regression Flow

Normalizing Flows 모델에도 여러 가지 종류가 있는데 이 논문에서는 기본적으로 RealNVP [(Laurent Dinh et al., 2017)](https://arxiv.org/abs/1605.08803) 모델을 사용합니다. RealNVP는 크기 변환(scaling)과 이동(shift)으로 이루어진 affine coupling layer를 변환 함수로 하여 딥러닝 신경망이 크기 변환과 이동 파라미터를 출력하도록 모델링합니다.

원래의 RealNVP는 위의 VAE 손실 함수에서처럼 Jacobian matrix의 행렬식(determinant)가 역전파(backpropagation)에 사용되어야 하는데 신디사이저 파라미터 $\small \mathbf{v}$ 를 추정하는 모델에서는 학습이 성공적으로 되지 않았다고 합니다. 하지만 RealNVP 모델은 그대로 사용한 채로 손실 함수는 단순히 $\small \log p\_{\lambda} (\mathbf{v} \vert \mathbf{z}\_K)$ 만 최대화하도록 했을 때에는 학습이 성공적으로 이루어졌습니다. 따라서 신디사이저 파라미터 추정에는 Flow 모델이 일반적인 회귀 모델의 피드포워드 신경망(feedforward network)처럼 사용되고 Jacobian 행렬식은 학습에 사용되지 않습니다.

<br><br>

## Synthesizer Parameters Vector

신디사이저 파라미터들은 종류에 따라 수치형(numerical)이거나 (e.g. 오실레이터 진폭, 컷오프 주파수, ...) 카테고리형(categorical) (e.g. 파형, 필터 종류, ...) 모두 존재합니다. 이 논문에서는 세 가지 형태로 파라미터들을 다룹니다. 첫 번째로 $\small \text{Num only}$는 모든 파라미터들을 수치형으로 다룹니다.

$\small \text{NumCat}$ 은 수치형 파라미터들은 수치형으로 다루고 카테고리형 파라미터들은 원핫 인코딩합니다. 아래 그림에서 왼쪽은 $\small \text{Num only}$, 오른쪽은 $\small \text{NumCat}$ 을 나타냅니다.

<p align="center">
<img src="https://i.ibb.co/6Bb7V47/parameter-types.png" alt="parameter-types" border="0">
</p>

파라미터 39는 원래 카테고리형인데 왼쪽 그림에서는 그냥 수치형으로 다루는 것을 볼 수 있습니다. 마지막으로 $\small \text{NumCat++}$ 는 32개 이하의 이산적인(discrete) 수치형 파라미터들도 원핫 인코딩 하는 방법입니다.

<br><br>

## Dataset

연구에 사용한 신디사이저는 Yamaha DX7을 모방한 오픈소스 FM 신디사이저인 Dexed 신디사이저입니다. 데이터셋으로는 여러 소스에서 수집하고 중복되는 것을 제거한 약 3만 개의 프리셋을 사용합니다.

Dexed에는 조작할 수 있는 155개의 파라미터가 있지만 10개는 모든 데이터셋에 대해서 고정되어 있습니다. 따라서 학습 가능한 파라미터는 144개입니다.

모든 프리셋에 대해서 오디오는 각각의 MIDI 노트로 4초 동안 연주된 소리입니다. 3초 동안은 노트가 눌러져 있고 마지막 1초 동안은 릴리즈되는 소리에 해당합니다. 멜스펙트로그램은 크기 1024의 Hann window, 홉 크기(hop size) 256, Mel 주파수 구간(bin) 257개로 STFT를 적용하여 만듭니다. 멜스펙트로그램에는 -120 dB의 임계값(threshold)를 적용하고 전체 데이터셋의 진폭 최소값과 최대값을 사용하여 [-1, 1] 범위로 정규화합니다.

<br><br>

## Loss Functions

모델 학습의 손실 함수는 VAE의 재구성 손실(reconstruction loss)와 정칙화 손실(regularization loss), 그리고 파라미터 손실(parameter loss)로 구성됩니다. 내구성 손실은 디코더 출력 $\small \hat{\mathbf{x}}$ 에 대한 MSE로 계산됩니다. 재구성 손실에는 위의 Latent Space Normalizing Flows 섹션에 있는 식에서 볼 수 있듯이 로그 확률과 Jacobian의 행렬식이 포함됩니다.

파라미터 손실은 수치형 파라미터에 대해서는 MSE를, 카테고리형 손실에 대해서는 크로스 엔트로피를 계산합니다.

<br><br>

## Multiple Input Notes

많은 신디사이저에는 MIDI 노트를 연주하는 음높이(pitch)와 세기(intensity)에 따라 소리를 다르게 변화시키는 파라미터들이 존재합니다. 예를 들어 컷오프 주파수는 노트 음높이에 영향을 받고 오실레이터 진폭은 노트의 속도(velocity)와 관련이 있습니다. 이러한 파라미터들을 학습하기 위해 여러 개의 MIDI 노트들에 대한 스펙트로그램이 사용되어야 합니다.

따라서 이 논문에서는 같은 프리셋에 대해 여러 개의 스펙트로그램을 만들어서 모델에 입력하는 방법을 제안합니다. 그 구조는 아래 그림에 표현되어 있습니다.

<p align="center">
<img src="https://i.ibb.co/jLTtqjX/multi-note-channels.png" alt="multi-note-channels" border="0">
</p>

먼저 MIDI 값 $\small (\text{pitch, intensity})$에 대해 $\small (40, 85), (50, 85), (60, 42), (60, 85), (60, 127), (70, 85)$ 의 여섯 개 쌍을 사용하여 VAE 입력이 6개 채널의 스펙트로그램이 되도록 합니다. 각각의 채널에 대한 출력은 스태킹된 뒤 컨볼루션 층에 의해 합쳐집니다.

<br><br>

## 실험

실험 결과에 대한 데모 사운드 샘플은 [프로젝트 웹사이트](https://gwendal-lv.github.io/preset-gen-vae/)에서 들어볼 수 있습니다. 첫 번째로 (60, 85)의 $\small (\text{pitch, intensity})$ 를 갖는 MIDI 노트만 사용했을 때의 결과가 아래 표에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/hMrsRXP/result-table.png" alt="result-table" border="0">
</p>

파라미터의 형태 분류에 따른 결과를 보면 $\small \text{Num only}$ 가 가장 안좋고 $\small \text{NumCat}++$ 가 가장 좋은 경향이 있습니다. 또한 신디사이저 파라미터를 예측하는 네트워크로 Flow 대신 MLP를 사용하면 성능이 약간 하락합니다.

6개의 MIDI 노트를 사용한 다중 채널 모델의 실험 결과는 아래 표에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/6WPh4gj/multi-channel-result.png" alt="multi-channel-result" border="0">
</p>

1개의 MIDI 노트만 사용했을 때보다 성능이 약간 증가하는 것을 알 수 있습니다.

## Reference

[Gwendal Le Vaillant, Thierry Dutoit and Sébastien Dekeyser. Improving Synthesizer Programming from Variational Autoencoders Latent Space. In DAFx, 2021.](https://ieeexplore.ieee.org/document/9768218/)

[Official Source Code of Preset-Gen-VAE](https://github.com/gwendal-lv/preset-gen-vae)