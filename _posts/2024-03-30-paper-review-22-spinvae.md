---
layout: post
title: "[논문 리뷰] Synthesizer Preset Interpolation Using Transformer Auto-Encoders"
image: https://i.ibb.co/7kd0Ztc/thumbnail.png
date: 2024-03-30
tags: 
categories: paper-review
use_math: true
---

## 논문 개요

<!-- excerpt-start -->

이 논문은 Frequency Modulation (FM) 기반의 신디사이저에서 두 프리셋(preset)이 주어졌을 때 보간된(interpolated) 소리를 생성하는 SPINVAE 모델을 제안합니다. 프리셋의 보간을 위해 VAE를 사용하여 프리셋의 합성 파라미터(synthesis parameter)들을 잠재 공간(latent space)으로 임베딩한 뒤 잠재 벡터의 선형 보간을 적용하는 방법을 취합니다. 이러한 방법은 단순히 각각의 파라미터들을 독립적으로 선형보간하는 것에 비해 두 타겟 소리에서 동떨어지지 않고 점진적으로 변화하는 결과를 보여줍니다.

<br><br>

## Synthesizer and Datasets

신디사이저로는 Yamaha DX7을 기반으로 한 오픈소스 FM 신디사이저인 Dexed를 사용합니다. 이에 대한 프리셋으로는 이전 연구에서 [(Gwendal Le Vaillant et al., 2021)](https://ieeexplore.ieee.org/abstract/document/9768218) 수집하여 공개한 3만 개의 프리셋을 사용합니다. 각각의 프리셋은 144개의 합성 파라미터를 포함합니다.

<br><br>

## Model

SPINVAE의 목적은 프리셋의 파라미터들을 잠재 벡터로 임베딩하여 잠재 공간에서의 보간을 수행할 수 있도록 하는 것입니다. 이러한 태스크에서 중요한 점은 잠재 벡터 $\small \mathbf{z}$ 가 의미 있는 표현을 잘 학습하여 잠재 공간에서 서로 비슷한 $\small \mathbf{z}^{(n)}$ 와 $\small \mathbf{z}^{(m)}$ 이 디코딩 된 파라미터 $\small \mathbf{u}^{(n)}$ 과 $\small \mathbf{u}^{(m)}$ 이 서로 비슷한 소리를 만들어내야 된다는 것입니다. 이를 위해 VAE의 입력으로 프리셋 $\small \mathbf{u}$ 뿐만 아니라 스펙트로그램 $\small \mathbf{x}$ 도 같이 넣어줍니다. VAE 손실은 아래 식과 같습니다.

<br>
\begin{equation}
\mathcal{L}(\mathbf{x}, \mathbf{u}) = \beta D\_{KL} [q(\mathbf{z} \vert \mathbf{x}, \mathbf{u}) \Vert p(\mathbf{z})] - \mathbb{E}\_{q(\mathbf{z} \vert \mathbf{x}, \mathbf{u})} [\log p(\mathbf{x}, \mathbf{u} \vert \mathbf{z})]
\end{equation}
<br>

디코딩되는 $\small p(\mathbf{x} \vert \mathbf{z})$ 와 $\small p(\mathbf{u} \vert \mathbf{z})$ 는 서로 독립적인 분포로 모델링됩니다. 오디오 디코더 $\small p(\mathbf{x} \vert \mathbf{z})$ 는 CNN과 잔차 연결(residual connection)을 기반으로 스펙트로그램 픽셀을 가우시안 분포로 모델링합니다. 프리셋 디코더 $\small p(\mathbf{u} \vert \mathbf{z})$ 는 각각의 파라미터 분포를 모델링하는데 파라미터 종류에 따라 적합한 분포를 사용합니다.

예를 들어 FM 신디사이저에는 오실레이터들 사이의 경로를 제어하는 알고리즘 파라미터가 있습니다. 이러한 알고리즘이나 파형 종류 등은 카테고리형이므로 마지막에 소프트맥스 함수를 사용합니다.

반면 주파수나 어택 시간(attack time) 등은 이산적인(discrete) 값이기 때문에 discretized logistic mixture (DLM) 분포의 [(Tim Salimans et al., 2017)](https://openreview.net/forum?id=BJrFC6ceg) 로그 우도(log-likelihood)를 최대화하도록 학습합니다. DLM은 여러 개의 로지스틱 분포의 혼합으로 이산적인 값들에 대한 확률 분포를 모델링하는 것인데 SPINVAE에서는 3개의 로지스틱 분포를 사용합니다.

프리셋 파라미터들은 시퀀스의 형태로 트랜스포머를 통해 인코딩되고 디코딩됩니다. 학습 가능한 입력 토큰 $\small \mathbf{e}\_{\mu}$ 와 $\small \mathbf{e}\_{\sigma}$ 가 임베딩된 프리셋 시퀀스의 앞에 연결(concatenate)되어 트랜스포머 인코더로 들어갑니다. 트랜스포머 디코더에서는 $\small \mathbf{z}$ 가 키와 밸류로 사용되고 학습된 프리셋 임베딩이 쿼리로 사용됩니다. 전체 모델 구조는 아래 그림에 표현되어 있습니다.

<p align="center">
<img src="https://i.ibb.co/PN0GZnq/model-architecture.png" alt="model-architecture" border="0">
</p>

<br><br>

## Audio Interpolation Metrics

학습이 끝나면 1500 쌍의 샘플들에 대한 보간을 진행합니다. 먼저 임의로 섞인 3000개의 테스트 셋에서 두 샘플 $\small (\mathbf{x}^{(n)}, \mathbf{u}^{(n)})$ 과 $\small (\mathbf{x}^{(m)}, \mathbf{u}^{(m)})$ 을 $\small \mathbf{z}^{(n)} = \mathbf{\mu}^{(n)}$ 과 $\small \mathbf{z}^{(m)} = \mathbf{\mu}^{(m)}$ 으로 인코딩합니다. 그 뒤 잠재 공간에서 선형 보간을 하여 $\small \\{ \mathbf{z}\_t, t \in [1, T] \\}$ 를 얻습니다. 각각의 $\small \mathbf{z}\_t$ 는 $\small \mathbf{u}\_t$ 프리셋으로 디코딩되고 각각의 시퀀스는 $\small T=9$ 개의 스텝을 포함합니다.

프리셋 보간의 품질을 정량적으로 평가하기 위해 먼저 Timbre Toolbox를 [(Geoffroy Peeters et al., 2011)](https://pubs.aip.org/asa/jasa/article/130/5/2902/842365/The-Timbre-Toolbox-Extracting-audio-descriptors) 사용해서 전문가들에 의해 디자인된 오디오 특징(feature)들을 추출합니다. 이 특징들에는 어택 시간(attack time), 디케이 기울기(decay slope), 스펙트럼 센트로이드(spectral centroid), 부조화(inharmonicity) 등이 포함됩니다.

프리셋 보간 시퀀스에서 각각의 오디오 특징에 대한 부드러움(smoothness)과 비선형성(non-linearity)을 계산하여 보간 품질에 대한 지표로 사용합니다. 부드러움은 시퀀스 내에서 특징의 이차 미분 값들의 RMS 값으로 정의됩니다. 비선형성은 특징에 대한 이상적인 선형 보간과 모델에서 나온 보간 값 사이의 RMS 거리로 측정됩니다. 따라서 두 지표 모두 값이 낮을수록 보간 품질이 높다는 것을 의미합니다.

<br><br>

## 실험

가장 일반적인 프리셋 보간 방법은 모든 파라미터들을 독립적으로 선형 보간하는 것입니다. 따라서 이 방법을 실험의 레퍼런스로 사용합니다. 실험 결과에 대한 오디오 데모 샘플은 [프로젝트 웹페이지](https://gwendal-lv.github.io/spinvae/)에서 들어볼 수 있습니다.

### Results

아래 그림은 46개의 모든 오디오 특징들에 대해 계산된 보간의 부드러움 값을 나타낸 것입니다.

<p align="center">
<img src="https://i.ibb.co/VCC8Rpk/smoothness.png" alt="smoothness" border="0">
</p>

대부분의 특징들에 대해 SPINVAE가 레퍼런스보다 낮은 값을 나타내고 $\small \text{p-value} < 0.05$ 인 유의미한 차이를 기준으로 하면 35개의 특징에 대해 SPINVAE가 우세합니다. 또한 평균적으로 SPINVAE의 부드러움이 레퍼런스보다 12.6% 감소된 값을 나타냅니다. 이 결과와 비선형성에 대한 결과는 아래 표에 정리되어 있습니다.

<p align="center">
<img src="https://i.ibb.co/JkjZTqC/results-table.png" alt="results-table" border="0">
</p>

아래 그림은 SPINVAE와 레퍼런스의 보간 결과에 대한 스펙트로그램 예시입니다. 정량적 평가 지표에서 나타나듯이 SPINVAE가 더 부드럽고 자연스럽게 보간되는 결과를 보여줍니다.

<p align="center">
<img src="https://i.ibb.co/Gv8RwKq/spectrograms.png" alt="spectrograms" border="0">
</p>

### Ablation Study

위의 표에는 제거 연구(ablation study) 결과들도 포함되어 있습니다. 먼저 표의 첫 번째 섹션에 있는 Preset-only는 스펙트로그램 $\small \mathbf{x}$ 을 사용하지 않은 것이고 Sound match는 프리셋 인코더를 사용하지 않고 $\small q(\mathbf{z} \vert \mathbf{x})$ 를 인코더로 사용한 것입니다. 두 버전의 모델 모두 SPINVAE에 비해 성능이 떨어지는 것으로 스펙트로그램과 프리셋을 같이 인코딩 하는 것의 중요성을 알 수 있습니다.

표의 두 번째 섹션에 있는 모델들은 실수 값을 갖는 프리셋 파라미터들에 대해 서로 다른 분포를 적용한 것입니다. DLM 2와 DLM 4는 SPINVAE에서 3개 사용하는 DLM 분포를 각각 2개와 4개로 변경한 것이고 Softmax는 이산적인 파라미터 값들을 카테고리로 취급하여 학습한 것입니다. 모두 레퍼런스에 비해서는 좋은 보간 품질을 나타내지만 SPINVAE에 비해서는 성능이 떨어집니다.

표의 마지막 섹션에 있는 모델들은 프리셋 인코더와 디코더로 트랜스포머 대신 각각 MLP와 LSTM을 사용한 것입니다. 이 경우 성능이 많이 떨어지는 것을 볼 수 있습니다.

<br><br>

## Reference

[Gwendal Le Vaillant and Thierry Dutoit. Synthesizer Preset Interpolation Using Transformer Auto-Encoders. In ICASSP, 2023.](https://ieeexplore.ieee.org/document/10096397/)

[Official Source Code of SPINVAE](https://github.com/gwendal-lv/spinvae)