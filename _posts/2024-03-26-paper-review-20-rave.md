---
layout: post
title: "[논문 리뷰] RAVE: A Variational Autoencoder for Fast and High-Quality Neural Audio Synthesis"
image: https://i.ibb.co/SwcTrq6/thumbnail.png
date: 2024-03-26
tags: 
categories: paper-review
use_math: true
---

## 논문 개요

<!-- excerpt-start -->

RAVE는 고품질(high-fidelity)의 소리를 빠른 속도로 생성하기 위한 모델입니다. 모델의 학습은 두 단계로 이루어지는데 첫 번째 단계는 VAE에 기반한 표현 학습(representation learning)이고 두 번째 단계는 오디오 품질을 향상시키기 위한 적대적(adversarial) 파인튜닝입니다. 또한 속도를 향상시키기 위해 다중 밴드 분해(multi-band decomposition)를 도입합니다.

<br><br>

## Stage 1: Representation Learning

VAE의 재구성 손실(reconstruction loss)은 다음과 같은 DDSP의 [(Jesse Engel et al., 2020)](https://openreview.net/forum?id=B1x1ma4tDr) 다중스케일 스펙트럼 거리(multiscale spectral distance)를 사용합니다.

<br>
\begin{equation}
S(\mathbf{x},\mathbf{y}) = \sum\_{n \in \mathcal{N}} \left[ \frac{\lVert \text{STFT}\_{n}(\mathbf{x}) - \text{STFT}\_{n} (\mathbf{y}) \rVert\_F}{\lVert \text{STFT}\_{n}(\mathbf{x}) \rVert\_{F}} + \log (\lVert \text{STFT}\_{n} (\mathbf{x}) - \text{STFT}\_{n} (\mathbf{y}) \rVert\_1) \right]
\end{equation}
<br>

여기서 $\small \mathcal{N}$ 은 스케일들의 집합이고 $\small \text{STFT}\_{n}$ 은 윈도우 크기 $\small n$ 과 홉 크기(hop size) $\small n/4$ 의 STFT를 적용한 스펙트로그램의 크기(magnitude)입니다. $\small \lVert \cdot \rVert\_F$ 와 $\small \lVert \cdot \rVert\_1$ 은 각각 Frobenius norm (L2 norm)과 L1 norm입니다. VAE의 인코더 $\small q\_{\phi}$ 와 디코더 $\small p(\mathbf{x} \vert \mathbf{z})$ 는 다음의 ELBO 손실로 학습합니다.

<br>
\begin{equation}
\mathcal{L}\_{\text{vae}}(\mathbf{x}) = \mathbb{E}\_{\hat{\mathbf{x}} \sim p(x \vert z)} [S(\mathbf{x}, \hat{\mathbf{x}})] + \beta \times \mathcal{D}\_{\text{KL}}[q\_{\phi}(\mathbf{z} \vert \mathbf{x}) \Vert p(\mathbf{z})]
\end{equation}
<br>

$\small \mathcal{L}\_{\text{vae}}$ 가 충분히 수렴하도록 모델을 학습시킨 뒤에 다음 학습 단계로 넘어갑니다.

<br><br>

## Stage 2: Adversarial Fine-tuning

두 번째 단계에서는 인코더를 프리즈(freeze)하고 적대적 손실을 통해 디코더만 학습시킵니다. GAN의 샘플링 분포로는 첫 번째 단계에서 학습된 잠재(latent) 분포를 사용하고 디코더가 생성자(generator)의 역할을 합니다. 판별자(discriminator)를 $\small D(\cdot)$ 라고 할 때 적대적 손실은 다음과 같습니다.

<br>
\begin{align}
\mathcal{L}\_{\text{dis}}(\mathbf{x}, \mathbf{z}) &= \max (0, 1 - D(\mathbf{x})) + \mathbb{E}\_{\hat{\mathbf{x}} \sim p(\mathbf{x} \vert \mathbf{z})} [\max (0, 1 + D(\hat{\mathbf{x}}))] \newline
\mathcal{L}\_{\text{gen}}(\mathbf{z}) &= - \mathbb{E}\_{\hat{\mathbf{x}} \sim p(\mathbf{x} \vert \mathbf{z})} [D(\hat{\mathbf{x}})]
\end{align}
<br>

생성된 신호 $\small \hat{\mathbf{x}}$ 가 실제값(ground truth) $\small \mathbf{x}$ 에서 너무 많이 벗어나지 않게 하기 위해 스펙트럼 거리를 최소화 하는 재구성 손실을 유지하면서 MelGAN의 [(Kumar et al., 2019)](http://arxiv.org/abs/ 1910.06711) 특징 매칭 손실(feature matching loss) $\small \mathcal{L}\_{FM}$ 도 추가합니다. 디코더의 전체 손실은 다음과 같습니다.

<br>
\begin{equation}
\mathcal{L}\_{\text{total}}(\mathbf{x}, \mathbf{z}) = \mathcal{L}\_{\text{gen}}(\mathbf{z}) + \mathbb{E}\_{\hat{\mathbf{x}} \sim p(\mathbf{x} \vert \mathbf{z})}[S(\mathbf{x}, \hat{\mathbf{x}}) + \mathcal{L}\_{\text{FM}}(\mathbf{x}, \hat{\mathbf{x}})]
\end{equation}
<br>

<br><br>

## Latent Representation Compactness

VAE의 잠재 표현 차원(dimension)이 높으면 많은 정보를 담을 수 있지만 사후 분포가 무시되도록 학습되는 posterior collapse가 일어날 수 있습니다. 이 논문에서는 잠재 표현 차원을 먼저 결정하지 않고 학습 이후의 분석(post-training analysis)을 통해 유용한 정보가 담긴(informative) 차원을 남기는 방법을 제안합니다.

우선 사후 분포 $\small q\_{\phi} (\mathbf{z} \vert \mathbf{x})$ 에서 샘플링한 $\small b$ 개의 샘플 $\small \mathbf{z} \in \mathbb{R}^d$ 로 구성된 행렬을 생각하는데 사후 분포의 분산을 무시하고 평균만 고려하여 다음과 같이 $\small \mathbf{Z}^{\prime} \in \mathbb{R}^{b \times d}$ 을 만듭니다.

<br>
\begin{equation}
\mathbf{Z}\_{i}^{\prime} = \arg\max\_{\mathbf{z}} q\_{\phi} (\mathbf{z} \vert \mathbf{x})
\end{equation}
<br>

만약 사후 분포 $\small q\_{\phi} (\mathbf{z} \vert \mathbf{x})$ 에서 어떤 차원이 붕괴되었다면 다양한 $\small \mathbf{x}$ 로부터 나온 $\small \mathbf{z}$ 의 값이 모두 동일하고 $\small \mathbf{Z}^{\prime}$ 에서 그 차원에 해당하는 값은 모두 동일할 것입니다. 따라서 $\small \mathbf{Z}^{\prime}$ 에서 배치 축의 평균 값을 빼주었을 때 0이 아닌 값을 갖는 차원만 잠재 공간에서 유용한 정보를 담고 있습니다.

이렇게 평균을 빼준 행렬에 SVD를 적용하여 $\small \mathbf{Z}^{\prime} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$ 를 만들면 $\small \mathbf{\Sigma}$ 에 있는 0이 아닌 값들의 개수가 랭크(rank) $\small r$ 이 되어 앞에서부터 중요한 정보를 담은 $\small r$ 개의 차원만 남길 수 있습니다. Fidelity 파라미터 $\small f \in [0, -1]$ 를 다음과 같이 정의하여 $\small f$ 가 주어졌을 때 잠재 표현의 차원을 $\small r\_f$ 개로 축소합니다.

<br>
\begin{equation}
\frac{\sum\_{i \leq r\_f} \mathbf{\Sigma}\_{ii}}{\sum\_{i} \mathbf{\Sigma}\_{ii}} \geq f
\end{equation}
<br>

<br><br>

## Multiband Decomposition

입력 오디오 파형은 48 kHz의 샘플 레이트로 만들어지고 시간 축으로 압축을 하기 위해 다중 밴드 분해(multiband decomposition)를 적용합니다. DurIAN에서와 [(Chengzhu Yu et al., 2019)](https://arxiv.org/pdf/1909.01700.pdf) 같이 Pseudo Quadrature Mirror filters (PQMF)를 사용하여 원본 파형을 각각의 주파수 대역에서 3 kHz의 낮은 샘플 레이트를 갖는 보조신호(sub-signal)들로 분해합니다. 필터 뱅크를 시간 축으로 다시 뒤집어서 적용하면 신호를 복원할 수 있습니다.

<br><br>

## Model Architecture

### Encoder

인코더의 구조는 아래 그림에 나와 있습니다. 블럭의 개수 $\small N=4$ 를 기본값으로 사용합니다.

<p align="center">
<img src="https://i.ibb.co/cCCVSpH/encoder.png" alt="encoder" border="0">
</p>

### Decoder

디코더에서는 업샘플링 층과 잔차 네트워크(residual stack)들을 거친 출력이 세 개의 네트워크로 들어갑니다. 그 구조는 아래에 요약되어 있습니다.

<p align="center">
<img src="https://i.ibb.co/Cv3myX0/decoder.png" alt="decoder" border="0">
</p>

Waveform conv는 다중 밴드 파형을 만들고 loudness conv의 출력과 곱해져서 진폭 엔벨로프(amplitude envelope)를 만듭니다. Noise synthesizer는 DDSP에서 사용하는 것과 같이 노이즈 필터로 다중 밴드 노이즈를 만들어서 신호에 더해줍니다. Upsampling layer, residual stack, 그리고 noise synthesizer의 구조는 아래 그림에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/7Vg5Dgn/detailed-architectures.png" alt="detailed-architectures" border="0">
</p>

### Discriminator

판별자의 구조는 MelGAN과 동일하게 여러 스케일의 다운샘플링 컨볼루션 층들을 사용합니다. 특징 매칭 손실을 사용하는 방식도 MelGAN과 동일합니다.

<br><br>

## 실험

실험의 데이터셋으로는 현악기 소리를 녹음한 전체 약 30시간의 Strings 데이터셋과 110명의 발화자가 녹음한 전체 44시간 짜리 음성 데이터셋인 VCTK를 [(Junichi Yamagishi et al., 2019)](https://datashare.ed.ac.uk/handle/10283/3443) 사용합니다. 베이스라인으로는 NSynth와 [(Jesse Engel et al., 2017)](https://proceedings.mlr.press/v70/engel17a.html) SING [(Alexandre Défossez et al., 2018)](https://proceedings.neurips.cc/paper/2018/hash/56dc0997d871e9177069bb472574eb29-Abstract.html) 오토인코더를 사용합니다.

실험 결과에 대한 데모 오디오는 [프로젝트 웹페이지](https://anonymous84654.github.io/RAVE_anonymous/)에서 들어볼 수 있습니다.

### Synthesis Quality

주관적 평가로 참가자들은 학습 도중에 보지 못한 Strings 데이터셋의 샘플과 세 개의 모델로 재구성한 소리를 듣고 1부터 5 사이의 스케일로 오디오 품질을 평가합니다. 그 결과는 아래 표에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/Pm5ML14/subjective-results.png" alt="subjective-results" border="0">
</p>

RAVE가 다른 두 베이스라인보다 적은 파라미터 수를 가지고 있으면서도 우세한 성능을 나타냅니다. 하지만 실제값(ground truth)과 비교하면 여전히 큰 차이가 존재합니다.

### Synthesis Speed

CPU와 GPU에서의 오디오 생성 속도도 측정합니다. 그 결과는 아래 표에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/Y8K3x6J/speed-results.png" alt="speed-results" border="0">
</p>

자기회귀적(autoregressive) 방식의 NSynth보다 다른 두 모델의 속도가 훨씬 빠르고 다중 밴드 분할이 추가되면 RAVE의 속도가 가장 빠릅니다. 오디오 샘플 레이트가 48 kHz이기 때문에 이 설정에서 RAVE는 실시간(realtime)보다 CPU에서는 20배, GPU에서는 약 240배 빠른 것입니다.

<br><br>

### Balancing Compactness and Fidelity

적은 차원으로 표현된 잠재 표현은 분석과 조작이 용이하다는 장점이 있지만 재구성 품질 측면에서는 불리합니다. RAVE는 잠재 표현 차원을 학습 전에 결정하지 않고 학습 이후에 fidelity 파라미터 $\small f$ 를 이용하여 정보가 없는 차원을 잘라내기 때문에 $\small f$ 를 바꿔가며 영향을 분석할 수 있습니다. 이 실험 결과는 아래 그림에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/34DF5Gt/fidelity-graphs.png" alt="fidelity-graphs" border="0">
</p>

$\small f=0.99$ 로 설정했을 때 strings 데이터셋의 잠재 표현 차원은 128에서 24로, vctk 데이터셋에서는 16으로 줄어듭니다. 오른쪽 그래프를 보면 $\small f$ 를 줄일수록 오디오 재구성 품질이 떨어져서 스펙트럼 거리가 늘어나는 것을 알 수 있습니다. 아래의 그림은 $\small f$ 의 크기에 따라 vctk 데이터셋에서 재구성된 샘플의 멜 스펙트로그램을 나타낸 것입니다.

<p align="center">
<img src="https://i.ibb.co/WpHz53G/fidelity-values.png" alt="fidelity-values" border="0">
</p>

$\small f$ 값이 작아질수록 재구성된 샘플의 정확도가 떨어지고 음소나 발화자의 특징 등의 정보를 잃는 것이 나타납니다.

<br><br>

### Timbre Transfer
RAVE 모델을 사용하여 학습 데이터셋의 분포와 다른 샘플을 재구성하도록 하는 방식으로 음색 전이(timbre transfer)를 수행할 수 있습니다. 예를 들어 아래 그림은 strings와 vctk 데이터셋 간에 서로 전이를 한 결과를 스펙트로그램으로 나타낸 것입니다.

<p align="center">
<img src="https://i.ibb.co/R6ZbG2v/transfer-results.png" alt="transfer-results" border="0">
</p>

전체적인 음량(loudness)이나 기본 주파수(fundamental frequency)와 같은 요소들은 그대로 남아 있고 음소와 같은 요소들은 strings 데이터셋에는 없다가 vctk로 전이되어 재구성된 샘플에는 추가되는 결과가 나타납니다.

<br><br>

### Signal Compression

RAVE로 학습된 표현이 원래의 파형보다 훨씬 경제적이기 때문에 오디오 압축을 위한 용도로도 사용할 수 있습니다. RAVE의 압축 비율은 2048로 23 Hz에 해당하는 잠재 신호를 만듭니다. 이에 대한 실험으로 WaveNet 기반의 자기회귀적 모델을 RAVE의 디코더와 결합하였을 때 실시간보다 5배 빠른 생성 속도를 나타냅니다.

<br><br>

## Reference

[Antoine Caillon and Philippe Esling. RAVE: A Variational Autoencoder for Fast and High-Quality Neural Audio Synthesis. arXiv preprint, 2021.](http://arxiv.org/abs/2111.05011)