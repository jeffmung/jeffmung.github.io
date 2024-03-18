---
layout: post
title: "[논문 리뷰] Jukebox: A Generative Model for Music"
image: https://i.ibb.co/X43MyCZ/thumbnail.png
date: 2024-03-18
tags: 
categories: paper-review
use_math: true
---

## 논문 개요

<!-- excerpt-start -->

Jukebox는 직접적인 파형(direct waveform)의 형태로 긴 시간 동안 지속되는 음악을 생성할 수 있는 모델입니다. 음악 생성에는 장르, 아티스트, 가사 등을 조건으로 넣어줄 수 있습니다.

Jukebox의 음악 생성 방법은 기본적으로 계층적인 VQ-VAE로 여러 압축률의 토큰을 학습시키고 Sparse Transformer로 자기회귀적인(autoregressive) 생성을 모델링하는 것으로 구성되어 있습니다. 자기회귀적인 생성 역시 계층적인 구조로 업샘플링을 통해 점점 세부적인 정보를 반영할 수 있도록 이루어집니다.

<br><br>

## VQ-VAE

오디오 신호는 일반적으로 16 kHz에서 48 kHz까지 높은 샘플 레이트로 만들어지고 이렇게 긴 시퀀스를 처리하는 것은 매우 많은 계산량을 요구하게 됩니다. 따라서 오디오 신호를 낮은 차원의 공간으로 압축하는 방법이 주로 사용되는데 Jukebox는 여기에 VQ-VAE를 [(Aaron van den Oord et al., 2017)](https://proceedings.neurips.cc/paper/2017/hash/7a98af17e63a0ac09ce2e96d03992fbc-Abstract.html) 이용합니다.

입력의 형태는 연속적인 파형 $\small \mathbf{x} \in [-1, 1]^T$ 입니다. 1차원의 VQ-VAE는 입력 시퀀스 $\small \mathbf{x} = \langle \mathbf{x}\_t \rangle\_{t=1}^T$ 를 이산적인 토큰 시퀀스 $\small \mathbf{z} = \langle z\_s \in [K] \rangle\_{s=1}^S$ 로 인코딩합니다. 여기서 $\small K$ 는 코드북의 어휘(vocabulary) 크기이고 $\small T/S$ 는 홉 길이(hop length)가 됩니다.

VQ-VAE의 인코더 $\small E(\mathbf{x})$ 는 $\small \mathbf{x}$ 를 잠재(latent) 벡터 시퀀스 $\small \mathbf{h} = \langle \mathbf{h}\_s \rangle\_{s=1}^S$ 로 인코딩합니다. 그리고 코드북 $\small C = \\{ \mathbf{e}\_k \\}\_{k=1}^K$ 가 존재하여 각각의 $\small \mathbf{h}\_s$ 는 가장 가까운 코드북 벡터 $\small \mathbf{e}\_{z\_s}$ 로 매핑됩니다. 디코더 $\small D(\mathbf{e})$ 는 다시 양자화된 벡터를 입력 공간으로 디코딩합니다. VQ-VAE의 학습 손실 함수는 다음과 같습니다.

<br>
\begin{align}
\mathcal{L} &= \mathcal{L}\_{\text{recons}} + \mathcal{L}\_{\text{codebook}} + \beta \mathcal{L}\_{\text{commit}} \newline
\newline
\mathcal{L}\_{\text{recons}} &= \frac{1}{T} \sum\_t \lVert \mathbf{x}\_t - D(\mathbf{e}\_{z\_t}) \rVert\_2^2 \newline
\newline
\mathcal{L}\_{\text{codebook}} &= \frac{1}{S} \sum\_s \lVert \text{sg} [\mathbf{h}\_s] - \mathbf{e}\_{z\_s} \rVert\_2^2 \newline
\newline
\mathcal{L}\_{\text{commit}} &= \frac{1}{S} \sum\_s \lVert \mathbf{h}\_s - \text{sg} [\mathbf{e}\_{z\_s}] \rVert\_2^2
\end{align}
<br>

이떄 $\small \text{sg}$ 는 스탑-그래디언트 오퍼레이션을 나타냅니다. 또한 학습을 효율적으로 하기 위해 코드북 손실 $\small \mathcal{L}_{\text{codebook}}$ 에는 EMA 업데이트를 사용합니다.


VQ-VAE 2에서는 [(Ali Razavi et al., 2019)](https://arxiv.org/abs/1906.00446) 잠재 시퀀스 $\small \mathbf{h}$ 를 다중 레벨의 $\small [\mathbf{h}^{(1)}, \cdots, \mathbf{h}^{(L)}]$ 로 나눠서 각각의 코드북 $\small C^{(l)}$ 을 학습하게 합니다. 이때 잠재 시퀀스의 길이가 상위 레벨로 갈수록 줄어드는 계층적인 구조입니다. 디코더는 하나를 사용하고 모든 레벨의 양자화된 잠재 시퀀스를 다같이 받아서 입력 공간으로 디코딩하도록 학습됩니다.

<br><br>

## Music VQ-VAE

Jukebox의 Music VQ-VAE는 VQ-VAE 2의 계층적인 구조를 따라 위, 중간, 아래의 세 레벨을 사용하지만 세부적인 몇 가지 다른 점들이 있습니다.

### Separated Autoencoders

VQ-VAE 2와 같이 하나의 디코더를 사용하여 모든 레벨을 같이 학습시키면 위 레벨이 거의 사용되지 않고 코드북이 완전히 붕괴(collapse)되어서 모델이 모든 정보를 덜 압축된 아래 레벨에서 처리하도록 학습되는 문제가 관찰되었습니다. 이러한 문제를 해결하기 위해 Jukebox에서는 각 레벨의 오토인코더를 분리하여 학습시킵니다. 따라서 각 레벨의 이산적인(discrete) 코드들도 서로 독립적입니다. 그 구조는 아래 그림에 표현되어 있습니다.

<p align="center">
<img src="https://i.ibb.co/0sXCdd7/music-vqvae.png" alt="music-vqvae" border="0">
</p>

각 레벨의 인코더는 다운샘플링 컨볼루션과 확장된 컨볼루션(dilated convolution)을 포함하고 있는 블럭들로 이루어져 있습니다. 이 인코더 블럭의 구조는 아래 그림에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/DzdByv0/encoder-block.png" alt="encoder-block" border="0">
</p>

인코더 블럭의 첫 컨볼루션은 다운샘플링을 위한 것으로 커널 크기 4와 스트라이드 2를 갖습니다. 따라서 각 인코더 블럭에서는 시간 축으로 2배씩 압축되고 홉 길이를 더 늘리기 위해서는 단순히 블럭을 더 쌓습니다.

다운샘플링 컨볼루션 다음에 오는 잔차 네트워크(residual network)는 확장된 컨볼루션과 $\small 1 \times 1$ 컨볼루션으로 이루어져 있습니다. 확장된 컨볼루션의 확장 인자는 3으로 설정되어 수용 영역(receptive field)을 증가시킵니다. 각각의 인코더 블럭 안에 $\small D$ 개의 잔차 블럭으로 된 잔차 네트워크가 있고 이를 포함한 전체 인코더 블럭이 $\small L$ 개 있는 구조입니다.

디코더는 인코더를 반대로 뒤집어놓은 구조입니다. $\small D$ 개의 확장된 컨볼루션 블럭으로 이루어진 잔차 네트워크와 업샘플링을 위한 전치 컨볼루션(transposed convolution)이 하나의 디코더 블럭을 구성하고 이것이 $\small L$ 개 모여 디코더가 됩니다. 마지막의 컨볼루션 층은 출력의 채널 차원을 원래의 입력 오디오 채널 개수와 같아지도록 맞춰줍니다. 아래 그림은 디코더 블럭의 구조입니다.

<p align="center">
<img src="https://i.ibb.co/jvyctxr/decoder-block.png" alt="decoder-block" border="0">
</p>

인코더와 디코더 블럭의 개수는 위, 중간, 아래 레벨에서 각각 3, 5, 7개입니다. 위 레벨과 중간 레벨의 잔차 네트워크는 동일하게 4개씩의 블럭을 갖고 이에 해당하는 수용 영역의 크기는 각각 토큰 당 480 ms와 120 ms입니다. 아래 레벨의 잔차 네트워크는 8개의 블럭을 갖습니다.

### Random Restarts for Embeddings

VQ-VAE의 유명한 문제는 코드북의 몇 개 임베딩 벡터만 사용되고 나머지는 사용되지 않는 코드북 붕괴(codebook collapse)입니다. 이를 예방하기 위하여 랜덤 재시작(random restart)을 사용합니다. 코드북 벡터의 평균 사용량이 기준값 이하로 떨어지면 그 벡터를 현재 배치에 있는 임의의 인코더 출력 하나로 재설정 해주는 방법입니다.

### Spectral Loss

샘플 레벨의 재구성 손실(reconstruction loss)만 사용하면 모델이 낮은 주파수만을 재구성하도록 학습되는 경향이 있습니다. 따라서 다음과 같은 스펙트럼 손실(spectral loss)를 추가합니다.

<br>
\begin{equation}
\mathcal{L}\_{\text{spec}} = \lVert \vert \text{STFT}(\mathbf{x}) \vert - \vert \text{STFT}(\hat{\mathbf{x}}) \rVert\_2
\end{equation}
<br>

특정한 STFT 파라미터 선택에 모델이 오버피팅되지 않도록 시간과 주파수 해상도에 대한 여러 STFT 파라미터들에 대해 계산된 $\small \mathcal{L}_{\text{spec}}$ 의 합을 사용합니다.

<br><br>

## Music Priors and Upsamplers

VQ-VAE를 학습한 뒤에는 생성에 사용할 압축된 공간에서의 사전분포(prior) $\small p(\mathbf{z})$ 를 학습해야 합니다. 사전분포는 아래 식과 같이 분해되어 위 레벨 사전분포 $\small p(\mathbf{z}^{\text{top}})$ 과 업샘플러 $\small p(\mathbf{z}^{\text{middle}} \vert \mathbf{z}^{\text{top}})$, 그리고 $\small p(\mathbf{z}^{\text{bottom}} \vert \mathbf{z}^{\text{middle}}, \mathbf{z}^{\text{top}})$ 가 분리된 자기회귀적 모델링 문제로 학습됩니다.

<br>
\begin{align}
p(\mathbf{z}) &= p(\mathbf{z}^{\text{top}}, \mathbf{z}^{\text{middle}}, \mathbf{z}^{\text{bottom}}) \newline
&= p(\mathbf{z}^{\text{top}}) p(\mathbf{z}^{\text{middle}} \vert \mathbf{z}^{\text{top}}) p(\mathbf{z}^{\text{bottom}} \vert \mathbf{z}^{\text{middle}}, \mathbf{z}^{\text{top}})
\end{align}
<br>

각 레벨에서 자기회귀적 모델은 다음 토큰을 예측하도록 학습되는데 중간과 아래 레벨에서는 상위 레벨에서 나온 토큰들을 조건부로 입력 받습니다. 같은 청크(chunk)에 해당하는 상위 레벨 토큰만 조건으로 사용하고 각 레벨에서의 시간적 해상도가 다르기 때문에 상위 레벨의 토큰들은 업샘플링됩니다.

### Artist, Genre, and Timing Conditioning

모델을 학습시킬 때 장르와 아티스트에 대한 정보를 조건부로 같이 넣어주고 생성 시에 생성되는 음악의 스타일을 제어할 수 있습니다. 각각의 장르와 아티스트 레이블은 임베딩 벡터로 학습되고 둘이 합해져서 시퀀스의 맨 처음 토큰으로 붙습니다.

또한 전체 곡의 길이와 각각의 토큰이 곡 안에서 어느 시간에 위치하고 있는지 알려주는 시간 정보도 임베딩되어 들어갑니다. 이러한 구조는 아래 그림에 요약되어 있습니다.

<p align="center">
<img src="https://i.ibb.co/FVRkfbr/prior-upsampler.png" alt="prior-upsampler" border="0">
</p>

### Scalable Transformer

자기회귀적 모델링에는 Sparse Transformer를 [(Rewon Child et al., 2019)](https://arxiv.org/abs/1904.10509) 기반으로 일부를 변경하여 단순화시킨 모델을 사용하고 Scalable Transformer라고 명명합니다. Sparse Transformer는 1D 입력 시퀀스를 (block, block length) 모양의 2D 시퀀스로 바꿔서 어텐션을 계산하고 이를 위해 특별한 CUDA 커널을 사용합니다.

Scalable Transformer에서는 masked row, masked column, 그리고 unmasked previous-row attention을 사용하여 특별한 CUDA 커널 없이도 비슷한 어텐션 패턴을 만들 수 있습니다. Masked row와 masked column attention은 자기회귀적 마스크를 사용하고 unmasked previous-row attention은 이전 행 전체를 다 봅니다. 아래 그림과 같이 두 어텐션 패턴의 조합으로 각각의 위치에서 이전 위치에 대한 어텐션을 모두 표현할 수 있습니다.

<p align="center">
<img src="https://i.ibb.co/T4XX2CT/scalable-transformer.png" alt="scalable-transformer" border="0">
</p>

### Conditioner Network

중간과 아래 레벨의 업샘플러 모델에서는 상위 레벨의 토큰이 컨디셔너 네트워크(conditioner network)에 의해 업샘플링 됩니다. 컨디셔너 네트워크는 WaveNet과 같이 확장된 컨볼루션과 전치 컨볼루션으로 이루어져 있습니다. 그 구조는 아래 그림에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/djkh3TR/conditioner-network.png" alt="conditioner-network" border="0">
</p>

### Lyrics Conditioning

보컬이 있는 음악의 경우 가사를 조건부 입력으로 넣어줍니다. 가사와 노래 사이의 정렬을 맞추기 위해 먼저 각각의 곡에서 Spleeter로 [(Romain Hennequin et al., 2019)](https://archives.ismir.net/ismir2019/latebreaking/000036.pdf) 보컬을 추출하고 NUS AutoLyricsAlign을 [(Chitralekha Gupta et al., 2020)](https://ieeexplore.ieee.org/abstract/document/9054567) 적용하여 보컬과 가사의 단어 레벨 정렬을 얻습니다.

가사의 조건부 입력은 위 레벨에만 사용됩니다. 인코더-디코더 스타일을 사용하여 인코더는 가사 토큰을 자기회귀적으로 예측하도록 학습되는 트랜스포머이고 디코더는 위 레벨 사전분포를 모델링하는 트랜스포머입니다. 디코더의 시퀀스가 쿼리가 되고 인코더의 마지막 출력을 키와 밸류로 하여 어텐션을 계산하는 인코더-디코더 어텐션 층이 디코더의 중간에 추가됩니다. 이러한 구조는 아래 그림에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/ZY5Rn8w/lyrics-conditioning.png" alt="lyrics-conditioning" border="0">
</p>

### Decoder Pretraining

가사 조건을 사용하는 모델을 학습시키기 위해 필요한 계산량을 줄이기 위해 사전학습된 가사 조건 없는 위 레벨 사전분포 모델을 사용하고 가사 인코더를 추가하는 model surgery 방법을 [(Christopher Berner et al., 2019)](https://arxiv.org/abs/1912.06680) 이용합니다. 추가된 가사 인코더의 MLP와 어텐션 층들의 가중치를 0으로 초기화하여 모델이 사전학습된 것과 동일하게 작동하지만 그래디언트가 흐르면서 인코더가 학습되게 하는 방법입니다.

### Sampling

VQ-VAE와 위 레벨 사전분포, 업샘플러 모델이 다 학습되고 나면 위 레벨 사전분포에서의 샘플링으로부터 새로운 음악을 생성할 수 있습니다.

기본적으로는 위 레벨에서 먼저 첫 토큰을 샘플링하고 자기회귀적으로 다음 토큰들을 생성하면서 하위 레벨의 토큰들도 차례로 업샘플러에 의해 샘플링되는 ancestral sampling 방법을 사용합니다. 생성된 아래 레벨의 토큰들은 마지막에 VQ-VAE 디코더에 의해 오디오로 디코딩됩니다. 이러한 샘플링 방법은 아래 그림에 표현되어 있습니다.

<p align="center">
<img src="https://i.ibb.co/5M9vJ17/ancestral-sampling.png" alt="ancestral-sampling" border="0">
</p>

트랜스포머의 문맥(context) 길이보다 더 긴 시퀀스를 생성하기 위해서는 windowed sampling 방법을 사용합니다. 각 레벨에서 샘플링할 때 이전의 정해진 구간에 기반해서 다음 샘플들을 생성하고 윈도우를 이동시키는 것을 반복하면서 생성해나가는 방법입니다. 이러한 방법은 아래 그림에 표현되어 있습니다.

<p align="center">
<img src="https://i.ibb.co/PtxnZLR/windowed-sampling.png" alt="windowed-sampling" border="0">
</p>

실제 존재하는 곡의 일부분으로부터 시작하여 다음 토큰들을 이어서 생성하는 방법도 사용할 수 있습니다. 이러한 primed sampling 방법은 아래 그림에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/BjGqrxv/primed-sampling.png" alt="primed-sampling" border="0">
</p>

<br><br>

## 실험

실험에는 120만개의 곡으로 이루어진 새로 수집한 데이터셋을 사용합니다. 데이터셋이 공개되어 있지는 않습니다. 하이퍼파라미터들은 논문에 자세하게 나와 있고 위, 중간, 아래 레벨의 트랜스포머 문맥 길이는 각각 약 24, 6, 1.5초에 해당합니다.

분야 특성 상 평가 방법으로 정량적인 지표가 크게 유의미하지 않으므로 몇 가지 주목하는 특징에 따라 데모 샘플을 공개하고 주관적으로 평가합니다. 실험에 사용된 데모 샘플들은 [프로젝트 페이지](https://jukebox.openai.com/)에서 전체적으로 들어볼 수 있고 특징에 따라 분류된 페이지는 아래 각각의 서브섹션 제목에 링크되어 있습니다.

### [Coherence](https://soundcloud.com/openai_audio/sets/jukebox-samples-coherence)

생성된 곡들은 대체로 리듬이나 악기, 분위기 측면에서 일관성을 보여줍니다. 하지만 코러스나 반복되는 멜로디와 같은 음악적인 패턴은 관찰되지 않고 곡의 뒷부분으로 갈수록 일관성이 점점 떨어지는 경향이 있습니다.

### [Musicality](https://soundcloud.com/openai_audio/sets/jukebox-samples-musicality)

샘플들은 주로 익숙한 음악적 하모니나 멜로디를 모방하기 때문에 아주 자연스럽게 들립니다. 하지만 생성된 멜로디는 반복이나 마무리 패턴 등의 부재로 인해 일반적으로 사람이 만든 멜로디만큼 흥미롭지는 않습니다.

### [Re-renditions](https://soundcloud.com/openai_audio/sets/jukebox-samples-re-renditions/s-IsBDzuVrO44)

같은 장르와 아티스트의 조건으로 여러 번 샘플을 생성했을 때 항상 멜로디나 분위기가 달라지고 학습 데이터셋에 있는 원본과도 다릅니다. 드럼이나 베이스는 종종 원본과 비슷하게 따라가기도 합니다.

### [Completions](https://soundcloud.com/openai_audio/sets/jukebox-samples-novel/s-OCmVIfH4il8)

실제 존재하는 곡이 12초 주어지고 이어서 샘플을 생성할 때 리듬 등이 자연스럽게 연결되면서 멜로디나 화성이 초기에는 가끔 원본과 비슷하더라도 시간이 지날수록 완전히 다른 진행으로 생성됩니다.

### [Full Tree](https://soundcloud.com/openai_audio/sets/jukebox-samples-full-tree/s-wbPtTR5KNh5)

생성되는 샘플의 다양성을 좀 더 체계적으로 분석하기 위해 1분 짜리 샘플로부터 시작해서 독립적인 4개의 1분 동안의 확장을 만듭니다. 그리고 다음 1분 동안 4개씩의 확장을 더 해서 총 16개의 3분 짜리 샘플을 만듭니다. 이 샘플들을 들어보면 초기에 같은 샘플로부터 시작됐더라도 완전히 다양하게 갈라져서 진행되는 것을 알 수 있습니다.

### [Novel Styles](https://soundcloud.com/openai_audio/sets/jukebox-samples-novel-styles/s-SMgMBHByEVd)

이 실험은 원래의 아티스트에 맞지 않는 장르로 곡을 생성하는 것입니다. Joe Banamassa와 Frank Sinatra의 경우처럼 그럴듯한 결과도 있지만 대체로 생성 시에 입력한 장르와 맞지 않는 샘플이 생성됩니다.

### [Novel Voices](https://soundcloud.com/openai_audio/sets/jukebox-samples-novel-voice/s-Erfshq53w9W)

두 아티스트의 목소리를 보간하는(interpolate) 실험 결과도 대체로 좋지는 않습니다. 반면 곡의 중간에 조건을 바꾸어 듀엣처럼 생성하는 실험의 경우에는 그럴듯한 결과의 샘플이 생성됩니다.

### [Novel Lyrics](https://soundcloud.com/openai_audio/sets/jukebox-samples-novel-lyrics/s-qc1XhCOSjLw)

학습 데이터셋에 없는 새로운 가사를 조건으로 입력하여 곡을 생성하는 경우의 결과는 괜찮은 편입니다. 발음도 대체로 자연스럽고 가사와 멜로디의 연결도 어색하지 않습니다.

### [Novel Riffs](https://soundcloud.com/openai_audio/sets/jukebox-samples-novel-riffs/s-lo81x4FZFs2)

학습 데이터셋에 없는 새로운 짧은 음악 조각을 입력해서 특정한 아티스트나 장르의 스타일로 이어서 생성하는 경우에는 넣어준 분위기나 진행을 자연스럽게 이어가면서 조건의 스타일과도 잘 부합하는 좋은 결과가 나옵니다.

### VQ-VAE Ablations

VQ-VAE의 하이퍼파라미터, 구조 등을 바꿔가면서 여러 가지 제거(ablation) 실험도 진행합니다. 아래 표는 각각의 레벨에서 홉 길이가 바뀌는 것에 따른 재구성 에러를 나타낸 것입니다. Spectral convergence로 표현되어 값이 낮을수록 높은 재구성 품질에 해당합니다.

<p align="center">
<img src="https://i.ibb.co/C1XkGmW/vqvae-ablation.png" alt="vqvae-ablation" border="0">
</p>

이때의 스펙트로그램은 아래 그림과 같은데 첫 번째 줄이 실제값(ground truth)이고 두 번째 줄은 왼쪽부터 아래, 중간, 위 레벨입니다.

<p align="center">
<img src="https://i.ibb.co/CmDKvG0/levels-spectrogram.png" alt="levels-spectrogram" border="0">
</p>

위의 표에는 코드북 붕괴를 방지하기 위한 재시작 유무에 따른 결과도 나와 있습니다. 아래 레벨에서 재시작 유무에 따른 차이가 있고 아래 그림을 보면 재시작을 안할 때에는 초반에 코드북의 엔트로피가 확실히 낮은 것을 볼 수 있습니다.

<p align="center">
<img src="https://i.ibb.co/YNj7Bgg/restart-codebook.png" alt="restart-codebook" border="0">
</p>

스펙트럼 손실을 제외하고 파형의 L2 손실만으로 학습하는 경우의 스펙트로그램은 아래 그림에 나와 있고 왼쪽부터 아래, 중간, 위 레벨입니다.

<p align="center">
<img src="https://i.ibb.co/8N2TvCH/without-spectral-loss.png" alt="without-spectral-loss" border="0">
</p>

중간과 위 레벨의 높은 주파수 영역에서 상당히 큰 차이가 있는 것이 보입니다. 마지막으로 각 레벨의 오토인코더를 분리하지 않고 학습하는 경우에는 아래 그림과 같이 중간과 위 레벨에서 정보를 제대로 담지 못합니다.

<p align="center">
<img src="https://i.ibb.co/gW1ZLKh/decoder-once.png" alt="decoder-once" border="0">
</p>

<br><br>

## Reference

[Prafulla Dhariwal, Heewoo Jun, Christine Payne, Jong Wook Kim, Alec Radford and Ilya Sutskever. Jukebox: A Generative Model for Music. arXiv preprint, 2020.](https://arxiv.org/abs/2005.00341)

[Official Source Code of Jukebox](https://github.com/openai/jukebox)