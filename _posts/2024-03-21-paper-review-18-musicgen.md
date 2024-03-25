---
layout: post
title: "[논문 리뷰] Simple and Controllable Music Generation"
image: https://i.ibb.co/HxbfPhS/thumbnail.png
date: 2024-03-21
tags: 
categories: paper-review
use_math: true
---

## 논문 개요

<!-- excerpt-start -->

Meta AI에서 개발한 MusicGen은 텍스트 또는 멜로디를 조건으로 입력받아 음악을 생성하는 모델입니다. 구글에서 개발한 MusicLM과 [(Andrea Agostinelli et al., 2023)](http://arxiv.org/abs/2301.11325) 유사하게 사전학습된 오디오 코덱으로 만든 이산적인(discrete) 토큰을 트랜스포머 디코더를 이용하여 자기회귀적(autoregressive)으로 생성하는 방법을 사용합니다.

MusicGen의 가장 큰 특징은 RVQ의 사용으로 인해 한 타임스텝에서 발생하는 여러 코드북의 토큰들을 병렬적으로 같이 생성하여 계산 비용을 효과적으로 감소시킨다는 것입니다. 논문에서는 여러 가지 코드북 패턴을 제안하고 실험적으로 각각이 성능과 계산 비용에 미치는 영향을 분석합니다.

<br><br>

## Audio Tokenization

MusicGen은 음악을 생성할 때 먼저 잠재 공간(latent space)에서 이산적인 토큰의 형태로 생성한 뒤 오디오로 디코딩하는 방식을 사용합니다. 오디오의 토큰 인코딩과 디코딩을 학습하는 모델로는 EnCodec을 [(Alexandre Défossez et al., 2022)](https://openreview.net/forum?id=ivCd8z8zR2) 사용합니다. EnCodec 모델은 컨볼루션 오토인코더와 양자화(quantization)를 위한 Residual Vector Quantizer (RVQ)로 이루어져 있습니다.

길이 $\small d$ 와 샘플 레이트 $\small f_s$ 의 오디오 파형(waveform) 입력 $\small X \in \mathbb{R}^{d \cdot f_s}$ 은 프레임 레이트 $\small f\_r \ll f\_s$ 의 연속적인 텐서로 먼저 인코딩되고 크기 $\small M$ 의 코드북 $\small K$ 개를 이용하여 $\small Q \in \\{ 1, \ldots, M \\}^{d \cdot f_r \times K}$ 로 양자화됩니다. 즉, 양자화가 끝나면 길이가 $\small T = d \cdot f_r$ 인 $\small K$ 개의 병렬적인 토큰 시퀀스가 얻어집니다. RVQ에서 각각의 양자화기는 이전 단계에서의 잔차(residual)를 처리하는 것이므로 첫 번째 코드북이 가장 중요한 정보를 포함하고 있습니다.

<br><br>

## Codebook Interleaving Patterns

자기회귀적인 딥러닝 모델은 길이 $\small S$ 의 시퀀스 $\small U \in {1, \ldots, M}$ 을 추정하는 $\small \tilde U$ 를 다음과 같은 확률 분포로 모델링합니다.

<br>
\begin{equation}
\forall t > 0, \quad \mathbb{P} \left[ \tilde{U}\_t \vert \tilde{U}\_{t-1}, \ldots, \tilde{U}\_0 \right] = p\_t \left( \tilde{U}\_{t-1}, \ldots, \tilde{U}\_0 \right)
\end{equation}
<br>

만약 확률분포 $\small p$ 를 딥러닝 모델로 완벽하게 예측할 수 있다면 시퀀스 $\small U$ 를 정확하게 추정할 수 있는 것입니다.

EnCodec 모델에서 얻어지는 $\small Q$ 에는 각 타임스텝마다 $\small K$ 개의 코드북이 사용되기 때문에 분포를 정확하게 추정하려면 $\small Q$ 를 펼쳐서(flatten) 길이 $\small S = d \cdot f\_r \cdot K$ 의 시퀀스를 만들어야 합니다. 즉, 처음에 첫 번째 타임스텝에서 첫 번째 코드북을 예측하고 다음에 두 번째 코드북의 첫 번째 타임스텝을 예측하는 식으로 나아가는 식입니다. 이러한 flattening pattern은 아래 그림의 왼쪽 위에 표현되어 있습니다.

<p align="center">
<img src="https://i.ibb.co/cv0TZMs/codebook-patterns.png" alt="codebook-patterns" border="0">
</p>

하지만 이러한 방식은 계산량을 증가시킨다는 단점이 있습니다. 다른 방법으로는 $\small p_t$ 에 대한 추정에 여러 개의 모델을 사용할 수도 있습니다. 예를 들어 MusicLM은 두 개의 모델을 사용하여 하나는 처음 몇 개 코드북의 coarse token을 모델링하고 다른 하나는 나머지의 fine token을 모델링합니다. 이 경우에도 자기회귀적 스텝 수는 $\small d f\_r \cdot K$ 로 동일합니다.

다른 가능한 방법은 몇 개의 코드북을 병렬적으로 예측하는 것입니다. 예를 들어 $\small t \in \\{1, \ldots, T \\}$ 와 $\small k \in \\{1, \ldots, K \\}$ 에 대해서 시퀀스 $\small V_{t, k} = Q_{t, k}$ 를 추정하는 $\small \tilde{V}$ 를 다음과 같이 타임스텝 $\small t$ 에서 모든 코드북을 한 번에 예측하는 확률 분포로 모델링할 수 있습니다.

<br>
\begin{equation}
\forall t > 0, \quad \forall k, \quad \mathbb{P} \left[ \tilde{V}\_{t, k} \right] = p\_{t, k} \left( \tilde{V}\_{t-1}, \ldots, \tilde{V}\_0 \right)
\end{equation}
<br>

여기서 $\small V\_t$ 는 타임스텝 $\small t$ 에서 모든 코드북을 연결한(concatentate) 것으로 $\small k$ 를 생략한 것입니다. 이 방식은 위 그림의 parallel pattern에 해당합니다.

이 경우에는 각각의 타임스텝에서 코드북들 사이의 종속성이 고려되지 못하기 때문에 $\small \tilde{V}$ 가 $\small V$ 를 정확하게 추정하지 못합니다. 또한 $\small t$ 가 증가할수록 에러가 누적되기 때문에 두 분포 간의 차이는 점점 더 벌어집니다. 하지만 이러한 방법은 긴 시퀀스에 대해 계산량을 상당히 줄일 수 있다는 장점이 있습니다.

이와 같이 다양한 코드북 상호 배치(interleaving) 패턴들을 실험하고 성능에 대한 영향을 분석하기 위해 다음과 같은 표기법을 정의합니다. 먼저 $\small \Omega = \\{ (t, k) : t \in \\{1, \ldots, d \cdot f_r \\}, k \in \\{1, \ldots, K \\} \\}$ 를 모든 타임스텝과 코드북 인덱스에 대한 쌍이라고 합니다. 코드북 패턴 시퀀스는 모든 $\small 0 < s \leq S$ 와 $\small \Omega$ 의 부분집합 $\small P \subset \Omega$ 에 대해서 $\small P = (P\_0, P\_1, P\_2, \ldots, P\_S)$ 입니다. 편의상 $\small P\_s$ 안에서 각각의 코드북 인덱스는 최대 한 번만 나타날 수 있다고 제한합니다.

이러한 표기법에 따라서 parallel pattern은 다음과 같이 정의할 수 있습니다.

<br>
\begin{equation}
P\_s = \\{ (s, k) : k \in \\{ 1, \ldots, K \\} \\}
\end{equation}
<br>

또한 위 그림에 나와 있는 delay pattern은 아래와 같이 표현됩니다.

<br>
\begin{equation}
P\_s = \\{ (s - k + 1, k) : k \in \\{ 1, \ldots, K \\}, s - k \geq 0 \\}
\end{equation}
<br>

<br><br>

## Model Conditioning

자기회귀적 모델의 차원(dimension)이 $\small D$ 일 때 텍스트 조건은 $\small C \in \mathbb{R}^{T_C \times D}$ 의 텐서 형태로 임베딩되어 입력됩니다. 텍스트 임베딩 모델로는 T5 인코더 [(Colin Raffel et al., 2020)](https://www.jmlr.org/papers/v21/20-074.html), FLAN-T5 [(Hyung Won Chung et al., 2022)](https://arxiv.org/abs/2210.11416), 그리고 CLAP을 [(Yusong Wu et al., 2023)](https://ieeexplore.ieee.org/abstract/document/10095969/) 사용하여 서로 비교합니다.

텍스트 인코더를 통해 임베딩된 연속적인 벡터는 RVQ를 이용하여 양자화됩니다. 음악과 텍스트의 결합 임베딩(joint embedding)을 추출하는 CLAP의 경우에는 학습 시에는 음악 임베딩을 사용하고 추론 시에는 텍스트 임베딩을 사용합니다.

MusicGen은 텍스트 조건에 더해서 어떤 음악의 멜로디나 사람의 휘파람 혹은 흥얼거림을 입력 조건으로 같이 사용하기도 합니다. 멜로디 조건은 크로마그램의 형태로 사용합니다.

추출된 크로마그램은 낮은 주파수 영역에 지배적인 영향을 받는 것이 초기 실험 결과 발견되었습니다. 따라서 먼저 Demucs를 [(Alexandre Défossez et al., 2019)](https://arxiv.org/abs/1911.13254) 사용하여 멜로디 조건 오디오를 드럼, 베이스, 보컬, 나머지의 네 요소로 분리한 뒤 드럼과 베이스를 제거합니다. 나머지 신호에서 크로마그램을 얻어낸 뒤 각각의 타임스텝에서 argmax를 적용하여 양자화한 것을 멜로디 조건 입력으로 사용합니다.

<br><br>

## Model Architecture

### Codebook Projection and Positional Embedding

코드북 패턴의 각 스텝 $\small P_s$ 에는 몇몇 코드북만 존재합니다. 존재하는 코드북에서는 대응하는 $\small D$ 차원의 임베딩 벡터를 매핑시키고 존재하지 않는 경우에는 이를 나타내는 스페셜 토큰을 사용합니다. $\small P\_0$는 항상 모든 스페셜 토큰의 합이 됩니다. 또한 현재 스텝 $\small s$ 를 알려주기 위한 사인파 기반 위치 임베딩(positional embedding)이 더해집니다.

### Transformer Decoder

트랜스포머에는 $\small D$ 차원의 입력이 들어갑니다. 먼저 인과적(causal) 셀프 어텐션 블럭이 있고 텍스트 조건 시퀀스 $\small C$ 와의 크로스 어텐션을 계산하는 블럭이 존재합니다. 멜로디 조건의 경우에는 크로스 어텐션 대신 입력의 맨 앞에 조건 시퀀스를 붙이는 방법을 사용합니다.

패턴 스텝 $\small P_s$ 에서의 트랜스포머 디코더의 출력은 다음 스텝 $\small P_{s+1}$ 의 $\small Q$ 의 값을 예측하도록 학습됩니다. 각각의 코드북마다 다른 선형 층(linear layer)이 있어 $\small D$ 차원 벡터를 $\small N$ 개의 로짓으로 바꿔줍니다. 샘플링에는 250개 샘플에 대한 top-k 샘플링 방법을 사용합니다.

<br><br>

## 실험

실험에 사용한 EnCodec은 코드북 크기 2048을 갖는 4개의 양자화기를 사용합니다. EnCodec의 학습에는 오디오 시퀀스에서 임의로 잘라낸 1초 짜리 조각을 사용합니다.

트랜스포머 디코더는 300M, 1.5B, 3.3B 개의 파라미터를 갖는 3개의 모델을 학습하여 비교합니다. 학습에 사용하는 오디오는 임의로 잘라낸 30초 짜리 조각입니다.

기본적인 실험 결과는 코드북 패턴으로 "delay pattern"을 사용하고 텍스트 인코더로 T5 인코더를 사용한 것입니다.
데이터셋은 ShutterStock에서 25K 개, Pond5에서 365K 개, 그리고 내부적으로 준비한 10K 개의 음악 트랙으로 이루어진 총 20K 시간 짜리 데이터셋을 사용합니다. 모든 트랙의 샘플 레이트는 32 kHz이고 텍스트 설명과 장르, BPM, 태그 등의 정보가 있는 메타데이터가 포함되어 있습니다. 평가 데이터셋으로는 MusicLM과 동일하게 MusicCaps를 사용합니다.

베이스라인으로는 텍스트 기반 음악 생성 모델들인 Riffusion, Mousai, MusicLM, 그리고 Noise2Music을 사용합니다. 실험 결과에 대한 데모 오디오는 [샘플 웹페이지](https://ai.honu.io/papers/musicgen/)에서 들어볼 수 있습니다.

### Evaluation Metrics

정량적인 평가 지표로는 Fréchet Audio Distance (FAD), Kullback-Leiber Divergence (KL), 그리고 CLAP 점수를 사용합니다. FAD 값은 낮을수록 그럴듯한 오디오를 생성한 것을 의미합니다. 낮은 KL 발산 값은 생성된 음악이 타겟과 유사한 레이블 분포를 가지고 있다는 것을 나타냅니다. CLAP 점수는 텍스트 설명과 생성된 오디오 사이의 연관성을 평가합니다.

주관적 평가는 전체적인 품질(overall quality, Ovl)과 텍스트 조건과의 연관성(relevance, Rel)을 기준으로 이루어집니다. 평가 참가자들은 샘플을 듣고 각각의 기준에 맞게 1에서 100점 사이로 점수를 부여합니다.

### Comparison with the Baselines

MusicGen을 다른 베이스라인 모델들과 비교한 결과는 아래 표에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/YBt4kS4/results.png" alt="results" border="0">
</p>

정량적인 지표와 주관적 평가 모두 MusicGen이 가장 우수한 성능을 보여주는 결과입니다. MusicLM과는 비슷한 수준의 값을 나타내고 실제로 샘플을 들어봤을 때에도 우열을 가리기는 어렵습니다. 또한 멜로디 조건을 추가했을 때 MusicGen의 정량적 지표 값은 낮아지지만 주관적 평가 수치에는 큰 차이가 없습니다.

### Melody Evaluation

텍스트와 멜로디 조건을 같이 사용했을 때의 평가를 위해 정량적 지표로 chroma cosine-similarity를 추가합니다. 동일한 타임스텝에서 생성된 오디오와 타겟의 양자화된 크로마 사이의 코사인 유사도의 평균을 계산한 것입니다. 또한 주관적 평가 기준도 생성된 오디오가 타겟과 얼마나 유사한 멜로디를 나타내는지 평가하는 Mel을 추가합니다.

<p align="center">
<img src="https://i.ibb.co/zF04KCX/chroma-result.png" alt="chroma-result" border="0">
</p>

위 표의 결과를 보면 텍스트와 멜로디 조건을 같이 사용했을 때 멜로디에 대한 평가 수치가 상당히 올라갑니다. 반면 멜로디 조건을 제외하더라도 Ovl과 Rel 측면에서는 성능 하락이 거의 없습니다. 크로마그램을 시각화한 그림을 보면 멜로디 조건의 유무에 따라 타겟 오디오의 멜로디를 따르는 정도가 얼마나 차이 나는지 알 수 있습니다. 아래 그림에서 모든 샘플은 "90s rock song with electric guitar and heavy drums" 라는 동일한 텍스트 조건으로 생성되었고 각 줄에 따라 멜로디 조건이 다릅니다.

<p align="center">
<img src="https://i.ibb.co/vmnyVTH/chroma-examples.png" alt="chroma-examples" border="0">
</p>

### Fine-tuning for Stereophonic Generation

스테레오 음악 생성을 위한 설정으로 EnCodec에 왼쪽 오른쪽을 분리하여 넣어주고 $\small 2 \cdot K = 8$ 개의 코드북을 사용합니다. 코드북 패턴으로는 delay pattern의 두 가지 변형을 사용하는데 하나는 stereo delay이고 다른 하나는 stereo partial delay입니다. 아래 그림에 두 패턴이 표현되어 있습니다.

<p align="center">
<img src="https://i.ibb.co/3pWsDY3/stereo-patterns.png" alt="stereo-patterns" border="0">
</p>

이에 대한 실험 결과는 아래 표에 나와 있습니다. Stereo partial delay 패턴이 더 우세한 성능을 보여주고 모노일 때보다 스테레오일 때 주관적인 평가 결과도 더 좋습니다.

<p align="center">
<img src="https://i.ibb.co/BqdXYys/stereo-results.png" alt="stereo-results" border="0">
</p>

### The Effect of Codebook Interleaving Patterns

앞에서 본 실험 결과들은 모두 delay pattern을 기준으로 한 것입니다. 위에 있는 코드북 패턴 그림에 없는 partial flattening과 partial delay pattern은 아래 그림에 표현되어 있습니다.

<p align="center">
<img src="https://i.ibb.co/ygV5bWT/partial-patterns.png" alt="partial-patterns" border="0">
</p>

코드북 패턴에 따른 실험 결과는 아래 표에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/tq4Z837/pattern-results.png" alt="pattern-results" border="0">
</p>

성능 측면에서는 flattening 패턴이 가장 좋다는 것을 알 수 있습니다. 하지만 계산량을 고려하면 delay pattern이나 partial delay pattern이 효율적이면서도 성능 하락이 크지 않습니다.

### Memorization Experiment

MusicLM 논문에서 했던 것과 같이 모델이 학습 데이터셋을 외워버려서 비슷한 음악을 생성하는지도 평가합니다. 이를 위해 학습 데이터셋에서 임의로 선택된 샘플로부터 5초 길이에 해당하는 250개의 토큰을 생성합니다. 모든 시퀀스에 대해 토큰들이 일치하는 샘플의 비율인 exact match와 80% 이상의 토큰이 일치하는 partial match를 기준으로 평가합니다.

<p align="center">
<img src="https://i.ibb.co/31ksxHW/memorization-results.png" alt="memorization-results" border="0">
</p>

위 그래프에서 실선은 exact match, 점선은 partial match입니다. MusicGen도 MusicLM과 마찬가지로 모델이 학습 데이터셋을 외워서 음악을 그대로 생성하는 비율은 매우 낮은 편입니다.

<br><br>

## Reference

[Jade Copet, Felix Kreuk, Itai Gat, Tal Remez, David Kant, Gabriel Synnaeve, Yossi Adi and Alexandre Défossez. Simple and Controllable Music Generation. In NeurIPS, 2023.](https://proceedings.neurips.cc/paper_files/paper/2023/hash/94b472a1842cd7c56dcb125fb2765fbd-Abstract-Conference.html)

[Official Source Code of MusicGen](https://github.com/facebookresearch/audiocraft)