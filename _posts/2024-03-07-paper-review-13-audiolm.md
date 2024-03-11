---
layout: post
title: "[논문 리뷰] AudioLM: A Language Modeling Approach to Audio Generation"
image: https://i.ibb.co/R7LJvNd/thumbnail.png
date: 2024-03-07
tags: 
categories: paper-review
use_math: true
---

## 논문 개요

<!-- excerpt-start -->

오디오 생성 모델이 달성해야 하는 중요한 목표 두 가지는 높은 품질과 긴 시간 동안의 일관성입니다. 많은 오디오 생성 모델들이 이 두 가지 중 한 부분에 우수한 성능을 보이면서 다른 한 부분에는 상대적으로 약점이 있는 특징을 갖습니다. AudioLM은 오디오 품질을 높이기 위한 acoustic token과 의미 정보를 담기 위한 semantic token으로 역할을 나눈 두 토큰을 모두 사용함으로써 두 가지 목표를 모두 달성할 수 있도록 개발된 오디오 생성 모델입니다.

AudioLM은 텍스트 프롬프트 조건이 없는 음성 생성, 텍스트 조건이 있는 음성 생성, 피아노 음악 생성 모두에 대해 고품질의 일관적인 오디오를 생성해냅니다. 특히 음성에 대해서는 발화자의 목소리, 억양, 녹음 환경 소리 등이 일정하게 유지되는 우수한 성능을 보여주면서 텍스트 조건이 없을 때에도 웅얼거리는 소리가 아닌 의미를 담은 음성을 만들 수 있습니다.

<br><br>

## Model Components

모델의 입력은 단일 채널의 오디오 시퀀스 $\small x \in \mathbb{R}^T$ 입니다. 이 입력은 AudioLM의 세 가지 구성요소를 통해 처리됩니다.

첫 번째로는 $\small x$ 를 이산적인(discrete) 토큰들의 시퀀스 $\small h = (h_1, \ldots, h_{T^{\prime}})$ 으로 매핑하는 토크나이저(tokenizer) 모델이 있습니다. 이떄 $\small T^{\prime}$ 이 $\small T$ 보다 훨씬 작기 때문에 모델이 처리해야 하는 계산량을 효과적으로 줄여줄 수 있습니다.

두 번째로는 토큰들의 우도(likelihood) $\small \pi_{t=1}^{T^{\prime}} p(h_t \vert h_{<t})$ 를 최대화시키도록 학습되는 트랜스포머 디코더 언어 모델이 있습니다. 이 모델은 추론(inference) 시에 토큰 시퀀스 $\small \hat{h}$ 를 자기회귀적(autoregressive)으로 생성합니다.

마지막으로 디토크나이저(detokenizer) 모델은 생성된 토큰들을 다시 오디오 파형(waveform) $\small \hat{x}$ 로 매핑합니다. 토크나이저와 디토크나이저는 언어 모델을 학습하기 전에 사전학습(pre-train)과 프리징이 됩니다.

<br><br>

## Trade-Offs of Discrete Audio Representations

토크나이저와 디토크나이저로 이산적인 오디오 표현을 만들어서 달성하고자 하는 것은 두 가지입니다. 첫 번째는 오디오 파형을 높은 품질로 복원하는 것입니다. 이를 위해서는 비트레이트가 높고 토큰 시퀀스 길이가 길수록 좋습니다. 다른 하나는 축약된 표현에 최대한 긴 시간 동안(long-term)의 정보를 담는 것입니다. 이것은 짧은 토큰 시퀀스 길이를 요구합니다.

이렇게 두 상충되는 목표를 같이 달성하기 위해 AudioLM은 두 가지 토큰을 같이 조합하여 사용합니다. Semantic token은 긴 시간 동안의 일관성(coherence)을 담당하고 acoustic token은 semantic token에 조건화되어 고품질의 오디오 합성을 가능하게 합니다. 이러한 토크나이저 구조는 아래 그림에 요약되어 있습니다.

<p align="center">
<img src="https://i.ibb.co/k8WP8r2/tokenizer.png" alt="tokenizer" border="0">
</p>

Acoustic token은 SoundStream을 [(Neil Zeghidour et al., 2021)](https://ieeexplore.ieee.org/abstract/document/9625818) 사용하여 만듭니다. SoundStream에 대한 자세한 설명은 [이전 포스트](https://jeffmung.github.io/2024/02/27/paper-review-10-soundstream/)를 참고하면 좋습니다.

SoundStream의 기본적인 세팅으로는 16 kHz의 입력 오디오가 50 Hz의 임베딩을 만들도록 스트라이드를 설정합니다. 즉, $\small 16000 / 50 = 320$ 배의 샘플 레이트 축소가 이루어집니다. 각각의 임베딩은 $\small Q$ 개의 계층적인 VQ로 이루어진 RVQ를 통해 양자화됩니다. 예를 들어 $\small N=1024$ 개의 코드북 크기를 가진 $\small Q=4$ 의 RVQ는 $\small 50 \cdot 4 \cdot \log_2 1024 = 2000$ bps의 비트레이트로 작동합니다.

결과적으로 입력 오디오 시퀀스 $\small x$ 는 양자화된 코드 시퀀스 $\small Y \in \\{ 1, \ldots, N \\}^{T_A \times Q}$ 로 임베딩됩니다. 그 뒤 SoundSteam의 디코더는 다시 이산적인 표현 $\small Y$ 를 연속적인 임베딩으로 매핑하고 파형을 복원합니다. 학습은 재구성 손실(reconstruction loss)과 적대적 손실(adversarial loss)을 같이 사용하여 이루어집니다.

Semantic token은 w2v-BERT를 [(Yu-An Chung et al., 2021)](https://ieeexplore.ieee.org/abstract/document/9688253) 이용하여 학습됩니다. w2v-BERT에 대한 자세한 설명도 [이전 포스트](https://jeffmung.github.io/2024/03/06/paper-review-12-w2vbert/)에 있습니다. 기존 논문의 모델은 주로 음성 인식(speech recognition)에 초점을 맞추어 유용한 표현을 자기지도 학습(self-supervised learning)으로 추출하지만 AudioLM에서는 사전학습된 w2v-BERT를 오디오의 긴 시간적 구조를 반영할 수 있는 표현 추출 모델로써 사용합니다.

구체적으로는, w2v-BERT의 MLM 모듈 중간 층에서 나온 특징(feature) 임베딩을 이용하여 토큰을 만듭니다. 이 임베딩들에 $\small k$-means를 적용하여 $\small K$ 개의 클러스터를 만든 뒤 센트로이드를 semantic token으로 사용합니다. 클러스터링을 하기 전 w2v-BERT 임베딩의 각 차원을 정규화 하는 것이 성능 향상에 도움을 줍니다.

w2v-BERT에서는 시간 축으로 다운샘플링이 이루어져 1024 차원의 특징 임베딩이 25 Hz로 만들어집니다. 따라서 입력 오디오 시퀀스 $\small x$ 는 semantic token 시퀀스 $\small z = (z_1, \ldots, z_{T_S}) \in \\{ 1, \ldots, K \\}^{T_S}$ 가 됩니다. $\small T_S = T/640$ 이며 예를 들어 $\small T=16000$ 이고 $\small K = 1024$ 일 때 semantic token은 250 bps의 비트레이트에 대응됩니다.

논문에서는 두 토큰의 서로 다른 특성을 비교하기 위해 각각의 토큰으로부터 오디오 복원 품질(reconstruction quality)과 음소 판별 능력(phonetic discriminability)을 평가합니다. 복원 품질 측정에는 값이 높을수록 좋은 VisQOL 점수를 사용하고 음소 판별에는 거리 기반의 ABX 에러율(error rate)을 사용하여 값이 낮을수록 좋은 성능을 나타냅니다. 그 결과는 아래 표에 있습니다.

<p align="center">
<img src="https://i.ibb.co/XSgkNh7/tokens-property.png" alt="tokens-property" border="0">
</p>

Acoustic token은 높은 복원 품질을 나타내지만 음소 판별 능력이 낮고, 반대로 semantic token은 음소 판별 능력이 높지만 비트레이트를 높여도 복원 품질이 떨어지는 것을 볼 수 있습니다.

또한 acoustic token만을 사용하여 시퀀스를 예측하도록 트랜스포머 디코더를 학습한 경우에 생성된 오디오를 들어보면 발화자의 목소리나 녹음 환경은 유지되지만 언어적인 내용에 일관성이 없고 옹알거리는 경우가 많습니다. 따라서 높은 품질과 긴 시간 동안의 일관성을 모두 달성하려면 두 토큰을 같이 사용하는 것이 효과적이라는 것을 짐작할 수 있습니다.

<br><br>

## Hierarchical Modeling of Semantic and Acoustic Tokens

두 토큰을 같이 사용하는 프레임워크를 만들기 위해 AudioLM에서는 먼저 semantic token의 전체 시퀀스를 모델링하고 다음에 이를 조건부로 사용하여 acoustic token을 예측하는 계층적인 구조를 설계합니다. 또한 acoustic token의 생성을 coarse acoustic modeling과 fine acoustic modeling의 두 단계로 나누어 전체 계층은 세 레벨이 됩니다. 그 구조는 아래 그림에 요약되어 있습니다.

<p align="center">
<img src="https://i.ibb.co/DYFLqwg/hierarchical-structure.png" alt="hierarchical-structure" border="0">
</p>

모든 단계에서는 이전의 실제값(ground truth) 토큰들이 주어졌을 때 다음의 토큰을 예측하도록 학습되는 트랜스포머 디코더를 각 단계마다 따로 사용합니다. 이러한 계층적인 구조는 계산량을 효과적으로 줄여주고, semantic token이 이전의 acoustic token에 독립적이라는 가정을 반영할 수 있다는 장점이 있습니다. 즉, $\small p(z_t \vert z_{<t}, y_{<t}) \approx p(z_t \vert z_{<t})$ 입니다.

첫 번째 단계의 semantic modeling에서는 자기회귀적으로 semantic token $\small p(z_t \vert z_{<t})$ 를 예측합니다.

두 번째 단계의 coarse acoustic modeling에서는 SoundStream의 RVQ 중에 첫 $\small Q^{\prime}$ 개의 VQ에서 나온 acoustic token을 예측합니다. 이때 전체 semantic token 시퀀스 $\small z$ 는 조건부로 들어갑니다. 따라서 두 번째 단계에서 모델링하는 것은 $\small p(y_t^q \vert z, y_{<t}^{\leq Q^{\prime}}, y_t^{<q})$ 입니다. 여기서 $\small q \leq Q^{\prime}$ 입니다.

RVQ의 계층적 토큰들은 평평해진(flattened) 상태로 입력됩니다. 예를 들어 $\small y_t^q$ 가 $\small q$ 번째 VQ가 만든 $\small t$ 번째 타임스텝의 토큰이면 $\small Y \in \\{1, \ldots, N\\}^{T_A \times Q}$ 는 원래 토큰 $\small y$ 에 VQ 오프셋 $\small o$ 를 반영한 $\small T_A \cdot Q$ 길이의 $\small y + o$ 가 됩니다. 풀어서 표기하면 $\small y = (y_1^1, y_1^2, \cdots, y_1^Q, y_2^1, \ldots, y_{T_Q}^Q)$ 이고 $\small o_i = [(i - 1) \, \text{mod} \, Q] \cdot N$ 일 때 $\small o = (o_1, o_2, \ldots, o_{T_A \cdot Q})$ 입니다. 이후의 표기에서는 가독성을 위해 오프셋은 생략하겠습니다.

결과적으로 coarse acoustic modeling 단계까지 예측되는 전체 시퀀스는 $\small (z_1, z_2, \ldots, z_{T_S}, y_1^1, y_1^2, \ldots, y_1^{Q^{\prime}}, y_2^1, y_2^2, \ldots, y_2^{Q^{\prime}}, \ldots, y_{T_A}^{Q^{\prime}})$ 입니다. SoundStream과 w2v-BERT의 임베딩 샘플 레이트 차이로 $\small T_A$ 와 $\small T_S$ 는 다를 수 있고 기본 설정으로는 $\small T_A = 2 T_S$ 입니다.

세 번째 단계의 fine acoustic modeling에서는 $\small Q^{\prime}$ 개의 VQ에 해당하는 coarse acoustic token을 조건부로 $\small q>Q^{\prime}$ 에 해당하는 fine acoustic token의 조건부 확률 $\small p(y_t^q \vert y^{\leq Q^{\prime}}, y_{< t}^{> Q^{\prime}}, y_t^{ < q})$ 를 모델링합니다. 이 단계의 목적은 두 번째 단계에서 남아 있는 아티팩트(artifact)를 제거하고 오디오 품질을 더 높이는 것으로 semantic token은 사용하지 않습니다.

또한 fine acoustic token의 디테일이 국소적인 coarse acoustic token에 의해 결정된다고 가정하여 세 번째 단계는 중첩되지 않은 3초 짜리 오디오 조각(chunk)들의 배치에 대해 수행됩니다. 따라서 이 단계는 타겟 오디오의 길이와 무관한 계산량을 갖게 됩니다.

<br><br>

## Inference

학습이 끝난 AudioLM으로 오디오를 생성할 때에는 조건에 따라 다른 형태의 생성 방법을 사용합니다.

### Unconditional Generation

이 세팅에서는 먼저 모든 semantic token $\small \hat{z}$ 를 조건부 없이 생성한 뒤 이것을 조건부 입력으로 넣어 acoustic token을 생성합니다.

### Acoustic Generation

이 세팅에서는 실제값 semantic token $\small z$ 를 테스트 시퀀스 $\small x$ 에서 추출하여 acoustic token 생성의 조건부 입력으로 사용합니다.

### Generating Continuations

AudioLM 논문에서 초점을 맞추는 가장 주요한 활용은 짧은 프롬프트 $\small x$ 로부터 이어지도록 오디오를 생성하는 것입니다. 이를 위해 먼저 프롬프트에서 semantic token $\small z_{\leq t_s}$ 와 coarse acoustic token $\small y_{\leq t_a}^{\leq Q^{\prime}}$ 를 만듭니다.

첫 번째 단계에서는 $\small z_{\leq t_s}$ 를 기반으로 이어지는 semantic token $\small \hat{z}\_{> t\_s}$ 을 자기회귀적으로 생성합니다. 두 번째 단계에서는 전체 semantic token 시퀀스 $\small (z\_{\leq t_s}, \hat{z}\_{> t\_s})$ 와 프롬프트의 coarse acoustic token $\small y_{\leq t_a}^{\leq Q^{\prime}}$ 를 연결한(concatenate) 뒤 조건부로 넣어줘서 이어지는 coarse acoustic token을 생성합니다. 세 번째 단계에서는 coarse acoustic token을 통해 fine acoustic token을 만듭니다.

마지막으로 프롬프트와 생성된 acoustic token을 SoundStream 디코더에 넣어줘서 파형 $\small \hat{x}$ 을 복원합니다.

<br><br>

## 실험

실험은 음성과 피아노 곡의 두 도메인에 대해 진행합니다. 음성 데이터셋으로는 Libri-Light의 [Jacob Kahn et al., 2020](https://ieeexplore.ieee.org/abstract/document/9052942) unlab-60k를 사용합니다. 피아노 데이터셋으로는 다양한 환경에서 녹음된 초보자부터 전문가까지 수준이 섞여 있는 데이터셋을 사용합니다. 피아노 실험에는 세 번째 단계를 무시하고 두 번째 단계에서 acoustic token을 모두 생성합니다.

실험에 대한 데모 사운드 샘플은 [프로젝트 웹사이트](https://google-research.github.io/seanet/audiolm/examples/)에서 들어볼 수 있습니다.

### Model Selection, Training and Inference

w2v-BERT의 MLM 모듈에서 몇 번째 중간 층을 사용할지와 $\small k$-means 클러스터 개수 $\small K$ 를 결정하기 위해 ABX 테스트를 진행하고 sWUGGY와 sBLIMP 점수를 측정합니다. 그 결과는 아래 그림에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/Gs8mb6f/model-selection.png" alt="model-selection" border="0">
</p>

sWUGGY는 "brick"과 "blick"처럼 비슷한 소리가 나지만 하나는 단어이고 하나는 존재하는 단어가 아닌 쌍에 대하여 모델이 단어에 더 높은 확률을 부여하는지 측정하는 방법입니다. sBLIMP는 "the dogs sleep"와 "the dog sleep"처럼 비슷하지만 하나만 문법적으로 옳은 쌍에 대하여 모델이 문법적으로 옳은 문장에 얼마나 자주 더 높은 확률을 부여하는지 측정합니다. 이러한 정량적 평가 결과와 추가적인 주관적 평가를 바탕으로 MLM 모듈의 7번째 층을 사용하고 $\small K=1024$ 를 기본값으로 설정합니다.

SoundStream은 1024의 코드북 크기, 12개의 RVQ 층, 그리고 스트라이드 (2, 4, 5, 8)을 사용하여 입력 오디오가 16 kHz일 때 50 Hz의 임베딩 샘플 레이트와 6000 bps의 비트레이트를 만듭니다. Coarse acoustic modeling과 fine acoustic modeling은 $\small Q^{\prime}=4$ 로 구분합니다. 따라서 세 번째 단계에서 비트레이트가 2000 bps에서 6000 bps로 증가되고 오디오 품질이 상당히 향상됩니다.

트랜스포머 디코더 언어 모델로는 모든 단계에서 동일하게 12개의 층, 16개의 어텐션 헤드, 1024의 임베딩 차원, 그리고 T5 스타일의 상대적 위치 임베딩(relative positional embedding)을 [(Colin Raffel et al., 2020)](https://www.jmlr.org/papers/v21/20-074.html) 사용합니다. 학습 시에는 세 단계의 입력을 임의로 잘라서 길이를 각각 30, 10, 그리고 3초로 맞춥니다.

추론 시에는 모든 단계에서 top-p 샘플링을 사용하고 각 단계의 온도는 0.6, 0.8, 그리고 0.6으로 설정합니다. 이 온도 값은 생성된 오디오의 다양성과 의미적 일관성 사이의 트레이드오프를 제공합니다.

### Information Represented by the Semantic Tokens

첫 번째 실험으로는 음성을 모델링할 때 언어적인 내용은 대부분 semantic token에 포착되고 발화자나 음향 환경에 대한 정보는 acoustic token에 포착된다는 가설을 확인하기 위한 실험을 진행합니다.
이를 위해 "acoustic generation" 세팅에서 생성된 음성과 원래 음성의 내용을 비교합니다. 실제값 semantic token을 추출하여 사용했기 때문에 가설에 따르면 두 음성의 내용은 동일해야 합니다. 이를 확인하기 위해 Conformer Transducer-L을 [(Anmol Gulati et al., 2020)](https://arxiv.org/abs/2005.08100) 사용하여 생성된 오디오의 음성 인식을 수행하고 word error rate(WER)와 character error rate(CER)을 계산합니다. 그 결과는 아래 표에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/FnSYMW5/cer-wer.png" alt="cer-wer" border="0">
</p>

AudioLM의 CER과 WER은 낮은 편으로 원본의 내용과 비슷한 음성을 생성한다는 것을 보여줍니다. 이를 통해 의미적인 정보는 semantic token에 담긴다는 것을 알 수 있습니다.

### Information Represented by the Acoustic Tokens

두 번째 실험으로는 발화자와 음향 환경에 대한 정보가 acoustic token에 포착된다는 가설을 바탕으로 이전 실험에서 동일한 semantic token으로부터 반복적으로 acoustic token을 샘플링했을 때 발화자와 음향 환경이 다양하게 달라지는지 확인합니다. 데모 샘플을 직접 들어봤을 때도 이를 확인할 수 있지만 정량적인 평가도 수행합니다.

먼저 로그 멜 스펙트로그램을 입력으로 받는 CNN 기반의 발화자 분류 모델을 학습시킵니다. 이 모델의 학습 데이터셋은 총 291명의 발화자를 포함하고 있습니다. 학습 데이터셋에 존재하는 발화자의 음성으로부터 AudioLM의 "acoustic generation" 세팅으로 새로운 음성을 만들었을 때 발화자 분류 모델의 정확도(accuracy)를 측정합니다. 그 결과는 아래 표에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/grNGy1n/speaker-classification.png" alt="speaker-classification" border="0">
</p>

SoundStream으로 복원했을 때 100%의 정확도가 나온 것으로 보아 코덱의 압축으로 인한 정보 손실의 영향은 크게 없습니다. "Acoustic generation" 시에는 정확도가 3.2%로 매우 낮은데 실제값 semantic token을 사용했음에도 불구하고 발화자에 대한 정보는 거의 담겨 있지 않아 학습 시에 본 발화자의 목소리를 동일하게 생성하지 않는다는 것을 알 수 있습니다.

### Probing the Linguistic Knowledge of AudioLM

이번에는 AudioLM의 semantic token이 어휘와 문법에 대한 지식을 얼마나 가지고 있는지 평가하기 위해 ZeroResource Challenge 2021의 [(Ewan Dunbar et al., 2021)](https://arxiv.org/abs/2104.14700) 기준으로 평가합니다. 이 챌린지는 제로샷으로 언어 모델을 사용하여 sWUGGY와 sBLIMP 점수를 측정하는 것입니다. sWUGGY의 경우 평가 쌍 내의 단어가 학습 데이터셋에 포함된 경우를 "in-vocab"이라고 따로 분류합니다. 그 결과는 아래 표에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/bJPthfC/linguistic-knowledge.png" alt="linguistic-knowledge" border="0">
</p>

AudioLM은 다른 언어 모델들과 비교하여 높은 성능을 나타냅니다. 즉, 음성을 생성할 때 어휘와 문법적인 측면에서 상당히 좋은 판단을 할 수 있습니다.

### Generating Coherent Continuations

발화자 분류 실험 결과 표의 마지막 열은 "generating continuation" 세팅에서의 실험 결과를 나타냅니다. 3초 짜리 프롬프트가 주어지고 semantic token과 acoustic token을 모두 추출하여 사용할 때에는 동일한 발화자의 목소리를 잘 이어지도록 생성한다는 것을 보여줍니다.

### Subjective Evaluation

"generating continuations" 실험의 결과는 주관적 평가를 실시합니다. 참가자들은 첫 3초가 원본 사람 음성이라는 사실을 알고 있는 상태에서 이후 7초의 샘플이 원본인지 생성된 음성인지 판별합니다.

1000개의 평가를 종합한 결과 올바르게 판별한 비율이 51.2%로 평가 참여자들이 실제 음성과 생성된 음성을 거의 판별하지 못했습니다. 이는 동일한 내용의 쌍을 일대일로 비교한 결과는 아니지만 문법이나 의미적 적절성과 발화자의 목소리, 억양, 음향 환경 등의 일관성이 실제와 구분하기 어려울 정도로 뛰어나다는 것을 나타냅니다.

### Detecting Synthesized Speech

AudioLM이 생성한 음성을 실제와 구분하기 어렵다면 윤리적 측면에서 잠재적인 위험성이 있을 수 있습니다. 따라서 논문에서는 오디오가 원본 음성인지 프롬프트로부터 이어져서 생성된 AudioLM의 음성인지 분류하는 CNN 기반의 모델을 제시합니다.

이 모델은 테스트 셋에서 98.6%의 정확도를 보여줍니다. 따라서 사람이 청각적으로 구분하기는 힘들어도 이러한 모델을 학습시켰을 때 구분되는 특징이 존재한다는 것을 알 수 있습니다.

### Piano Continuation
피아노 데이터셋에 대한 실험 결과도 마찬가지로 높은 오디오 품질과 멜로디나 시간적 구조 측면의 일관성을 보여줍니다. 이에 대한 주관적 평가로는 동일한 프롬프트에 대해 AudioLM 전체와 acoustic token만 사용했을 때 생성된 샘플을 하나의 쌍으로 들려주고 더 선호하는 쪽을 고르도록 했습니다.

평가 결과 83.3%의 참가자가 AudioLM의 생성 샘플을 더 선호하여 음악 생성에서도 두 토큰을 동시에 사용하는 것이 효과적이라는 것을 보여줍니다.

<br><br>

## Reference

[Zalán Borsos, Raphaël Marinier, Damien Vincent, Eugene Kharitonov, Olivier Pietquin, Matt Sharifi, Dominik Roblek, Olivier Teboul, David Grangier, Marco Tagliasacchi and Neil Zeghidour. AudioLM: A Language Modeling Approach to Audio Generation. In TASLP, 2023.](https://ieeexplore.ieee.org/abstract/document/10158503)
