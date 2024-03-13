---
layout: post
title: "[논문 리뷰] MusicLM: Generating Music From Text"
image: https://i.ibb.co/3ynZ1sS/thumbnail.png
date: 2024-03-13
tags: 
categories: paper-review
use_math: true
---

## 논문 개요

<!-- excerpt-start -->

MusicLM은 텍스트 입력으로부터 고품질의 음악을 생성하는 모델입니다. 예를 들어 "a calming violin melody backed by a distorted guitar riff"와 같은 텍스트가 입력되면 그 설명에 맞고 몇 분 동안 일관성이 유지되는 음악이 생성될 수 있습니다.

또한 텍스트 프롬프트 입력을 확장시켜 사람이 멜로디를 흥얼거린 것을 같이 넣어줘서 그 멜로디를 따르는 음악 클립을 생성하는 것도 가능합니다. 학습에 사용된 데이터셋은 공개되어 있지 않지만 음악-텍스트 쌍에 초점을 맞춘 평가를 위해 새로 제작한 데이터셋인 MusicCaps도 이 논문을 통해 공개되었습니다.

<br><br>

## Representation and Tokenization of Audio and Text

MusicLM의 오디오 생성은 계층적으로 자기회귀적인(autoregressive) 생성을 하는 AudioLM에 [(Zalán Borsos et al., 2023)](https://ieeexplore.ieee.org/abstract/document/10158503) 기반합니다. 따라서 AudioLM의 두 토큰화 모델인 SoundStream과 [(Neil Zeghidour et al., 2021)](https://ieeexplore.ieee.org/abstract/document/9625818) w2v-BERT도 [(Yu-An Chung et al., 2021)](https://ieeexplore.ieee.org/abstract/document/9688253) 동일하게 사용합니다.

텍스트 조건의 처리에는 MuLan을 [(Qingqing Huang et al., 2022)](https://ismir2022program.ismir.net/poster_150.html) 활용합니다. MuLan의 오디오-텍스트 공동 임베딩(joint embedding)은 오디오 데이터만을 가지고 모델을 학습시킨 뒤 추론 시에 텍스트 임베딩을 사용하는 것을 가능하게 합니다. 이 방식은 방대한 양의 레이블 없는 오디오 학습 데이터셋을 사용할 수 있다는 점에서 매우 효과적입니다.

[SoundStream](https://jeffmung.github.io/2024/02/27/paper-review-10-soundstream/), [w2v-BERT](https://jeffmung.github.io/2024/03/06/paper-review-12-w2vbert/), [AudioLM](https://jeffmung.github.io/2024/03/07/paper-review-13-audiolm/), 그리고 [MuLan](https://jeffmung.github.io/2024/03/11/paper-review-14-mulan/)에 대한 자세한 설명은 각각 링크되어 있는 이전 포스트를 참고하면 좋습니다. 임베딩을 학습하기 위한 세 모델은 독립적으로 사전학습된 뒤 프리징됩니다. 그 구조는 아래 그림에 요약되어 있습니다.

<p align="center">
<img src="https://i.ibb.co/7JCvHCw/architecture.png" alt="architecture" border="0">
</p>

AudioLM의 SoundStream과 w2v-BERT에서 만들어지는 acoustic token과 semantic token은 각각 $\small A$ 와 $\small S$ 라고 명명합니다. 세부적인 하이퍼파라미터 설정은 논문에 나와 있습니다.

MuLan의 오디오 임베딩은 MusicLM의 작동 환경에 맞게 몇 가지를 수정합니다. 먼저 MuLan은 10초 짜리 오디오 입력에 대해 작동하는데 MusicLM은 더 긴 오디오 시퀀스를 처리해야 하므로 1초의 스트라이드로 만들어진 임베딩들의 평균을 사용합니다.

또한 MuLan 임베딩은 연속적인데(continuous) AudioLM의 이산적인(discrete) 토큰과의 일관성을 유지하기 위해 양자화를(quantization) 적용합니다. 위에서 평균으로 얻어진 임베딩에 코드북 크기 1024의 12층 짜리 RVQ를 적용하여 12개의 MuLan 오디오 토큰 $\small M_A$ 를 만듭니다.

추론 시에는 동일한 RVQ를 사용하여 텍스트 프롬프트에서 MuLan 텍스트 토큰 $\small M_T$ 12개를 추출합니다.

<br><br>

## Hierarchical Modeling of Audio Representation

오디오 생성 방식은 AudioLM의 계층적 방식을 비슷하게 따릅니다. 각각의 단계는 분리된 트랜스포머 디코더 모델로 학습됩니다. 그 구조는 아래 그림에 표현되어 있습니다.

<p align="center">
<img src="https://i.ibb.co/Sw6W7Xv/autoregressive-modeling.png" alt="autoregressive-modeling" border="0">
</p>

첫 번째 semantic modeling 단계에서는 MuLan 오디오 토큰으로부터 semantic token을 예측하는 확률 분포 $\small p(S_t \vert S_{< t}, M_A)$ 를 모델링합니다. 두 번째 acoustic modeling 단계에서는 MuLan 오디오 토큰과 semantic token을 조건부로 acoustic token을 예측하는 확률 분포 $\small p(A_t \vert A_{< t}, S, M_A)$ 를 모델링합니다.

AudioLM과 마찬가지로 너무 긴 토큰 시퀀스를 처리하지 않기 위해 acoustic modeling은 coarse와 fine modeling 단계로 나눕니다. Coarse modeling 단계에서는 SoundStream RVQ의 첫 4개 토큰을 사용하고 fine modeling 단계에서는 나머지 8개를 사용합니다.

<br><br>

## Training and Inference

MusicLM의 구성요소들 중 SoundStream과 w2v-BERT는 Free Music Archive (FMA) 데이터셋으로 [(Michaël Defferrard et al., 2017)](https://arxiv.org/abs/1612.01840) 학습시키고, 트랜스포머 디코더들은 5백만 개의 24 kHz 음악 클립을 포함한 총 28만 시간짜리 데이터셋으로 학습시킵니다. 세 토큰 예측 모델링 단계에서는 각각 30, 10, 3초씩 잘라낸 오디오 클립을 타겟으로 사용합니다.

MuLan은 사전학습된 모델을 프리징해서 사용한다고 논문에 나와 있지만 인터넷 상에 사전학습된 MuLan 모델이 공개되어 있지는 않습니다. MuLan 논문에서 설명한 뮤직비디오와 Audioset 데이터셋으로 학습된 모델인지도 명확하게 언급되어 있지는 않습니다.

추론 시에는 MuLan의 공동 임베딩 공간을 활용하여 $\small M_A$ 를 $\small M_T$ 로 바꿔서 semantic token과 acoustic token을 생성합니다. 이때 top-p 샘플링 방법을 사용하고 온도 값은 세 단계에서 각각 1.0, 0.95, 그리고 0.4로 설정합니다.

<br><br>

## Evaluation Dataset

학습된 모델의 평가를 위한 데이터셋은 새로 제작되어 온라인 상에 공개되어 있습니다. 이 데이터셋 이름은 MusicCaps이고 AudioSet의 [(Jort F. Gemmeke et al., 2017)](https://ieeexplore.ieee.org/abstract/document/7952261) 데이터 일부를 가지고 만들었습니다. AudioSet은 음악이 아닌 데이터도 포함하고 있는 반면 MusicCaps는 음악에만 초점을 맞추고 전문가가 작성한 텍스트 설명을 제공합니다.

MusicCaps에는 5500개의 10초 짜리 음악 클립이 포함되어 있습니다. 각각의 클립에는 10명의 프로 뮤지션이 작성한 텍스트 설명이 수반됩니다.

설명은 각 클립마다 평균 4문장으로 이루어진 자유로운 형식의 caption과 장르, 분위기, 템포, 목소리, 악기, 리듬 등을 묘사하는 aspect로 구성되어 있습니다. 각각의 클립은 평균 11개의 aspect를 포함합니다.

MusicCaps는 다양한 장르를 포함하고 있습니다. 이 중 각 장르의 분포를 동일하게 맞춰서 1000개의 클립을 선정한 genre-balanced 데이터셋을 따로 만듭니다. 아래 그림에서 위쪽은 5500개 MusicCaps 전체 클립의 장르 분포이고 아래쪽은 genre-balanced의 장르 분포입니다.

<p align="center">
<img src="https://i.ibb.co/StvZhJr/musiccaps-genre.png" alt="musiccaps-genre" border="0">
</p>

<br><br>

## Metrics

MusicLM의 평가에는 다양한 지표를 사용합니다. 평가 시에 고려하는 두 가지 중요한 측면은 오디오 품질과 텍스트 설명에 대한 부합도입니다.

### Fréchet Audio Distance (FAD)

Fréchet Audio Distance (FAD)는 인간 청각을 반영한 오디오 품질 평가 지표입니다. 낮은 FAD 점수는 실제와 같은 그럴듯한 오디오를 생성했다는 것을 의미합니다. 하지만 이러한 FAD 점수는 조건으로 제공된 텍스트와의 연관성을 반영하지 않습니다.

논문에서는 두 가지 오디오 임베딩 모델에 기반한 FAD를 사용합니다. Trill은 [(Joel Shor et al., 2020)](https://arxiv.org/abs/2002.12764) 음성 데이터로 학습되고 VGGish는 Youtube-8M [(Sami Abu-El-Haija et al., 2016)](https://arxiv.org/abs/1609.08675) 오디오 이벤트 데이터셋으로 학습됩니다. 두 가지 다른 학습 데이터의 사용을 통해 서로 다른 측면의 오디오 품질을 측정할 수 있는 효과를 기대합니다.

### KL Divergence (KLD)

텍스트 설명과 음악 클립 사이에는 many-to-many 관계가 있기 때문에 생성된 음악을 레퍼런스 오디오 파형과 직접적으로 비교하는 것은 불가능합니다. 따라서 생성된 음악이 텍스트 입력 조건에 잘 부합하는지 평가하기 위해 LEAF [(Neil Zeghidour et al., 2021)](https://openreview.net/forum?id=jM76BCb6F9m) 분류기를 사용합니다.

LEAF 분류기는 AudioSet의 다중 레이블 분류 태스크에 대해 학습됩니다. 학습된 LEAF를 사용하여 생성된 음악과 레퍼런스의 클래스 분류 확률 분포를 얻고, 두 분포 사이의 KL 발산을 계산합니다. KL 발산 값이 낮으면 생성된 음악이 레퍼런스와 비슷한 음악적 특징을 갖는다고 볼 수 있습니다.

### MuLan Cycle Consistency (MCC)

MuLan의 공동 임베딩은 음악-텍스트 쌍의 유사도를 측정하는 데 사용될 수 있습니다. MusicCaps의 텍스트 설명들과 생성된 음악의 MuLan 임베딩을 추출하고 양쪽 임베딩 사이의 평균 코사인 유사도를 MuLan Cycle Consistency (MCC)로 정의합니다.

### Qualitative Evaluation

생성된 음악이 텍스트 설명에 잘 부합하는지는 주관적 테스트를 통해서도 평가합니다. 참여자들은 텍스트 설명과 그로부터 생성된 두 가지 서로 다른 모델의 음악을 듣고 선호도를 평가합니다. 텍스트 샘플과 레퍼런스 음악은 genre-balanced 데이터셋으로부터 선택됩니다.

음악 품질에 대한 평가는 FAD에 의해 잘 이루어진다고 간주하고 주관적 평가 참여자들은 오로지 텍스트와의 연관성만 평가에 반영하라고 지시받습니다. 최종적으로는 각 샘플 쌍에서 우세하다고("wins") 평가된 횟수를 집계하여 결과를 종합합니다.

### Training Data Memorization

LLM은 흔히 학습 데이터에서 본 패턴을 외워버릴 가능성이 있습니다. 이러한 현상은 생성된 샘플의 다양성을 저하시키고 개인적인 데이터를 유출하는 등의 문제들을 발생시킬 수 있기 때문에 지양되어야 합니다. 따라서 MusicLM이 음악 조각 패턴을 외우는지 확인하기 위한 방법을 제안합니다.

우선 semantic modeling 단계에 초점을 맞춥니다. 학습 데이터셋에서 임의의 데이터를 샘플링하고 MuLan 오디오 토큰과 처음 몇 초에 해당하는 semantic token을 트랜스포머 디코더 모델에 넣어줍니다. 그 뒤 이어지는 5초에 해당하는 semantic token을 생성하고 데이터셋에 있는 타겟 토큰과 비교합니다.

정량적인 지표로는 여러 샘플들을 레퍼런스와 비교해서 토큰 시퀀스가 서로 유사한 샘플의 비율을 점수로 사용합니다. 두 샘플의 토큰 시퀀스가 유사하다는 것을 판단할 때에는 두 가지 기준을 사용합니다. 먼저 exact matches는 전체 토큰 시퀀스가 정확히 일치하는 경우의 샘플들을 집계합니다. 또한 토큰이 정확하게 일치하지 않더라도 음악적으로 비슷한 특징을 가질 수 있기 때문에 approximate matches의 판별 방법을 정의합니다.

Approximate matches 판별을 위해 생성된 오디오와 타겟으로부터 토큰 어휘 $\small \\{0, \ldots, 1023 \\}$ 에 대한 히스토그램을 계산합니다. 두 히스토그램 분포 사이의 유사도를 측정하기 위해서는 각 토큰 간의 거리를 계산해야 합니다. 이 거리는 w2v-BERT의 semantic token 양자화에 사용하는 k-means 센트로이드들 사이의 거리로 정의합니다.

두 히스토그램 사이의 매칭 비용(matching cost)은 Sinkhorn 알고리즘을 [(Cuturi, 2013)](https://proceedings.neurips.cc/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html) 통해 얻습니다. 매칭 비용이 기준값 0.85보다 낮은 경우 approximate matches에 해당한다고 판단합니다.

<br><br>

## 실험

평가 데이터셋으로는 MusicCaps를 사용하고 비교군으로는 텍스트로부터 음악을 생성하는 최신 모델들인 Mubert와 [(Mubert-Inc, 2022)](https://mubert. com/) Riffusion을 [(Forsgren and Martiros, 2022)](https:// riffusion.com/about) 사용합니다. 실험에 대한 데모 샘플들은 [프로젝트 웹사이트](google-research.github.io/seanet/musiclm/examples)에서 들어볼 수 있습니다.

### Comparison to Baselines

전체적인 실험 결과는 아래 표에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/0ZwZTdY/results.png" alt="results" border="0">
</p>

FAD를 보면 MusicLM이 Mubert와 Riffusion보다 높은 성능을 나타냅니다. 특히 Mubert는 뮤지션과 사운드 디자이너들이 미리 녹음한 소리들을 사용하여 음악을 생성하는데 MusicLM이 이와 비슷한 높은 품질의 음악을 생성한다는 것을 보여줍니다.

KLD와 MCC 역시 MusicLM이 가장 높은 성능을 나타내고 주관적 평가에서도 두 모델에 비하면 MusicLM이 훨씬 우세합니다. MusicLM이 텍스트 설명과 잘 부합하는 음악을 생성한다는 것을 알 수 있습니다. 물론 실제 레퍼런스 음악과 비교했을 때에는 아직 큰 차이가 존재합니다.

MusicLM이 레퍼런스에 비해 못하다고 평가된 경우의 샘플들을 분석해보면 몇 가지 패턴들이 발견됩니다. 먼저 텍스트가 시간적인 순서대로 묘사되어 있거나 "wind, people talking"과 같이 음악과 관련 없는 특징을 지시하는 경우가 있습니다. 또한 지시가 매우 세부적이거나 다섯 개 이상의 악기를 포함할 때, 그리고 부정문이 사용되었을 때에도 낮은 성능을 보여줍니다.

### Importance of Semantic Tokens

Semantic modeling과 acoustic modeling을 분리하는 것의 장점을 분석하기 위해 semantic token 없이 acostic token을 바로 예측하도록 $\small p(A_t \vert A_{< t}, M_A)$ 를 모델링하는 트랜스포머 디코더를 학습시킵니다. 그 결과 FAD는 비슷한 값을 나타내고 KLD와 MCC 값은 더 안좋아졌다고 합니다.

하지만 이 때의 KLD 값이 1.05이고 MCC 값이 0.49인데, semantic token을 사용했을 때의 1.05와 0.51과 비교했을 때 얼마나 유의미한 차이인지는 불분명합니다. 또한 논문에서 직접 샘플을 들어봤을 때에도 저하가 있었다고 하지만 데모 웹사이트에 이 결과는 공개되어 있지 않습니다.

### Information Represented by Audio Tokens

Semantic token과 acoustic token이 담고 있는 정보를 더 분석하기 위해 두 가지 실험을 진행합니다.

첫 번째로는 MuLan 텍스트 토큰과 semantic token을 고정하고 acoustic modeling stage를 여러 번 수행해서 여러 샘플들을 생성합니다. 이러한 샘플들을 들어보면 음악적인 구조는 거의 동일하지만 세부적인 음향적인 특징이 조금씩 달라져서 다양한 음악이 생성됩니다. 예를 들면 메인 멜로디, 리듬, 장르 등은 동일하고 리버브, 악기 등이 달라집니다.

두 번째로는 MuLan 텍스트 토큰을 고정하고 semantic token과 acoustic token을 모두 생성합니다. 이 때에는 다양성이 훨씬 더 증가됩니다. 텍스트 설명과의 부합성은 유지되지만 멜로디나 리듬 등의 요소들도 다양하게 바뀌는 것을 들을 수 있습니다.

### Memorization Analysis

아래 그림은 semantic token 프롬프트의 길이를 0부터 10까지 조정했을 때의 exact matches와 approximate matches의 측정 결과를 보여줍니다.

<p align="center">
<img src="https://i.ibb.co/0DtFcdJ/memorization-result.png" alt="memorization-result" border="0">
</p>

Exact matches의 비율은 항상 0.2% 미만으로 낮고 approximate matches의 비율은 semantic token 프롬프트의 길이가 0일 때에도 그보다 높습니다. 또한 프롬프트의 길이가 증가할 수록 유사도도 증가하는 경향을 보입니다. 하지만 이전 섹션의 실험 결과 샘플을 들어보면 알 수 있듯이 MusicLM이 생성하는 샘플의 다양성은 높은 편이고 심지어 semantic token이 모두 동일할 때에도 acoustic token에 의한 차이가 존재합니다.

### Melody Conditioning

텍스트 설명과 멜로디 프롬프트를 같이 사용하여 음악을 생성하도록 MusicLM을 확장시킬 수도 있습니다. 멜로디 프롬프트는 사람의 노래, 휘파람, 악기 연주 등의 형태로 다양하게 주어질 수 있습니다. 이러한 멜로디 조건을 사용하기 위해서는 다양한 형태의 신호로부터 타겟 멜로디를 포착하는 임베딩 모델을 학습해야 합니다.

이를 위해 다양한 음향적 형태를 가지고 동일한 멜로디를 표현하는 오디오 쌍들로 이루어진 데이터셋을 만듭니다. 이 데이터셋에는 커버, 보컬, 악기 연주 등의 음악 클립과 흥얼거림, 휘파람 등의 추가적인 녹음이 포함되어 있습니다. 이를 사용하여 같은 멜로디를 포함한 두 오디오 클립은 서로 가깝게 임베딩되는 모델을 학습시킵니다.

MuLan의 오디오 임베딩 모델과 비슷하게 ViT 기반의 모델을 사용하고 손실 함수로는 semi-hard triplet loss를 [(Schroff et al., 2015)](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Schroff_FaceNet_A_Unified_2015_CVPR_paper.html) 사용합니다. 학습 시에는 RVQ로 양자화된 멜로디 토큰을 MuLan 오디오 토큰 $\small M_A$ 와 연결하여 사용하고 추론 시에는 멜로디 토큰과 MuLan 텍스트 토큰 $\small M_T$ 를 연결합니다.

### Long Generation and Story Mode

MusicLM의 생성은 자기회귀적이기 때문에 학습 과정에서 사용된 시퀀스 길이보다 더 긴 시퀀스를 생성하는 것이 가능합니다. 예를 들어 semantic modeling은 30초 짜리 시퀀스에 대해 학습됩니다. 더 긴 시퀀스를 생성하기 위해서는 텍스트 조건을 동일하게 유지한 채 이전 15초를 기반으로 다음 15초의 semantic token을 생성하는 것을 15초 간격으로 반복하면 됩니다.

약간의 수정을 거치면 시간에 따라 텍스트 설명이 바뀌면서 오디오를 생성하는 것도 가능합니다. 이러한 방식을 story mode라고 명명합니다. 먼저 여러 텍스트 설명에 대한 $\small M_T$ 를 다 계산한 뒤 15초 마다 텍스트 입력 조건을 바꿔줍니다. 이러한 방식으로 모델은 템포가 유지되면서 부드럽게 텍스트 설명에 따라 바뀌는 음악을 생성할 수 있습니다.

<br><br>

## Reference

[Andrea Agostinelli, Timo I. Denk, Zalán Borsos, Jesse Engel, Mauro Verzetti, Antoine Caillon, Qingqing Huang, Aren Jansen, Adam Roberts, Marco Tagliasacchi et al. MusicLM: Generating Music From Text. arXiv preprint, 2023.](http://arxiv.org/abs/2301.11325)
