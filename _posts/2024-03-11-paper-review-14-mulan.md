---
layout: post
title: "[논문 리뷰] MuLan: A Joint Embedding of Music Audio and Natural Language"
image: https://i.ibb.co/qF9fdLX/thumbnail.png
date: 2024-03-11
tags: 
categories: paper-review
use_math: true
---

## 논문 개요

<!-- excerpt-start -->

MuLan은 음악과 텍스트에 대한 공동 임베딩(joint embedding)을 학습하는 모델입니다. 음악 오디오와 그 음악에 대한 태그나 문장으로 된 설명으로 이루어진 데이터셋으로부터 학습된 모델은 같은 임베딩 공간을 공유하는 음악과 텍스트의 표현을 추출할 수 있습니다. 이러한 공동 임베딩은 음악 도메인에서 언어를 활용한 다양한 태스크에 활용됩니다.

<br><br>

## Learning Framework

MuLan 모델은 각각 오디오와 텍스트 입력을 위한 두 개의 분리된 임베딩 신경망으로 구성되어 있습니다. 이 신경망들은 서로 파라미터를 공유하지 않지만 동일한 차원 $\small d$ 를 갖는 임베딩 공간을 만듭니다. 그 구조는 아래 그림에 요약되어 있습니다.

<p align="center">
<img src="https://i.ibb.co/v1szw5g/architecture.png" alt="architecture" border="0">
</p>

오디오 임베딩 신경망 $\small f : \mathbb{R}^{F \times T} \rightarrow \mathbb{R}^d$ 는 $\small F$ 개의 멜 스케일 채널과 $\small T$ 개의 프레임으로 이루어진 로그 멜 스펙트로그램 윈도우를 입력으로 받습니다. 텍스트 임베딩 신경망 $\small g : \mathcal{A}^n \rightarrow \mathbb{R}^d$ 는 $\small \mathcal{A}$ 개의 토큰 어휘(vocabulary)에 대한 텍스트 토큰 시퀀스를 입력으로 받습니다. 토큰 시퀀스의 길이는 $\small n$ 이 되도록 $\small null$로 패딩하거나 잘라냅니다.

학습 데이터셋은 오디오-텍스트 쌍으로 준비됩니다. 각각의 녹음에 대해서 임의의 스펙트로그램 윈도우 $\small \mathbf{x}^{(i)} \in \mathbb{R}^{F \times T}$ 와 텍스트 $\small \mathbf{t}^{(i)} \in \mathcal{A}^n$ 을 샘플링하고 $\small B$ 개의 타겟 오디오-텍스트 쌍으로 이루어진 미니배치 $\small \\{(\mathbf{x}^{(i)}, \mathbf{t}^{(i)})\\}_{i=1}^B$ 를 만듭니다. 학습은 batch-wise Contrastive Multivew Coding 손실 함수를 통해 이루어집니다. 그 식은 다음과 같습니다.

<br>
\begin{equation}
\sum\_{i=1}^B -\log \left[ \frac{h[ f(\mathbf{x}^{(i)}), g(\mathbf{t}^{(i)}) ]}{\sum\_{j \neq i} h[f(\mathbf{x}^{(i)}), g(\mathbf{t}^{(j)})] + h[ f(\mathbf{x}^{(j)}), g(\mathbf{t}^{(i)}) ]} \right]
\end{equation}
<br>

여기서 $\small \mathbf{a}, \mathbf{b} \in \mathbb{R}^d$ 에 대한 $\small h[\mathbf{a}, \mathbf{b}] = \exp(\mathbf{a}^T \mathbf{b} / \tau)$ 는 유사도를 계산하는 크리틱 함수이고 $\small \tau \in (0, 1]$ 는 학습 가능한 온도(temperature) 하이퍼파라미터입니다. 크리틱의 목표는 타겟 오디오-텍스트 쌍에 높은 양의 값을 부여하고 배치 안에서 타겟이 아닌 쌍에 대해서는 0에 가까운 작은 값을 부여하는 것입니다.

<br><br>

## Audio Embedding Network

오디오 임베딩 신경망은 기반이 되는 구조에 따라 두 가지 버전이 있습니다. 각각 Resnet-50과 Audio Spectrogram Transformer(AST)입니다.

### Resnet-50

Resnet-50은 오디오 분류에 대한 이전 연구에서 [(Shawn Hershey et al., 2017)](https://ieeexplore.ieee.org/abstract/document/7952132) 실험적으로 검증된 구조를 사용합니다. 먼저 $\small F=64$ 멜 채널, $\small 25 \, \text{ms}$ Hanning 윈도우, $\small 10 \, \text{ms}$ 스텝 사이즈를 적용하여 그레이스케일의 로그 멜 스펙트로그램을 만듭니다. 신경망의 입력은 스펙트로그램에서 임의로 선택된 10초 짜리 윈도우로 $\small (F=64) \times (T=1000)$ 스펙트로그램 패치의 형태를 갖습니다.

학습 시에는 입력을 신경망에 넣기 전에 SpecAugment를 [(Daniel S. Park et al., 2019)](https://arxiv.org/abs/1904.08779) 적용합니다. 마지막에는 시간과 멜 채널에 평균 풀링(mean pooling)이 적용되고 완전연결층(fully connected layer)을 통해 만들어진 $\small d=128$ 차원의 임베딩은 $\small l_2$ 정규화됩니다.

신경망은 마지막 완전연결층만 제외하고 AudioSet [(Jort F. Gemmeke et al., 2017)](https://ieeexplore.ieee.org/abstract/document/7952261) 데이터셋의 527개 클래스에 대한 로지스틱 회귀(logistic regression)로 사전학습됩니다. 사전학습이 끝난 뒤에는 마지막 분류 층(classifier layer)을 제외하고 MuLan의 batch-wise Contrastive Multivie Coding 손실로 파인튜닝됩니다.

### Audio Spectrogram Transformer (AST)

AST는 [(Yuan Gong et al., 2021)](https://arxiv.org/abs/2104.01778) 오디오 분류에서 높은 성능을 보여주는 ViT를 기반으로 한 모델입니다. 먼저 $\small (F = 128) \times (T = 1000)$ 의 로그 멜 스펙트로그램 윈도우는 $\small 16 \times 16$ 의 패치로 나눠져서 선형 변환을 통해 평평해진(flattened) 토큰 시퀀스를 만듭니다.

Resnet-50과 마찬가지로 학습 시에는 SpecAugment가 적용됩니다. 많은 트랜스포머 기반의 언어 모델들을 따라서 학습가능한 위치 임베딩(positional embedding)이 더해지고 토큰 시퀀스의 맨 앞에는 $\small \text{[CLS]}$ 토큰이 추가됩니다. 마지막에는 $\small \text{[CLS]}$ 토큰 위치의 768 차원 인코딩이 완전연결층을 통해 $\small d=128$ 차원으로 변환되고 $\small l_2$ 정규화가 적용됩니다.

마지막 완전연결층을 제외한 신경망은 AST 원본 논문에서 공개한 체크포인트로부터 학습됩니다.

<br><br>

## Text Embedding Network

텍스트 임베딩 신경망으로는 BERT-base-uncased 모델을 [(Jacob Devlin et al., 2019)](https://arxiv.org/pdf/1810.04805.pdf) 사용합니다. 텍스트 입력은 BERT wordpiece tokenizer를 이용하여 $\small n=512$ 의 토큰 시퀀스로 만들어집니다. $\small \text{[CLS]}$ 토큰의 최종 임베딩은 $\small d=128$ 차원으로 선형 변환되고 $\small l_2$ 정규화됩니다.

텍스트 임베딩 신경망도 공개되어 있는 체크포인트로부터 학습됩니다.

<br><br>

## Training Dataset Mining

MuLan 임베딩 모델을 학습시키기 위해서는 많은 양의 오디오-텍스트 쌍이 필요합니다. 이를 위해 인터넷에 있는 5천만 개의 뮤직비디오와 AudioSet으로 학습 데이터셋을 만듭니다.

뮤직비디오에서는 30초 짜리 클립을 추출하고 시중에 존재하는 음악 오디오 판별기를 통해 전체 길이의 절반 이상 음악이 없는 클립은 제외합니다. 이 필터링 과정 뒤에는 약 4,400만 개 클립의 37만 시간 짜리 오디오가 남습니다.

각각의 뮤직비디오에 대한 텍스트는 세 가지 형태를 사용합니다. Short-form (SF) 텍스트는 비디오 제목과 태그를 포함합니다. Long-form (LF) 텍스트는 비디오 설명과 코멘트를 포함합니다. Playlists (PL) 텍스트는 데이터셋의 뮤직비디오를 포함하고 있는 17억 개의 플레이리스트 제목입니다. 아래 표는 텍스트 데이터셋의 예시를 보여줍니다.

<p align="center">
<img src="https://i.ibb.co/kGGzRmB/text-examples.png" alt="text-examples" border="0">
</p>

이러한 텍스트들은 항상 음악과 관련된 내용만을 포함하고 있지는 않기 때문에 상당히 노이즈가 많은 데이터입니다. 따라서 SF와 LF 데이터에는 필터링을 적용하는 경우도 테스트합니다.

LF 필터링을 위해서는 먼저 사전학습된 BERT 모델을 음악에 관련되었는지 사람이 레이블링한 700개 문장의 분류 태스크에 대해 파인튜닝합니다. 이 분류 모델을 LF 텍스트에 적용하여 음악과 관련된 텍스트만 걸러냅니다.

SF 텍스트에는 룰베이스의 필터링 방법을 적용합니다. 아래 표는 텍스트 데이터에 이러한 필터링을 적용한 전후의 크기를 보여줍니다. 데이터 크기는 전체 토큰 개수(Tokens)와 각각의 비디오에 대한 평균 텍스트 수(APV)로 표현되어 있습니다.

<p align="center">
<img src="https://i.ibb.co/kXphGrH/filtering.png" alt="filtering" border="0">
</p>

추가로 AudioSet을 오디오-텍스트 쌍으로 만든 데이터셋도 사용하는데 이것을 ASET이라고 명명합니다. 텍스트로는 527개의 레이블 이름을 사용합니다. 이렇게 만들어진 ASET 전체 데이터는 10초 짜리 오디오 클립 2백만 개와 각 클립에 평균 1.8개씩 달려 있는 레이블 텍스트입니다.

각각의 데이터 형태가 불균형한 크기를 가지고 있기 때문에 미니배치를 만들 때 SF:LF:PL:ASET의 비율을 2:2:1:1로 고정합니다. 이 비율은 별도의 최적화 과정 없이 임의로 정한 것입니다.

<br><br>

## 실험

실험에서는 MuLan의 Resnet-50 (M-Resnet-50)과 AST (M-AST) 오디오 신경망을 모두 테스트하고 텍스트 신경망은 BERT-base-uncased로 동일합니다. 실험 결과를 보면 M-Resnet-50과 M-AST의 성능이 비슷하기 때문에 텍스트 제거(ablation) 실험에는 M-Resnet-50만 사용합니다.

### Zero-shot Music Tagging

이 실험은 처음 보는 음악 클립이 주어졌을 때 여러 개의 텍스트 태그 후보 중에서 정답 태그를 맞추는 것입니다. 음악 클립의 오디오 임베딩과 타겟 태그의 텍스트 임베딩 사이에서 계산된 코사인 유사도(cosine similarity)를 점수로 사용하여 가장 높은 점수의 태그로 분류합니다.

평가 데이터셋으로는 MagnaTagATune (MTAT)과 [(Edith Law and Luis von Ahn, 2009)](https://dl.acm.org/doi/abs/10.1145/1518701.1518881) AudioSet을 사용합니다. MTAT은 전체 188개의 태그를 다 사용하는 All-188과 50개를 추린 Top-50의 두 가지 종류를 사용합니다. AudioSet은 25개의 장르 태그를 분류하는 Gen-25와 141개 태그를 분류하는 Mu-141을 사용합니다.

사실 AudioSet이 학습 데이터셋에 포함되어 있고 MTAT의 레이블도 일부 AudioSet과 겹치기 때문에 엄밀히 말하면 제로샷 평가는 아닙니다. 하지만 MuLan 학습 시에 엄청나게 많은 자유로운 형식의 텍스트들이 사용되기 때문에 AudioSet 레이블의 영향력이 희석됩니다. 따라서 이 실험이 모델의 일반화 능력을 평가할 수 있다고 간주합니다.

<p align="center">
<img src="https://i.ibb.co/YpTRQwb/music-tagging.png" alt="music-tagging" border="0">
</p>

위의 표는 AUC-ROC로 평가된 실험 결과를 보여줍니다. 먼저 M-AST와 M-Resnet-50 간의 성능 차이는 거의 없습니다. 전반적으로 MTAT이 AudioSet보다 점수가 낮은데 MTAT 태그에 포함된 "not rock" 같은 부정문을 제대로 모델링 하지 못하는 것이 BERT의 고질적 문제이고 "weird"나 "beats" 같은 다양한 의미로 해석될 수 있는 단어들 때문에 MTAT의 난이도가 더 높습니다.

(b)의 텍스트 제거 실험 결과를 보면 당연하게도 ASET만 사용했을 때 AudioSet의 점수가 가장 높습니다. 그 외에는 앞으로 볼 다른 실험 결과들을 포함해서 대체로 텍스트 데이터셋 종류를 많이 포함할수록 성능이 좋아집니다. 그리고 SF와 LF에 대해 필터링을 하지 않았을 때 오히려 점수가 더 높은 경향이 있는데 필터링 과정에서 음악과 관련이 없는 텍스트가 제거되긴 하지만 그에 못지 않게 의미를 학습하기에 유용한 정보들도 손실되기 때문일 가능성이 있습니다.

### Transfer Learning with Linear Probes

이 실험은 프리징된 128차원의 오디오 임베딩 신경망 위에 로지스틱 회귀 층을 추가한 뒤 다운스트림 태스크에 대해 학습시켜 전이 학습 (transfer learning) 성능을 확인하는 것입니다. 다운스트림 태스크는 마찬가지로 MagnaTagATune과 AudioSet의 태그 분류입니다. 그 결과는 아래 표의 (c)에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/Jkr7rS3/transfer-learning.png" alt="transfer-learning" border="0">
</p>

MuLan이 다른 베이스라인들보다 우수한 성능을 나타내는 것을 볼 수 있습니다. 또한 (d)는 엔드투엔드로 학습하는 베이스라인 모델들의 결과인데 MuLan이 이들과도 비슷하거나 능가하는 성능을 나타냅니다.

### Music Retrieval from Text Queries

MuLan의 공동 임베딩을 이용하여 텍스트 쿼리가 주어졌을 때 음악 목록 중에서 가장 가까운 음악을 검색하는 태스크를 수행할 수도 있습니다. 실험에 사용하는 음악 목록은 전문가가 선정한 7000개의 플레이리스트로 학습에 사용한 플레이리스트와 겹치지 않도록 준비합니다.

각각의 플레이리스트는 10-100개의 곡들을 포함하고 있고 플레이리스트 제목은 장르, 분위기, 구성 등을 포함하는 짧은 구문으로 되어 있습니다. 플레이리스트 설명은 한 개 또는 몇 개의 완전한 문장으로 이루어져 있습니다. 예를 들어 제목은 "Relaxing Korean Pop"이고 설명은 "Lets make your chill mood with a collection of easy-going sounds from Korean artists." 입니다.

평가 시에는 각각 제목과 설명을 쿼리로 한 것을 모두 테스트합니다. 태그 실험과 마찬가지로 임베딩에 대한 코사인 유사도 점수를 이용하고 10만 개의 곡들을 모두 후보로 사용합니다. 실제값 타겟은 해당하는 플레이리스트 안에 포함된 곡들입니다. 아래 표는 AUC-ROC와 mean average precision (mAP)을 지표로 평가한 결과입니다.

<p align="center">
<img src="https://i.ibb.co/NKnY0Xd/text-query.png" alt="text-query" border="0">
</p>

텍스트 제거 실험 결과를 보면 LF를 추가했을 때 성능 향상 폭이 큰 것을 알 수 있습니다. 음악적인 내용과 관련된 자잘한 표현들을 인터넷에 있는 LF 텍스트의 문장들로부터 학습할 수 있다는 것을 보여줍니다.

### Text Triplet Classification

이 실험은 MuLan의 텍스트 임베딩이 음악과 관련된 언어적 지식을 잘 반영하고 있는지 확인하기 위한 테스트입니다. AudioSet과 Playlist 데이터셋에서 $\small \text{(anchor, pos, neg)}$ 쌍들을 만들고 $\small \text{pos}$ 가 $\small \text{neg}$ 보다 $\small \text{anchor}$ 에 더 가까우면 맞다고 간주하는 방법입니다. 각 데이터셋에 대한 텍스트 쌍 예시는 아래 표에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/VM0X1Sc/triplet-examples.png" alt="triplet-examples" border="0">
</p>

평가 지표로는 분류 정확도(accuracy)를 사용합니다. MuLan과 다른 베이스라인 모델들의 평가 결과는 아래 표에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/3sJS4rh/triplet-results.png" alt="triplet-results" border="0">
</p>

MuLan이 전체적으로 다른 모델들에 비해 높은 성능을 보여줍니다. 특히 BERT와 비교해보면 대조 학습 손실을 사용하여 음악 도메인의 텍스트 데이터셋에 대해 학습하는 것이 효과적이라는 것을 알 수 있습니다.

<br><br>

## Reference

[Qingqing Huang, Aren Jansen, Joonseok Lee, Ravi Ganti, Judith Yue Li and Daniel P. W. Ellis. MuLan: A Joint Embedding of Music Audio and Natural Language. In ISMIR, 2022.](https://ismir2022program.ismir.net/poster_150.html)
