---
layout: post
title: "[논문 리뷰] wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations"
image: https://i.ibb.co/8NN3Ly3/thumbnail.png
date: 2024-03-01
tags: 
categories: paper-review
use_math: true
---

<br><br>

## 논문 개요

<!-- excerpt-start -->

음성 처리와 관련된 문제들을 해결하는 머신러닝 모델을 학습할 때, 레이블이 부여된 많은 양의 학습 데이터를 이용하는 것은 아주 효과적입니다. 하지만 레이블링이 된 많은 양의 데이터를 구하는 것이 항상 쉽지는 않습니다. 예를 들어 음성 인식(speech recognition) 태스크는 각각의 음성을 받아 적은 텍스트를 레이블로 이용하는데 이에 대한 모델을 학습시키려면 최소 수천 시간 이상의 레이블이 필요하고 세상에 존재하는 언어들의 종류도 너무 많습니다.

wav2vec 2.0은 이러한 문제에 대한 대안으로 레이블이 없는 데이터에 대해 먼저 음성 신호의 표현(representation)을 자기지도 학습(self-supervised learning)으로 사전학습(pretrain)시키고 적은 양의 레이블링 된 데이터에 대해 파인튜닝(fine-tuning)할 수 있는 모델입니다. 이러한 표현 사전학습 모델은 일반적으로 여러 가지 다운스트림(downstream) 태스크에 적용할 수 있도록 설계된 경우가 많지만, 이 논문에서는 주로 음성을 텍스트로 변환하는 음성 인식 태스크를 위한 표현 학습에 초점을 맞추고 있습니다.

모델 학습에 사용되는 핵심 기법들은 마스킹(masking)을 이용한 대조 학습(contrastive)과 추출된 표현의 양자화(quantization)입니다. 비슷한 기법을 사용한 이전 연구들과 가장 차별화되는 부분은 양자화된 표현을 문맥(context) 학습을 위한 입력이 아닌 대조 학습을 위한 타겟으로만 사용한다는 점입니다.

<br><br>

## Model Architecture

wav2vec 2.0 모델의 학습 목표는 음성 신호를 입력으로 받아 자기지도 학습으로 이후 태스크에 도움이 될 수 있는 표현들을 추출하는 것입니다. 따라서 여기에서 제안하는 모델 구조는 표현을 추출하는 일종의 인코더 부분에 해당하고, 이후에 수행할 태스크에 따라 여러 가지 적절한 분류기(classifier)나 디코더 등을 결합하여 사용할 수 있습니다.

모델은 크게 특징 인코더(feature encoder), 트랜스포머, 양자화 모듈(quantization module)로 구성되어 있습니다. 특징 인코더 $\small f : \mathcal{X} \rightarrow \mathcal{Z}$ 는 오디오 신호 $\small \mathcal{X}$ 를 입력으로 받아 $\small T$ 타임스텝의 잠재(latent) 표현 $\small \mathbf{z}\_1, \cdots, \mathbf{z}\_T$ 를 출력합니다. 이 잠재 표현들은 트랜스포머 $\small g : \mathcal{Z} \rightarrow \mathcal{C}$ 에 들어가서 전체 시퀀스의 문맥 정보를 반영한 표현 $\small \mathbf{c}\_1, \ldots, \mathbf{c}\_T$ 를 만듭니다.

특징 인코더의 출력은 양자화 모듈 $\small \mathcal{Z} \rightarrow \mathcal{Q}$ 을 통해 $\small \mathbf{q}\_t$ 로 양자화됩니다. 이 양자화된 벡터는 트랜스포머의 입력으로 들어가지 않고 대조 학습의 타겟으로 사용됩니다. 전체 구조는 아래 그림에 요약되어 있습니다.

<p align="center">
<img src="https://i.ibb.co/QJ3zT0t/architecture.png" alt="architecture" border="0">
</p>

### Feature Encoder

특징 인코더는 1D 컨볼루션 층, 레이어 정규화(layer normalization), GELU 활성화 함수로 이루어진 블록 7개로 구성되어 있습니다. 각 블록의 채널 개수는 512, 스트라이드는 (5, 2, 2, 2, 2, 2, 2), 커널 크기는 (10, 3, 3, 3, 3, 2, 2)입니다. 따라서 수용 영역(receptive field)은 400 샘플이 되고 이는 16 kHz의 오디오 신호에 대해 25 ms에 해당합니다.

오디오 신호 입력은 인코더에 들어가기 전에 평균 0과 표준편차 1을 갖도록 정규화(normalize) 됩니다.

### Contextualized Representation with Transformers

트랜스포머는 [(Ashish Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) 기본적인 구조를 유지하면서 실험에 따라 블록 수, 어텐션 헤드 수 등의 하이퍼파라미터를 다르게 설정하여 모델 크기를 결정합니다.

$\small \text{BASE}$ 모델은 블록 12개, 모델 차원 768, FFN 차원 3072, 그리고 어텐션 헤드 8개를 사용합니다. $\small \text{LARGE}$ 모델은 블록 24개, 모델 차원 1024, FFN 차원 4096, 그리고 어텐션 헤드 16개를 사용합니다. 이외에도 드롭아웃(dropout) 비율이나 러닝레이트(learning rate) 등 세부적인 하이퍼파라미터들이 조금씩 다릅니다. 구체적인 설정은 논문에 나와 있습니다.

그리고 고정된 위치 임베딩(positional embedding) 대신 컨볼루션 층을 이용한 상대 위치 임베딩을 합니다. 컨볼루션 층 다음에는 GELU와 레이어 정규화를 수행합니다.

### Quantization Module

양자화 모듈을 통해 특징 인코더의 출력 $\small \mathbf{z}$ 는 유한한 개수의 표현으로 양자화됩니다. 양자화에는 $\small G$ 개의 코드북(codebook)을 사용하고 각각의 코드북에는 길이가 $\small d/G$ 인 $\small V$ 개의 벡터가 있습니다. 각각의 코드북에서 하나씩 선택된 벡터들 $\small e\_1, \ldots, e\_G$ 은 같이 연결(concatenate)된 뒤에 선형 변환 $\small \mathbb{R}^d \rightarrow \mathbb{R}^f$ 을 거쳐 $\small \mathbf{q} \in \mathbb{R}^f$ 가 됩니다.

양자화 과정을 미분 가능하도록 만들기 위해서는 straight-through estimator와 Gumbel softmax를 [(Eric Jang et al., 2016)](https://arxiv.org/abs/1611.01144) 사용합니다. Straight-through estimator는 역전파(backpropagation)할 때 양자화되기 전 벡터의 그래디언트를 그대로 사용하는 것입니다.

Gumbel softmax는 다음의 식과 같이 로짓(logit) $\small \mathbf{l} \in \mathbb{R}^{G \times V}$ 에 무작위 샘플링된 노이즈를 더해서 소프트맥스를 취함으로써 미분 가능하면서 확률적인 다양한 선택이 가능하도록 도와줍니다.

<br>
\begin{equation}
p\_{g, v} = \frac{\exp (l\_{g,v} + n\_v) / \tau}{\sum\_{k=1}^V \exp (l\_{g, k} + n\_k) / \tau}
\end{equation}
<br>

여기서 $\small \tau$ 는 확률 분포를 조절해주는 온도(temperature)이고 $\small n = - \log (- \log u)$ 이며 $\small u$ 는 균일 분포(uniform distribution) $\small \mathcal{U}(0, 1)$ 에서 샘플링한 값입니다. 순전파(forward pass)에서는 $\small i = \text{argmax}\_j p\_{g, j}$ 에 해당하는 코드북 벡터가 선택되고 역전파 시에는 straight-through estimator에 따라 선택된 코드북 벡터가 아닌 Gumbel softmax의 그래디언트가 사용됩니다.

<br><br>

## Training

모델은 먼저 레이블이 없는 데이터를 이용하여 사전학습한 뒤 레이블이 있는 데이터에 대해 파인튜닝 됩니다.

### Masking

모델의 사전학습에서는 BERT와 [(Jacob Devlin et al., 2018)](https://arxiv.org/abs/1810.04805) 비슷하게 마스킹을 사용합니다. 마스킹은 특징 인코더의 출력 일부에 적용되어 트랜스포머로 들어가고, 양자화 모듈의 입력에는 마스킹을 하지 않습니다.

먼저 확률 $\small p$ 로 마스킹이 시작되는 타임스텝들을 샘플링합니다. 샘플링된 타임스텝들로부터 길이 $\small M$에 해당하는 연속된 타임스텝 구간(span)들이 마스킹되며 각각의 구간은 겹칠 수 있습니다. 이 구간의 잠재 표현들은 모두 동일한 학습 가능한 마스크 벡터로 교체됩니다.

마스킹 구간들이 서로 겹칠 수 있기 때문에 최종적으로 마스킹 된 구간들의 길이는 일정하지 않습니다. 아래의 그림은 15초 짜리 샘플에 대해 $\small p=0.065$ 와 $\small M=10$ 으로 적용한 마스킹 구간 길이의 분포를 보여줍니다.

<p align="center">
<img src="https://i.ibb.co/XzsQFc3/masking-spans.png" alt="masking-spans" border="0">
</p>

### Objective

사전학습 시 모델은 마스킹을 활용한 대조 손실(contrastive loss) $\small \mathcal{L}\_m$ 과 코드북의 다양성 손실(diversity loss) $\small \mathcal{L}\_d$ 로 학습됩니다. 다음 식과 같이 두 손실 함수 사이의 비율 $\small \alpha$ 는 하이퍼파라미터로 설정됩니다.

<br>
\begin{equation}
\mathcal{L} = \mathcal{L}_m + \alpha \mathcal{L}\_d
\end{equation}
<br>

마스킹된 타임스텝의 트랜스포머 출력 $\small \mathbf{c}\_t$ 가 주어졌을 때 모델은 $\small K+1$ 개의 양자화 벡터 후보들 $\small \tilde{\mathbf{q}} \in \mathbf{Q}\_t$ 내에서 실제 양자화 벡터 $\small \mathbf{q}\_t$ 를 식별해야 합니다. $\small \mathbf{Q}\_t$ 는 $\small \mathbf{q}\_t$ 와 $\small K$ 개의 오답(distractor)들로 이루어져 있습니다. 대조 손실 함수는 아래와 같이 정의됩니다.

<br>
\begin{equation}
\mathcal{L}\_m = - \log \frac{\exp (sim(\mathbf{c}\_t, \mathbf{q}\_t) / \kappa)}{\sum\_{\tilde{\mathbf{q}} \sim \mathbf{Q}\_t} \exp (sim(\mathbf{c}\_t, \tilde{\mathbf{q}}) / \kappa)}
\end{equation}
<br>

이때 $\small sim(\mathbf{a}, \mathbf{b}) = \mathbf{a}^T\mathbf{b} / \lVert \mathbf{a} \rVert \lVert \mathbf{b} \rVert$ 는 코사인 유사도(cosine similarity)입니다.

이러한 대조 학습이 코드북을 타겟으로 이용하기 때문에 양자화된 벡터의 다양성을 증가시키는 것이 필요합니다. 구체적으로, $\small G$ 개의 각 코드북 내에 있는 $\small V$ 개의 양자화된 벡터들을 골고루 사용하기 위해 각 코드북의 평균 소프트맥스 분포 $\small \bar{p}\_g$ 의 엔트로피를 최대화합니다. 이 소프트맥스 분포에는 Gumbel 노이즈나 온도를 적용하지 않습니다.

<br>
\begin{equation}
\mathcal{L}\_d = \frac{1}{GV} \sum\_{g=1}^{G} - H(\bar{p}\_g) = \frac{1}{GV} \sum\_{g=1}^{G} \sum\_{v=1}^{V} \bar{p}\_{g,v} \log \bar{p}\_{g,v}
\end{equation}
<br>

### Fine-tuning

사전학습이 끝난 모델은 음성 인식 태스크를 수행하도록 파인튜닝 됩니다. 파인튜닝은 각각의 타임스텝을 문자 토큰 클래스로 분류할 수 있도록 CTC 손실을 [(Alex Graves et al., 2006)](https://dl.acm.org/doi/abs/10.1145/1143844.1143891) 최소화하는 것으로 학습됩니다. CTC에 대한 자세한 설명은 Distill에 출간된 논문으로 [(Awni Hannun, 2017)](https://distill.pub/2017/ctc/) 대체합니다.

문자 토큰 분류를 위해서 문맥 정보를 추출하는 트랜스포머 위에 단순하게 선형 프로젝션(linear projection) 층만 추가하여 문자들에 대한 로짓을 출력할 수도 있고 별도의 언어 모델(language model) 디코더를 추가할 수도 있습니다. 언어 모델은 Librispeech LM corpus에 [(Vassil Panayotov et al., 2015)](https://ieeexplore.ieee.org/abstract/document/7178964/) 대해 학습된 4-gram 모델이나 트랜스포머를 사용하여 실험했습니다.

과적합(overfitting)을 방지하기 위해 파인튜닝 과정에서 특징 인코더 출력의 일부 타임스텝과 채널은 마스킹을 해줍니다. 마스킹 방법은 사전학습에서 한 것과 비슷하게 시작 타임스텝과 채널을 임의로 샘플링하고 그로부터 연속된 몇 개를 마스킹 구간으로 사용하는 것입니다. 마스킹 구간들은 서로 겹칠 수 있고 타임스텝 마스킹은 사전학습에서 사용한 것과 동일한 임베딩 벡터로, 채널 마스킹은 0으로 대체합니다.

<br><br>

## 실험

레이블이 없는 데이터는 960시간 짜리 Librispeech(LS-960) 또는 53.2시간 짜리 LibriVox(LV-60k)의 오디오 데이터를 사용합니다. 파인튜닝에 사용되는 레이블링된 데이터는 텍스트가 포함된 Librispeech(LS-960)와 이로부터 각각 100시간, 10시간, 1시간, 10분 분량으로 추출된 부분 데이터셋을 사용합니다.

또한 파인튜닝에는 음소(phoneme) 분류를 위한 TIMIT 데이터셋도 [(John S. Garofolo et al., 1993)](https://catalog.ldc.upenn.edu/LDC93S1) 사용합니다. 이 데이터셋에는 5시간 분량의 음성이 세부적인 음소 레이블과 함께 존재합니다.

평가 지표로는 각 데이터셋에서 표준으로 사용하는 train/development/test 분할(split)에 대한 word error rate(WER)을 사용합니다. Librispeech 데이터셋은 추가로 음성 인식 난이도에 따라 clean/other 두 종류로 분할됩니다.

### Low-Resource Labeled Data Evaluation

먼저 레이블이 없는 데이터에 대해 사전학습된 표현이 적은 양의 제한된 레이블 데이터에 대해 효과적으로 사용될 수 있는지 평가합니다. 그 결과는 아래 표에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/qxdYQFv/low-resource.png" alt="low-resource" border="0">
</p>

다른 모델들과 비교하였을 때 상당히 우수한 성능을 보여주고 특히 10분 짜리 데이터셋으로만 파인튜닝 했을 때에도 높은 성능을 나타냅니다. 비교 모델들 중 Discrete BERT는 [(Alexei Baevski et al., 2019)](https://arxiv.org/abs/1911.03912) 오디오 신호의 양자화를 먼저 하고 이에 대해 문맥 정보를 추출하는 모델입니다. 양자화를 따로 분리하여 대조 학습 타겟으로만 사용하는 wav2vec 2.0의 방법이 성능에 큰 영향을 미친다는 것을 보여줍니다.

Noisy student는 [(Daniel S. Park et al., 2020)](https://arxiv.org/abs/2005.09629) 레이블링, 필터링, 재학습을 여러 번 반복하는 방식으로 100시간 짜리 Librispeech에 대해 기존에 가장 높은 성능을 나타내던 모델입니다. wav2vec 2.0은 이보다 간단하면서도 더 좋은 성능을 보여줍니다.

파인튜닝할 때 언어 모델을 추가로 사용하지 않는 경우에는 성능이 훨씬 떨어집니다. 아래 표는 그 결과를 보여줍니다.

<p align="center">
<img src="https://i.ibb.co/XyfWC5t/language-model.png" alt="language-model" border="0">
</p>

### High-Resource Labeled Data Evaluation

다음으로는 많은 양의 레이블링 된 데이터셋을 사용 가능할 때의 성능을 평가합니다. 그 결과는 아래 표에 정리되어 있습니다.

<p align="center">
<img src="https://i.ibb.co/9tk8SJy/high-resource.png" alt="high-resource" border="0">
</p>

역시 wav2vec 2.0이 가장 우수한 성능을 보여줍니다. $\small \text{LARGE}$ \- from scratch는 처음부터 지도 학습을 한 것인데 이렇게 많은 양의 레이블링 된 데이터셋을 이용할 수 있는 경우에도 자기지도 학습으로 사전학습 하는 것이 성능에 도움을 준다는 것을 보여줍니다.

### Phoneme Recognition on TIMIT

TIMIT 데이터셋을 사용하여 음소 인식 태스크를 수행한 결과는 아래 표에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/MfvHsMd/timit-result.png" alt="timit-result" border="0">
</p>

이 실험에서는 phoneme error rate(PER)를 평가 기준으로 사용합니다. wav2vec 2.0이 가장 우수한 성능을 보여줍니다. 또한 아래 그림은 양자화된 잠재 표현이 각각의 음소를 어떤 식으로 학습하는지 분석하기 위해 TIMIT 학습 데이터셋에 대한 조건부 확률 $\small P(phoneme \vert \mathbf{q}\_t)$ 를 시각화한 것입니다.

<p align="center">
<img src="https://i.ibb.co/zndZjvL/phoneme-probability.png" alt="phoneme-probability" border="0">
</p>

x축은 양자화된 잠재 벡터들, y축은 음소 클래스를 나타냅니다. 많은 양자화된 벡터들이 각각 특정한 음소에 대응되도록 학습되는 것을 볼 수 있습니다.

### Ablation

비슷한 모델 구조와 학습 방법을 사용한 기존 연구들과 wav2vec 2.0의 가장 큰 차이 중 하나는 특징 인코더에서 양자화된 잠재 표현을 이후 트랜스포머 입력으로 전달하지 않고 대조 학습 타겟으로만 사용한다는 것입니다. 이와 관련된 제거(ablation) 연구로 트랜스포머 입력과 대조 학습 타겟을 각각 연속적인(continuous) 벡터와 양자화된 벡터 중에서 선택하여 가능한 네 가지 조합을 모두 실험했습니다. 그 결과는 아래 표에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/fSXc9XZ/quantization-ablation.png" alt="quantization-ablation" border="0">
</p>

연속적인 벡터 입력을 사용하는 것이 성능에 꽤 큰 영향을 주는 것을 알 수 있습니다. 또한 아래 표와 같이 하이퍼파라미터 선택을 위한 다양한 실험도 진행했습니다.

<p align="center">
<img src="https://i.ibb.co/0XXFsGp/hyperparameter-ablation.png" alt="hyperparameter-ablation" border="0">
</p>

<br><br>

## Reference

[Alexei Baevski, Henry Zhou, Abdelrahman Mohamed and Michael Auli. wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations. In NeurIPS, 2020.](https://proceedings.neurips.cc/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html)

[Official Source Code of wav2vec 2.0](https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md)

[Hugging Face Implementation of wav2vec 2.0](https://huggingface.co/docs/transformers/main/model_doc/wav2vec2)
