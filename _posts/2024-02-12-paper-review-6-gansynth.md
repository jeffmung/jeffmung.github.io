---
layout: post
title: "[논문 리뷰] GANSynth: Adversarial Neural Audio Synthesis"
image: https://i.ibb.co/1XCTHTR/thumbnail.png
date: 2024-02-12
tags: 
categories: Paper-Review
use_math: true
---

<br><br>

## 논문 개요

GANSynth는 구글 AI가 개발한 음악 생성 모델입니다. PGGAN을 [(Tero Karras et al., 2018)](https://arxiv.org/abs/1710.10196) 기반으로 하여 스펙트로그램을 생성하는데, 크기(magnitude) 뿐만 아니라 프레임마다 틀어진 위상(phase) 정렬을 반영할 수 있는 instantaneous frequency를 포함한다는 것이 주요한 아이디어입니다.

이 논문은 NSynth 데이터셋의 [(Jesse Engel et al., 2017)](https://proceedings.mlr.press/v70/engel17a.html) 학습에 초점을 맞추고 있습니다. 악기, 음높이(pitch), 속도(velocity)에 따라 구분된 오디오 데이터를 학습하여 일정한 음높이의 수 초 짜리 짧은 소리를 생성하는 태스크에 대해 기존 모델들에 비해 향상된 성능을 보여줍니다.

GAN 기반의 모델 구조는 WaveNET과 [(Aaron van den Oord et al., 2016)](https://arxiv.org/abs/1609.03499) 같은 자기회귀적(autoregressive) 모델에 비해 학습과 생성 속도 측면에서 큰 이점이 있고 전역적인(global) 잠재(latent) 벡터를 통해 학습하므로 소리의 자연스러운 보간(interpolation)이 가능합니다. 예를 들어 피아노와 바이올린 소리에 해당하는 잠재 벡터 사이를 보간함에 따라 두 음색(timbre)이 섞인 소리를 생성하는 것이 가능합니다.

또한 같은 GAN 기반의 모델인 WaveGAN과 [(Chris Donahue et al., 2019)](https://openreview.net/forum?id=ByMVTsR5KQ) 비교하여 위상의 표현과 주파수 해상도 등의 요소가 더 높은 오디오 품질(fidelity)에 기여한다는 것을 실험적으로 잘 보여줍니다.

<br><br>

## Generating Instrument Timbres

이미지 합성 분야에서 GAN 모델의 빠른 발전을 생각해보면 처음에는 제한된 데이터셋에 초점을 맞추어 모델을 개발하고 점진적으로 자유도를 증가시켜가며 발전시켰습니다. 예를 들어 CelebA 데이터셋은 [(Liu et al., 2015)](https://openaccess.thecvf.com/content_iccv_2015/html/Liu_Deep_Learning_Face_ICCV_2015_paper.html) 자세나 동작 같은 요소들을 제거하고 얼굴만 이미지의 중앙에 위치하도록 잘라낸 데이터셋인데, 이를 기반으로 많은 연구자들이 모델의 성능을 평가하며 향상시키다가 점차 넓은 범위의 도메인으로 일반화하는 방식으로 발전이 이루어졌습니다.

이와 같은 목적으로 오디오 분야에서 만들어진 데이터셋이 NSynth입니다. NSynth는 광범위한 종류의 오디오를 모두 포함하는 대신 악기의 음색, 음높이, 속도로 요소를 제한하고 형식을 통일시켜 모델이 생성하는 소리의 품질을 평가하고 향상시킬 수 있도록 구성되어 있습니다. 또한 악기, 음높이 등의 레이블링이 되어 있기 때문에 조건부 학습을 가능하게 합니다. GANSynth 논문의 주된 기여는 GAN 기반의 오디오 생성 모델을 NSynth 데이터셋을 기반으로 평가하여 성능에 영향을 미치는 중요한 요소들을 발견하고 이후의 연구 방향을 제시한 것입니다.

<br><br>

## Effective Audio Generation for GANs

이미지와 달리 오디오 신호는 높은 주기성을 가지고 있습니다. 주기적인 파형이 중간에 불규칙하게 변하면 사람은 소리가 부자연스럽다고 느끼기 때문에 높은 품질의 오디오를 생성하려면 파형 전반에 걸쳐서 규칙성을 유지시켜주는 것이 중요합니다. 하지만 프레임 단위로 오디오를 생성할 때 규칙적으로 위상의 정렬을 유지시키는 것은 어려운 문제입니다. 아래의 그림은 프레임 기반으로 파형을 학습시킬 때 발생할 수 있는 위상 정렬과 관련된 이슈를 나타낸 것입니다.

<p align="center">
    <img src="https://i.ibb.co/YW6HsGJ/frame-phase.png" alt="frame-phase" border="0">
</p>

맨 위의 오디오 파형에서 세로 점선은 신호의 주기, 검정색 점은 프레임의 스트라이드를 나타냅니다. 일반적으로 오디오 신호는 다양한 주파수 성분들로 이루어져 있으므로 이렇게 원래 신호와 프레임의 주기가 다른 경우가 많습니다. 검정색 점에 연결된 가로 실선은 프레임 내에서의 위상 차이, 즉 정렬이 틀어진 것을 나타내고 마찬가지로 아래에 각 프레임에서의 위상을 노란색 막대로 표현하고 있습니다.

시간 도메인에서 이렇게 각 주파수에서 정렬이 틀어진 위상 정보까지 모든 조합을 다 학습하는 것은 상당히 도전적인 문제입니다. 따라서 GANSynth는 스펙트로그램을 기반으로 학습하고 틀어진 위상 정렬을 학습하기에 유리한 표현 방식을 제안합니다.

위 그림에서 세번째 줄의 주황색 막대는 $\small -\pi \sim \pi$ 범위의 위상을 $\small 0 \sim 2\pi$ 범위로 풀어낸(unwrap) 것입니다. 이 그림을 보면 직관적으로 신호와 프레임의 주기 차이에 의한 위상이 시간에 따라 일정하게 변하는 형태라는 것을 알 수 있습니다. 따라서 위상을 시간에 대해 미분하면 상수에 가까운 값이 나오고 그림의 네번째 줄에 이것이 표현되어 있습니다. 이렇게 위상의 미분으로 얻어지는 시간에 따른 주기적 파동의 변화를 설명하는 요소를 일반적으로 신호 처리 분야에서 순간 각주파수(instantaneous angular frequency)라고 하는데, 정확하게 일치하는 개념은 아니지만 이 논문에서도 용어를 차용하여 순간 주파수(instantaneous frequency)라고 표현합니다.

맨 아래의 스펙트로그램들은 NSynth 데이터셋에 있는 트럼펫 소리의 예시입니다. 각 프레임마다 위상이 주기적으로 변화하는 것이 보이고 이를 언래핑하면 부드럽게 발산하는 형태로 나타납니다. 그리고 미분을 취해 얻은 순간 주파수 스펙트로그램에서는 배음 주파수(harmonic frequency)가 비교적 색이 일정한 실선 대역으로 나타납니다.

<br><br>

## Dataset

NSynth 데이터셋은 1000개의 다른 악기로 연주한 300,000개의 4초 짜리 음(musical note)으로 이루어져 있습니다. 샘플 레이트는 16 kHz로 64000개 길이의 샘플을 만듭니다. GANSynth 논문에서는 대부분의 사람들에게 자연스럽다고 인식되는 소리들을 사용하기 위해서 데이터셋 중에서 어쿠스틱 악기와 MIDI 24-84에 해당하는 음높이의 샘플들만 골랐습니다.

이렇게 선택된 데이터의 총 개수는 70,379이고 대부분 현악기, 금관악기, 목관악기, 그리고 타악기 소리에 해당합니다. 트레인 셋과 테스트 셋은 80/20으로 나눕니다.

<br><br>

## Architecture and Representation

GANSynth는 기본적으로 낮은 해상도에서 시작하여 단계적으로 해상도를 높이면서 학습을 진행함으로써 안정적이고 품질이 좋은 이미지를 생성하는 PGGAN 모델 구조를 사용합니다. 먼저 가우시안 분포에서 벡터를 샘플링 한 뒤 generator에서 $\small (2 \times 2), (4 \times 4), ..., (128, \times, 128)$과 같이 해상도를 높여가면서 이미지를 생성하고 discriminator는 generator와 대칭 구조의 다운샘플링 과정을 통해 진짜 데이터셋의 이미지와 생성된 이미지를 구별하도록 학습합니다. 학습 방식으로는 WGAN-GP를 사용하는데 동일한 방법을 적용한 [WaveGAN 논문 리뷰 포스트](https://jeffmung.github.io/2024/02/08/paper-review-5-wavegan/)의 링크로 설명을 대체합니다. 논문의 부록에 더 구체적인 하이퍼파라미터도 나와 있습니다.

학습 데이터로 NSynth를 사용하기 때문에 음높이에 대한 원핫 벡터를 입력에 조건화하여 사용합니다. 또한 음높이 레이블을 예측하는 보조 분류 손실(auxiliary classification loss)을 discriminator에 추가합니다. 이러한 손실 함수는 AC-GAN에서 [(Augustus Odena et al., 2017)](https://proceedings.mlr.press/v70/odena17a.html) 제안한 방법과 같습니다.

타겟으로 사용하는 스펙트로그램은 여러 가지 종류를 사용하여 비교합니다. 기본적으로는 256의 스트라이드와 1024의 프레임 크기로 STFT를 적용하여 513개 주파수 구간(frequency bin)의 스펙트로그램을 얻습니다. 이때 Nyquist 주파수는 제외하고 시간 축으로 패딩을 하여 이미지의 크기를 (256, 512, 2)로 맞춥니다. 두 개의 채널은 각각 크기와 위상에 해당합니다. 크기는 로그 스케일으로 만들어준 뒤 -1에서 1 사이로 스케일링합니다. 위상도 마찬가지로 -1에서 1 사이의 값으로 스케일링하는데, 이 버전을 **"phase"** 모델이라고 명명합니다.

추가로 위의 그림에 있던 것처럼 위상을 언래핑하고 프레임 간의 차이를 계산해서 얻은 순간 주파수를 사용하는 것을 **"IF"** 모델이라고 부릅니다. 또한 주파수 해상도에 대한 영향을 확인할 수 있도록 프레임 크기와 스트라이드를 두 배씩 증가시켜 (128, 1024, 2)의 크기로 만든 더 높은 주파수 해상도의 스펙트로그램을 **"+H"** 버전이라고 합니다. 마지막으로 주파수 스케일을 mel 스케일로 변환한 **"IF-Mel"** 버전도 만듭니다.

이전 연구들 중 크기에 대한 하나의 채널로만 스펙트로그램을 생성하는 Tacotron이나 [(Yuxuan Wang et al., 2017)](https://arxiv.org/abs/1703.10135) WaveGAN 논문의 SpecGAN과 달리 GANSynth는 위상 채널도 같이 생성하므로 다시 파형을 복원할 때 inverse STFT를 사용합니다. 로그나 mel 스케일을 다시 선형 스케일로 복원할 때도 단순한 역변환을 해주는데, 이러한 과정에서 오디오 품질이 크게 손상되지 않았다고 합니다.

실험 비교군으로는 WaveGAN과 WaveNET autoencoder를 [(Jesse Engel et al., 2017)](https://arxiv.org/abs/1704.01279) 사용합니다. 비교군 모델들에 대한 구현 정보도 논문에 더 자세하게 나와 있습니다.

<br><br>

## Metrics

오디오 품질에 대해 평가할 수 있는 완벽하고 객관적인 지표는 사실상 없기 때문에 다양한 평가 지표를 사용하여 실험 결과를 종합적으로 해석합니다.

### Human Evaluation

평가 참여자들은 같은 음높이에 대한 두 모델의 샘플을 듣고 five-level Likert scale로 어떤 샘플이 더 좋은 오디오 품질을 갖는지 평가합니다. 각각의 모델에서 800개의 샘플을 생성하여 평가에 사용합니다.

### Number of Statistically-Different Bins (NDB)

NDB는 [(Eitan Richardson and Yair Weiss, 2018)](http://arxiv.org/abs/1805.12462) 생성된 샘플의 다양성을 평가하기 위한 지표입니다. 학습 데이터셋의 샘플들을 로그 스펙트로그램 공간에서 $\small k=50$으로 $\small k$-means 클러스터링 하고, 생성된 샘플들은 같은 공간에서 가장 가까운 셀로 배정합니다. 생성된 샘플과 학습 데이터 사이에서 이표본 이항 검정(two-sample Binomial test)에 의하여 통계적으로 충분히 다른 셀의 개수가 NDB 값이 됩니다. 따라서 NDB 값이 작을수록 생성된 샘플의 분포가 학습 데이터셋의 분포를 따르고 다양성이 높도록 하는 모델입니다.

### Inception Score (IS)

IS는 [(Tim Salimans et al., 2016)](https://arxiv.org/abs/1606.03498) 클래스 레이블이 있을 때 클래스에 대한 샘플의 조건부 확률과 주변(marginal) 확률 사이의 KL 발산의 기대값으로 정의합니다. 생성된 샘플들이 각각 하나의 클래스로 쉽게 구분되고, 여러 클래스에 다양하게 분포되어 있을 때 IS 값이 높습니다. IS에 대한 더 자세한 설명은 [WaveGAN 논문 리뷰 포스트](https://jeffmung.github.io/2024/02/08/paper-review-5-wavegan/)에 있습니다.

GANSynth 실험에서는 음높이를 클래스로 사용하고 NSynth 데이터셋에 대해 학습한 음높이 분류 모델에서 추출된 특징(feature)을 IS 계산에 사용합니다.

### Pitch Accuracy (PA) and Pitch Entropy (PE)

IS는 잘 구분되지 않는 음높이의 샘플을 생성하면서 다양하지 않은 음높이들만 생성하는 모델에 대해서도 높은 점수를 부여하는 한계가 있습니다. 이러한 경우를 구분해내기 위하여 음높이 분류 모델로 생성된 샘플의 PA와 PE를 따로 측정합니다. PA는 높고 PE는 낮아야 잘 구분되는 음높이의 샘플을 생성하는 모델입니다.

### Fréchet Inception Distance (FID)

FID는 [(Martin Heusel et al., 2017)](https://arxiv.org/abs/1706.08500) 사전학습된 클래스 분류 모델로 추출된 특징에 대한 2-Wasserstein 거리로 학습 데이터셋과 모델이 생성된 샘플들의 분포를 비교하는 지표입니다. IS와 마찬가지로 다양성이 높고 오디오 품질이 좋을 때 FID 값이 높습니다. 클래스 분류 모델로는 역시 음높이 분류 모델을 사용합니다.

<br><br>

## 실험

실험 결과에 대한 데모 오디오 샘플들은 구글 AI의 [GANSynth 프로젝트 웹페이지](https://storage.googleapis.com/magentadata/papers/gansynth/index.html)에서 들어볼 수 있습니다.

### Quantitative Analysis

실험 결과는 아래 표에 나와 있습니다. 막대 그래프는 전체 8개 중에서 상위 6개 모델에 대해서만 따로 사람의 비교 평가 결과를 정리한 것입니다. 전체적으로 일관적인 트렌드를 보여줍니다.

<p align="center">
    <img src="https://i.ibb.co/KmsPwn9/result.png" alt="result" border="0">
</p>

우선 +H가 붙은 주파수 해상도가 높은 모델과 mel 스케일의 모델이 좋은 성능을 나타냅니다. IF가 그냥 phase 모델에 비해 확연히 좋은 성능을 보여주는 것에서 위상 표현의 중요성도 알 수 있습니다. IS, PA, PE에 대해서는 모든 모델들이 기본적으로 음높이에 대해 조건화된 입력을 넣어주기 때문에 대부분 좋은 수치를 나타냅니다.

NDB는 음높이가 아닌 음색에 대한 분포도 영향을 미치므로 다양성을 좀 더 잘 나타낼 수 있습니다. 아래의 그림은 음높이 60에 대해 생성한 샘플들의 NDB 값과 각각의 클러스터에 대한 비율을 보여줍니다.

<p align="center">
    <img src="https://i.ibb.co/M9X9VQc/ndb-proportion.png" alt="ndb-proportion" border="0">
</p>

NDB 값이 45인 WaveGAN과 32인 IF-Mel + H 모델이 어느 정도 다른 분포를 나타내는 것인지를 알 수 있습니다.

### Phase Coherence

아래 그림은 각 모델들이 생성한 샘플의 위상 일관성(coherence)을 나타낸 것입니다. 샘플들은 비교를 위해 각각의 모델에서 최대한 비슷해 보이는 파형들을 고른 것입니다.

<p align="center">
    <img src="https://i.ibb.co/7zP7psg/phase-coherence.png" alt="phase-coherence" border="0">
</p>

맨 위 줄은 샘플의 음높이 MIDI C60에 해당하는 기본 주기(fundamental periodicity)에 따라 파형을 겹쳐놓은 것입니다. 실제 데이터는 파형이 완벽하게 주기적이기 때문에 전부 겹쳐져 있습니다. WaveGAN과 PhaseGAN의 경우에는 위상 정렬이 많이 틀어져 있기 때문에 파형들이 뚜렷하게 겹쳐져 있지 않습니다. 이에 비해 IFGAN은 위상이 훨씬 더 잘 정렬되어서 사이클마다 큰 차이가 없는 것을 볼 수 있습니다.

아래 줄의 레인보우그램 역시 같은 경향성을 보여줍니다. WaveGAN과 PhaseGAN에 비해 IF 모델은 각각의 배음에 대해서 뚜렷하고 일관적인 색을 보여줍니다.

### Interpolation

GAN 기반 모델의 장점은 전체 시퀀스가 같은 잠재 벡터에 대해 생성되기 때문에 의미 있는 보간이 가능하다는 것입니다. WaveNET과 같은 자기회귀적 모델은 짧은 서브시퀀스에 대해서 국소적인 잠재 벡터를 학습하기 때문에 이러한 측면에서 한계가 있습니다.

<p align="center">
    <img src="https://i.ibb.co/yX0yrVk/interpolation.png" alt="interpolation" border="0">
</p>

위의 그림은 두 개의 다른 음색을 갖는 소리 사이에서 보간을 하여 생성한 샘플들의 레인보우그램입니다. WaveNET의 경우에는 중간 단계의 소리가 두 소리가 섞인 것이라기 보다는 동떨어진 소리 로 나는 경향이 있습니다. 반면 GANSynth는 한 음색에서 다른 음색으로 점진적으로 변해가는 소리를 잘 나타냅니다.

좀 더 음악적인 예시로 GANSynth에서 생성된 음들로 Bach's Suite No.1 in G minor의 prelude를 연주하는데 보간을 통해 음색을 서서히 변화시키는 실험도 했습니다. 실제 샘플을 들어보면 상당히 흥미로운 결과이고, 아래 그림은 첫 20초에 해당하는 스펙트로그램입니다.

<p align="center">
    <img src="https://i.ibb.co/MPRPLr6/interpolation-bach.png" alt="interpolation-bach" border="0">
</p>

### Consistent Timbre Across Pitch

잠재 벡터를 고정시키고 음높이 조건 벡터만 다르게 입력을 넣어서 샘플을 생성했을 때 GANSynth는 일관적인 음색의 소리를 들려줍니다. 이에 대한 실험 결과도 오디오 샘플이 공개되어 있으니 들어볼 수 있습니다.

### Fast Generation

자기회귀적 모델에 비해 확실한 장점인 학습과 생성 속도에 대한 실험 결과도 명시되어 있습니다. 같은 조건으로 실험했을 때 WaveNET autoencoder에 비해 GANSynth가 4초 짜리 오디오를 53,880배 빠르게 생성합니다.

<br><br>

## Reference

[Jesse Engel, Kumar Krishna Agrawal, Shuo Chen, Ishaan Gulrajani, Chris Donahue and Adam Roberts. GANSynth: Adversarial Neural Audio Synthesis. In ICLR, 2019.](https://openreview.net/forum?id=H1xQVn09FX)

[Official source code of GANSynth](http://goo.gl/magenta/gansynth-code)