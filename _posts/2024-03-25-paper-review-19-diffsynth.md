---
layout: post
title: "[논문 리뷰] Synthesizer Sound Matching with Differentiable DSP"
image: https://i.ibb.co/n36gRwT/thumbnail.png
date: 2024-03-25
tags: 
categories: paper-review
use_math: true
---

## 논문 개요

<!-- excerpt-start -->

원하는 소리를 신디사이저로 모방하기 위해서는 도메인 지식에 기반하여 신디사이저의 파라미터를 직접 조정하거나 사운드 디자이너들이 만든 적절한 프리셋을 찾아야 합니다. 이러한 사운드 매칭 태스크를 잘 수행하기 위해서는 많은 경험과 숙련도가 필요하기 때문에 원하는 소리를 모방할 수 있도록 신디사이저의 파라미터를 자동으로 찾아주는 모델의 개발에 대한 필요성이 높습니다.

기존의 신경망 기반 사운드 매칭 모델 학습에는 주로 타겟 소리의 파라미터와 모델에서 추정한 파라미터 사이의 손실만이 이용되었습니다. 이러한 파라미터 손실(parameter loss)은 값이 작더라도 실제 만들어진 소리에는 큰 차이가 있을 수 있다는 한계가 있습니다. 또한 동일한 신디사이저로 만들어지지 않은(out-of-domain) 소리에 대해서는 파라미터 값을 알 수 없으므로 학습을 할 수 없습니다. 이러한 기존 방법은 아래 그림에 표현되어 있습니다.

<p align="center">
<img src="https://i.ibb.co/njJnNHm/nn-synthesizer.png" alt="nn-synthesizer" border="0">
</p>

이 논문에서는 미분 가능한(differentiable) DSP 모듈을 사용하여 파라미터 손실 뿐만 아니라 스펙트럼 손실(spectral loss)을 통한 엔드 투 엔드 학습이 가능한 사운드 매칭 모델을 제안합니다. 제안된 모델 DiffSynth는 같은 신디사이저로 만들어진(in-domain) 소리 데이터에 대해 먼저 파라미터 손실로 사전학습 되고 out-of-domain 데이터에 대해 파인튜닝됩니다.

<br><br>

## Architecture

DiffSynth의 구조는 아래 그림에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/j67yBm6/architecture.png" alt="architecture" border="0">
</p>

먼저 오디오 데이터로부터 멜 스펙트로그램이 추출되어 모델의 입력으로 들어갑니다. In-domain 데이터에 대해서는 파라미터의 실제값(ground-truth)과 estimator network에서 나온 값 사이의 파라미터 손실이 계산되어 모델을 학습시킵니다.

Differentiable sythesizer는 추정된 파라미터를 사용하여 소리를 합성합니다. 합성된 소리로부터 추출된 스펙트로그램은 스펙트럼 손실(spectral loss) 계산에 사용되어 in-domain과 out-of-domain 데이터 모두에 대한 엔드 투 엔드 학습을 가능하게 합니다.

<br><br>

## Differentiable Synthesizer

사운드 매칭에 사용하는 신디사이저는 DDSP의 [(Jesse Engel et al., 2020)](https://openreview.net/forum?id=B1x1ma4tDr) 아이디어에 기반한 미분 가능한 신디사이저(differentiable synthesizer)입니다. 원본 DDSP 논문에서는 harmonics-plus-noise 모델을 사용하지만 여기서는 Native Instruments의 Razor나 Image-Line의 Harmor 같은 additive-subtractive 신디사이저의 단순한 버전을 사용합니다.
신디사이저는 주파수와 진폭에 따라 달라지는 두 개의 오실레이터를 사용합니다. 각각의 오실레이터는 톱니파(sawtooth wave)와 방형파(square wave)를 생성하고 두 파형은 보간될 수 있습니다. 기본 주파수(fundamental frequency) $\small f$ 에 대한 두 파형은 아래 식과 같이 정의됩니다.

<br>
\begin{align}
x\_{\text{sawtooth}} (t) &= \frac{2}{\pi} \sum\_{k=1}^{\infty} \frac{\sin(k \cdot 2 \pi ft)}{k} \newline
\newline
x\_{\text{square}} (t) &= \frac{4}{\pi} \sum\_{k=1}^{\infty} \frac{\sin ((2k - 1) \cdot 2 \pi ft)}{2k - 1}
\end{align}
<br>

두 오실레이터의 출력은 합쳐진 뒤 공진 로우패스 필터(resonant low-pass filter)로 들어갑니다. 이 필터는 컷오프 주파수 이상의 배음(harmonics)을 약화시키고 공진 파라미터(resonance parameter)에 따라 컷오프 주파수 근처를 강조되게 합니다. 공진 필터는 배음들의 진폭에 필터의 주파수 반응(frequency response)를 곱하는 것으로 구현합니다.

이러한 신디사이저의 합성 파라미터는 진폭, 주파수, 오실레이터에서 나온 톱니파와 방형파의 혼합 파형, 컷오프 주파수, 필터의 공진 값이 됩니다. 이 중 진폭과 컷오프 주파수는 ADSR 엔벨로프(envelope)에 의해 프레임 단위로 변하는 값을 사용하고 다른 파라미터들은 고정된 단일 값으로 설정합니다.

<br><br>

## Estimator Network

Estimator network는 멜 스펙트로그램으로부터 각각의 타임스텝에서의 합성 파라미터를 예측하도록 학습됩니다. 멜 스펙트로그램은 프레임 단위로 분리되어 1D CNN으로 들어갑니다. CNN에서 추출된 표현들(representations)은 GRU와 선형 층(linear layer)을 통해 $\small (0, 1)$ 범위로 정규화된 합성 파라미터 예측값을 출력합니다. 시간에 따라 변하지 않는 파라미터들은 GRU의 마지막 타임스텝 출력으로부터 얻어집니다.

<br><br>

## Training

Estimator network는 파라미터 손실과 스펙트럼 손실을 모두 사용하여 학습될 수 있습니다. 파라미터 손실은 예측된 파라미터와 실제값 사이의 L1 손실로 정의됩니다. 스펙트럼 로스는 DDSP의 다중 스케일 스펙트럼 손실(multi-scale spectral loss)을 그대로 사용합니다.

모델은 먼저 in-domain 데이터셋에 대해 파라미터 손실로 학습됩니다. 그 뒤 스펙트럼 손실이 점진적으로 추가되다가 최종적으로 파라미터 손실을 완전히 대체합니다. 마지막에는 out-of-domain 데이터셋을 사용하여 스펙트럼 손실을 통해서만 모델이 학습됩니다.

<br><br>

## 실험

파라미터 손실과 스펙트럼 손실의 영향을 분석하기 위해 세 가지 학습 방법으로 학습된 모델들을 사용하여 서로 비교합니다.

Parameter-loss only model (P-loss)은 400 epochs 동안 파라미터 손실로만 학습됩니다.

In-domain spectral loss model (Synth)은 50 epochs 동안 파라미터 손실로 학습된 뒤 150 epochs 동안은 스펙트럼 손실의 가중치가 선형적으로 증가하고 파라미터 손실의 가중치는 선형적으로 감소하도록 합니다. 마지막 200 epochs 동안은 in-domain 데이터셋에 대해 스펙트럼 손실로만 학습됩니다.

Out-of-domain spectral loss model (Real)은 첫 200 epochs 동안은 Synth 모델과 동일하게 학습하고 이후 200 epochs 동안 out-of-domain 데이터셋에 대해 스펙트럼 손실만 사용하여 학습됩니다.
In-domain 데이터셋으로는 임의로 샘플링된 파라미터로 합성된 소리를 사용합니다. 시간에 따라 변하는 파라미터의 경우 ADSR 엔벨로프의 어택 시간(attack time), 디케이 시간(decay time), 서스테인 레벨(sustain level), 릴리즈 시간(release time)도 모두 임의로 샘플링되고 생성된 엔벨로프에 가우시안 노이즈가 더해집니다. 소리는 3초까지 노트 입력이 지속된 이후 릴리즈되고 총 길이는 4초입니다.

Out-of-domain 데이터셋으로는 NSynth를 [(Jesse Engel et al., 2017)](https://proceedings.mlr.press/v70/engel17a.html) 사용합니다. 이 데이터셋의 소리들도 3초 동안 노트 입력이 지속되고 총 길이는 4초입니다.

실험 결과에 대한 데모 사운드는 [프로젝트 웹페이지](https://hyakuchiki.github.io/DiffSynthISMIR/)에서 들어볼 수 있습니다.

### Quantitative Results

정량적 지표로는 log-spectral distortion (LSD), multi-scale spectral loss (Multi), 그리고 in-domain 데이터에 대해서는 L1 parameter loss (Param)을 사용합니다. 정량적 평가 결과는 아래 표에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/D8F4SCw/quantitative-result.png" alt="quantitative-result" border="0">
</p>

P-loss 모델은 Param 지표 값이 가장 좋지만 다른 값들은 상대적으로 떨어집니다. 이는 파라미터 손실이 사운드 매칭 성능의 기준으로 적절하지 못하다는 것을 보여주는 결과입니다. Real 모델이 out-of-domain 데이터에 대해 가장 우수한 성능을 나타내는 것을 통해 파인 튜닝이 효과적이라는 것을 알 수 있습니다.

<p align="center">
<img src="https://i.ibb.co/6W2Xh1H/out-domain-lsd.png" alt="out-domain-lsd" border="0">
</p>

위 그림은 out-of-domain 평가 데이터셋(validation set)에 대한 LSD 값을 학습 도중에 기록한 것입니다. Synth와 Real 모델에서 50 epoch 부터 점진적으로 도입된 스펙트럼 손실이 LSD 값을 떨어뜨리고 200 epoch 이후에는 Real 모델에서 LSD 값이 급격하게 낮아지는 것이 나타나 있습니다.

### Subjective Evaluation

주관적 평가를 위해 참여자들은 타겟 소리를 먼저 듣고 두 모델에 의해 생성된 소리 중 어떤 것이 더 비슷한지 선택합니다. 타겟 소리는 out-of-domain 테스트 데이터셋에서 임의로 샘플링됩니다. 그 결과는 아래 그림에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/6g71LbH/subjective-result.png" alt="subjective-result" border="0">
</p>

정량적 평가 결과와 마찬가지로 Real 모델의 성능이 가장 우수하고 P-loss 모델이 가장 떨어지는 것을 볼 수 있습니다. 이 실험에 대한 특정한 스펙트로그램 예시는 아래 그림에 있습니다.

<p align="center">
<img src="https://i.ibb.co/0KkZh9H/subjective-spec.png" alt="subjective-spec" border="0">
</p>

Out-of-domain 데이터 중 첫 번째 줄은 배음이 많은 관악기 소리인데 Real 모델만 관악기와 비슷한 소리를 만들어냅니다. 두 번째 out-of-domain 소리에 대해서는 P-loss와 Synth 모델이 음높이(pitch)를 다르게 생성한 것을 볼 수 있습니다.

<br><br>

## Reference

[Naotake Masuda and Daisuke Saito. Synthesizer Sound Matching with Differentiable DSP. In ISMIR, 2021.](https://archives.ismir.net/ismir2021/paper/000053.pdf)

[Official Source Code of DiffSynth](https://github.com/hyakuchiki/diffsynth)