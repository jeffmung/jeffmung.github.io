---
layout: post
title: "[논문 리뷰] Improving Semi-Supervised Differentiable Synthesizer Sound Matching for Practical Applications"
image: https://i.ibb.co/NpxvGH2/thumbnail.png
date: 2024-03-27
tags: 
categories: paper-review
use_math: true
---

## 논문 개요

<!-- excerpt-start -->

SSSSM-DDSP는 2021년 ISMIR에 발표된 DiffSynth의 [(Naotake Masuda and Daisuke Saito, 2021)](https://archives.ismir.net/ismir2021/paper/000053.pdf) 후속 논문으로 기존 연구에서 미분 가능한 엔벨로프(envelope), 리버브, 코러스 FX의 설계 방법과 이 모듈들이 포함되었을 때의 결과 분석이 새롭게 추가되었습니다. DiffSynth 논문에 대한 설명은 [이전 포스트](https://jeffmung.github.io/2024/03/25/paper-review-19-diffsynth/)에 작성되어 있습니다. 이 포스트에서는 새롭게 추가된 내용에 대해서만 다루겠습니다.

<br><br>

## Architecture Overview

기본적인 모델 구조는 DiffSynth와 동일하고 실험에 따라서 일부 모듈이 변경되거나 추가되었습니다. 전체 구조는 아래 그림에 나와 있습니다.

<p align="center">
<img src="https://i.ibb.co/F0vgSQc/architecture.png" alt="architecture" border="0">
</p>

<br><br>

## Estimator network

Estimator network의 구조는 기존과 완전히 동일하고 1D CNN 대신 2D CNN을 썼을 경우의 결과를 비교하기 위한 설정이 추가되었습니다. 2D CNN을 사용하는 경우에는 멜 스펙트로그램 입력이 시간 축으로도 다운샘플링되고 마지막에 소프트맥스를 통해 시간에 따라 변하지 않는 단일 파라미터 값을 카테고리형 분포로 출력합니다.

<br><br>

## Differentiable Synthesizer

신디사이저도 기존 논문과 같이 톱니파(sawtooth wave)와 방형파(square wave)를 생성하는 두 오실레이터를 기반으로 한 additive-subtractive synthesizer를 사용하고 미분 가능하도록 구현된 리버브와 코러스 이펙트 모듈이 추가되었습니다. 또한 두 오실레이터의 진폭(amplitude)과 컷오프 주파수(cutoff frequency)에 적용되는 엔벨로프도 미분 가능하도록 구현됩니다. 전체 구조는 아래 그림에 요약되어 있습니다.

<p align="center">
<img src="https://i.ibb.co/M6sB5dr/synthesizer.png" alt="synthesizer" border="0">
</p>

코러스는 딜레이된 오디오를 원래의 신호와 혼합해주는 것으로 구현합니다. 딜레이의 양은 low-frequency oscillator (LFO)로 조정됩니다. "delay" 파라미터는 딜레이의 기준값(base value)을 조절하고 "mix" 파라미터는 원래 오디오와 딜레이된 신호의 비율을 결정합니다.

리버브는 DDSP 논문에서 사용한 것과 같이 finite impulse response (FIR) 컨볼루션으로 구현합니다. "decay" 파라미터는 리버브가 시간에 따라 감소하는 속도를 조절하고 "mix" 파라미터는 원래 오디오와 리버브 신호의 비율을 결정합니다.

오실레이터에서 만들어진 두 파동의 진폭과 컷오프 주파수에 적용되는 엔벨로프로는 attack time $\small t_a$, decay time $\small t_b$, sustain level $\small v_s$, release time $\small t_r$, base value $\small v_b$, 그리고 peak value $\small v_p$ 의 여섯 개 파라미터로 표현되는 ADSR 엔벨로프를 사용합니다. ADSR 엔벨로프를 미분 가능하게 구현할 수 있도록 아래 그림과 같이 세 개의 병렬적 요소로 분해합니다.

<p align="center">
<img src="https://i.ibb.co/2v2mX81/envelope.png" alt="envelope" border="0">
</p>

엔벨로프를 분해하는 수식은 다음과 같습니다.

<br>
\begin{equation}
v(t) = \left| \frac{v\_p}{t\_a}t \right|\_{[0, v\_p]} + \left| \frac{v\_s - v\_p}{t\_d}(t - t\_a) \right|\_{[v\_s - v\_p, 0]} + \left| - \frac{v\_s}{t\_r}(t - N\_{\text{off}}) \right|\_{[-v\_s, 0]}
\end{equation}
<br>

이때 $\small v\_p > v\_s > 0$ 이고 $\small \vert x \vert\_{[a,b]}$ 는 클램핑(clamp) 연산을 의미합니다.

<br><br>

## Synthetic Benchmark Experiment

미분 가능하도록 설계된 신디사이저를 사용하여 사운드 매칭 태스크를 수행할 때 스펙트럼 손실(spectral loss)의 경향성을 분석하기 위해 벤치마크 실험을 진행합니다. 이 실험의 목적은 스펙트럼 손실이 각각의 합성 파라미터에 대해 유용한 그래디언트를 제공하는지 확인하기 위한 것입니다.

$\small n$ 개의 합성 파라미터 $\small \boldsymbol{v} = (v_0, v_1, \ldots, v_n)$ 으로부터 합성된 소리를 $\small \boldsymbol{x}$ 라고 할 때 합성 과정을 $\small \boldsymbol{x} = f(\boldsymbol{v})$ 로 표현할 수 있습니다. 전체적인 파라미터들의 특성을 잘 나타낼 수 있는 소리가 나도록 $\small \boldsymbol{v}$ 를 정하고 각각의 파라미터 한 개씩의 값을 최소에서 최대값까지 바꿔가면서 $\small \boldsymbol{v}^{\prime} = (v_0, \ldots, v_k^{\prime}, \ldots, v_n)$ 에 대한 소리 $\small \boldsymbol{x}^{\prime} = f(\boldsymbol{v}^{\prime})$ 을 만듭니다. 그 뒤 다중 스케일 스펙트럼 손실 $\small L(\boldsymbol{x}, \boldsymbol{x}^{\prime})$ 과 그래디언트 $\small \frac{dL}{dv\_k^{\prime}}$ 을 계산하여 기록합니다. 전체 19개 파라미터 중 대표적인 11개에 대한 실험 결과는 아래 그림과 같습니다.

<p align="center">
<img src="https://i.ibb.co/CpjN8Gx/gradient-graphs.png" alt="gradient-graphs" border="0">
</p>

대부분의 파라미터들에 대해서 손실 곡선도 부드럽고 그래디언트도 빨간색으로 표시된 경사 하강법(gradient descent)에 적합한 범위 내에 존재합니다. 하지만 오른쪽 위의 오실레이터 주파수 그래프를 보면 스펙트럼 손실과 그래디언트 경향이 적합하지 않다는 것을 알 수 있습니다. 중간중간에 국소 최적점(local optimum)이 존재하는데 한 오실레이터가 다른 오실레이터의 배음 주파수가 될 때 스펙트럼 손실 값이 낮아지기 때문인 것으로 추정됩니다.

또한 코러스의 딜레이 파라미터도 원래 파라미터에서 조금만 멀어져도 그래디언트가 0 근처에 존재하는 경향이 있어 스펙트럼 손실이 적합하지 않습니다. 코러스 이펙트 역시 음높이에 영향을 미치기 때문에 오실레이터 주파수와 마찬가지로 스펙트럼 손실로 쉽게 조절되기 어려운 것으로 보입니다.

<br><br>

## 실험

실험 결과에 대한 데모 샘플들은 [프로젝트 웹페이지](https://hyakuchiki.github.io/SSSSM-DDSP/)에서 들어볼 수 있습니다.

### Synthesizer Architectures

실험에 사용하는 신디사이저 설정은 기본적으로 Raw, Raw-Env, FX, FX-Env의 4개를 비교하고 몇 가지 변형을 추가합니다. Raw는 코러스와 딜레이를 제외한 것이고 Env는 ADSR 엔벨로프를 추가한 것입니다. Env가 없는 경우에는 오실레이터 진폭과 컷오프 주파수 파라미터를 ADSR의 형태 없이 그냥 프레임 단위로 예측합니다.

앞선 실험 결과에서 스펙트럼 손실이 주파수에 적합하지 않다는 것이 보여졌기 때문에 오실레이터의 주파수를 CREPE로 [(Jong Wook Kim et al., 2018)](https://ieeexplore.ieee.org/abstract/document/8461329) 따로 분리해서 예측하는 모델도 실험합니다. 이 모델은 Raw-f0와 FX-f0라고 명명합니다.

또한 기본적으로는 두 오실레이터의 주파수를 $\small f_2 = \alpha f_1, \, (\alpha > 1)$ 로 정의하여 두 주파수 크기의 순서가 고정되도록 설정하는데 이렇게 순서를 고정하지 않은 Unordered 모델도 테스트합니다.

### Training Strategy

학습 방법은 손실 함수에 따라 기존 DiffSynth와 동일하게 Parameter-loss (P-loss), In-domain spectral loss strategy (Synth), 그리고 Out-of-domain spectral loss strategy (Real)을 사용합니다. 추가로 이펙트가 적용된 FX 모델의 경우 스펙트럼 손실만 사용하는 것이 적합하지 않아 파라미터 손실과 스펙트럼 손실을 계속해서 같이 사용합니다. 첫 50 epochs 동안은 P-loss 모델과 동일하고 이후 150 epochs 동안 스펙트럼 손실의 가중치가 0.5까지 증가합니다. 마지막 200 epochs 동안은 두 손실이 각각 0.5의 가중치로 고정됩니다. 아래 그림에 각각의 학습 방법이 표현되어 있습니다.

<p align="center">
<img src="https://i.ibb.co/XyjnP25/training-strategies.png" alt="training-strategies" border="0">
</p>

### Measures for Match Quality

정량적 지표로는 log-spectral distortion (LSD), loudness loss (LSD), 그리고 mel-cepstral distortion (MCD)를 사용합니다. LSD는 다음과 같이 두 스펙트럼 사이의 거리로 정의됩니다.

<br>
\begin{equation}
D\_{LS} = \sqrt{\sum\_i 10 \log\_{10} \frac{S(i)}{\hat{S}(i)}}
\end{equation}
<br>

MCD는 첫 번째를 제외한 MFCC 사이의 유클리드 거리입니다. MFCC가 음색과 관련이 있는 값이기 때문에 MCD를 통해 두 음색이 얼마나 비슷한지 대략적으로 판단할 수 있습니다. Loud는 A-weighting을 적용한 음량(loudness) 사이의 L1 거리입니다.

### Results

기본적인 실험 결과는 아래 표들에 정리되어 있습니다.

<p align="center">
<img src="https://i.ibb.co/R0DdnFr/overall-results.png" alt="overall-results" border="0">
</p>

ADSR 엔벨로프나 FX를 추가했을 때 모두 성능이 더 떨어집니다. 특히 FX를 추가했을 때 성능이 더 떨어지는데 아래 그림과 같이 예측된 파라미터의 분포를 보면 한 가지 원인을 알 수 있습니다.

<p align="center">
<img src="https://i.ibb.co/Vv1vy9D/chorus-parameters.png" alt="chorus-parameters" border="0">
</p>

학습이나 평가에 사용된 데이터셋에는 모든 파라미터들이 다 최소부터 최대값까지 고르게 분포하는데 예측된 결과를 보면 Synth와 Real 모델에서 코러스 파라미터가 굉장히 작은 범위 내에서만 사용됩니다. Mixed 모델은 더 낫긴 하지만 코러스 딜레이 분포가 균일하지는 않습니다.

### Frequency Conditioning

CREPE로 주파수를 따로 예측하는 모델은 하나의 오실레이터만 사용합니다. 이 모델에 대한 실험 결과는 아래 표에 있습니다.

<p align="center">
<img src="https://i.ibb.co/VHJZNDr/f0-results.png" alt="f0-results" border="0">
</p>

다른 모델들과 설정이 다르기 때문에 일대일로 비교할 수는 없지만 주파수를 따로 분리해서 예측하는 것이 성능에 도움이 되는 방향이라는 것은 실험으로 확인이 됩니다.

### Unordered Oscillators

두 오실레이터의 주파수 크기 순서가 없는 모델에 대한 실험 결과는 아래 표에 있습니다.

<p align="center">
<img src="https://i.ibb.co/0r7yRKn/unordered-results.png" alt="unordered-results" border="0">
</p>

주파수 순서가 고정된 모델에 비해 성능이 떨어지는 결과가 나타납니다. 이렇게 순서가 고정되지 않은 경우 소리의 어떤 부분이 어떤 오실레이터에 해당하는 부분인지 알기가 어렵기 때문일 가능성이 있습니다.

<br><br>

## Reference

[Naotake Masuda and Daisuke Saito. Improving Semi-Supervised Differentiable Synthesizer Sound Matching for Practical Applications. In TASLP, 2023.](https://ieeexplore.ieee.org/document/10017350/)

[Official Source Code of SSSSM-DDSP](https://github.com/hyakuchiki/SSSSM-DDSP)