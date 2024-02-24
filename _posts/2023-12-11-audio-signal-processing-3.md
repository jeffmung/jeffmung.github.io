---
layout: post
title: "[오디오 신호 처리] 3. 오디오 특징의 종류 (Types of Audio Features)"
image: https://i.ibb.co/YQgkh1n/audio-features-thumbnail.png
date: 2023-12-11
tags: 
categories: audio-signal-processing
use_math: true
---

<br><br>

## Audio Features
<!-- excerpt-start -->
오디오 신호의 다양한 feature들은 서로 다른 소리의 측면을 표현하고 설명합니다. 이러한 오디오 feature들은 지능형 오디오 시스템을 만드는 데에 사용됩니다. 예를 들어, 오디오 신호의 몇 가지 feature들을 추출한 다음 이를 머신러닝 알고리즘에 전달하여 오디오 신호의 패턴을 식별할 수 있습니다. 오디오 feature들의 종류에는 어떤 것들이 있는지 여러 기준의 구분 방식으로 알아보겠습니다.

<br><br>

## Level of Abstraction

Abstraction의 수준에 따라 오디오 feature들을 구분할 수 있습니다. 여기서 abstraction이라는 것은 그 정도가 낮을수록 컴퓨터가 이해하기 쉽고, 높을수록 컴퓨터가 직관적으로 이해하기 어려운 추상적인(abstract) feature라는 의미입니다. 반대로 사람은 높은 단계(high-level)의 feature들은 직관적으로 무엇인지 알 수 있지만 낮은 단계(low-level)의 feature들은 배경 지식 없이는 어떤 것을 말하는지 알기 어렵습니다.

<p align="center">
  <img src="https://i.ibb.co/FmFChB3/level-of-abstraction.png" alt="level of abstraction">
</p>

예를 들어, 사람은 어떤 오디오 샘플을 듣고 high-level feature인 멜로디, 분위기 등은 별다른 지식 없이도 파악하고 구분할 수 있지만 그 샘플의 spectral flux는 추출해낼 수 없습니다. 반대로 컴퓨터는 물리적인 특징과 통계량에 기반한 low-level feature인 amplitude envelope나 spectral flux 등은 쉽게 추출해낼 수 있지만 high-level feature인 분위기는 간단하게 구분해낼 수 없습니다.

<br><br>

## Temporal Scope

다른 관점의 오디오 feature 분류 방법 중 하나는 시간적인 범위(temporal scope)에서의 분류입니다.

- <span style="font-family: 'Kanit'; font-size: 115%">Instantaneous</span>
- <span style="font-family: 'Kanit'; font-size: 115%">Segment-level</span>
- <span style="font-family: 'Kanit'; font-size: 115%">Global</span>

첫 번째는 instantaneous feature로 수십~백 ms 정도의 시간에 해당합니다. 인간이 청각적으로 인식할 수 있는 최소한의 시간이 약 10 ms 정도이므로 즉각적인 정보는 그것과 비슷하거나 좀 더 긴 시간 정도라고 할 수 있습니다.

두 번째는 segement-level로 수 초 정도에 해당하는 feature입니다. Instantaneous feature보다 한 단계 긴 수준으로, 음악적인 틀(frame)이나 마디(bar)에 대한 정보를 제공해줄 수 있습니다.

마지막으로 전체 소리에 대한 정보를 제공해주는 global feature입니다. 더 낮은 시간적 수준의 feature들을 종합하고 통계적으로 가공한 정보들도 이에 해당할 수 있습니다.


<br><br>

## Music Aspect

오디오 신호들 중 일반적인 소리가 아닌 음악에 대해서는 음악적인 측면에서 feature들을 구분할 수도 있습니다. 음악 데이터를 다룰 때에는 이러한 음악적인 도메인 내의 다양한 feature들을 활용하는 것이 도움이 될 것입니다.

- <span style="font-family: 'Kanit'; font-size: 115%">Beat</span>
- <span style="font-family: 'Kanit'; font-size: 115%">Timbre</span>
- <span style="font-family: 'Kanit'; font-size: 115%">Pitch</span>
- <span style="font-family: 'Kanit'; font-size: 115%">Harmony</span>
- <span style="font-family: 'Kanit'; font-size: 115%">$\cdots$</span>

<br><br>

## Signal Domain

오디오 신호를 다룰 때 가장 중요할 수도 있는 측면으로, 오디오 feature는 time domain에 존재할 수도 있고 frequency domain에 존재할 수도 있습니다.

### Time Domain

시간에 대해 오디오 신호를 표현했을 때, 즉 x축이 시간이고 y축이 amplitude가 되는 파동의 형태일 때 추출할 수 있는 feature들을 time domain에 있다고 합니다. 예를 들어, amplitude envelope, root-mean square energy, zero crossing rate 등이 여기에 해당합니다.

### Frequency Domain

반면 소리에는 시간 뿐만 아니라 주파수에 대한 측면에서 나타나는 특징들도 많습니다. Time domain의 표현에서는 이러한 정보들을 알 수가 없습니다. 이러한 frequency domain feature에는 band energy ratio, spectral centroid, spectral flux 등이 있습니다. Time domain에서 표현된 오디오 신호에 대해 Fourier transform을 적용하면 frequency domain의 스펙트럼을 얻을 수 있습니다. 이때는 x축이 frequency, y축이 magnitude가 됩니다.

<p align="center">
  <img src="https://i.ibb.co/pXQWWgM/time-frequency.png" alt="time and frequency domain">
</p>

### Time-Frequency Representation

시간과 주파수 모두에 대한 정보를 제공해주는 표현이 있다면 아주 유용할 것입니다. Spectrogram, mel-spectrogram, constant-Q transform 등이 이에 해당합니다. 이 중 spectrogram은 time domain의 파형에 Short-Time Fourier Transform(SFTT)를 적용하여 얻을 수 있습니다. x축은 시간, y축은 주파수를 나타내며 색상은 에너지 강도를 나타냅니다. 색이 밝은 부분은 해당 주파수와 시간에서 더 강한 에너지가 있다는 것을 의미합니다.

<p align="center">
  <img src="https://i.ibb.co/v1BWQfQ/spectrogram.png" alt="spectrogram">
</p>

<br><br>

## Reference

[[Youtube] Valerio Velardo - The Sound of AI, "Types of Audio Features for Machine Learning"](https://youtu.be/ZZ9u1vUtcIA?feature=shared)