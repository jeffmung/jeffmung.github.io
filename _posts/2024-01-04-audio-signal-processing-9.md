---
layout: post
title: "[오디오 신호 처리] 9. Mel-Frequency Cepstral Coefficients (MFCC)"
image: https://i.ibb.co/9tk7R1M/thumbnail.png
date: 2024-01-04
tags: 
categories: audio-signal-processing
use_math: true
---

<br><br>

## Cepstrum

MFCC를 이해하기 위해서는 먼저 cepstrum의 개념에 대해 알아야 됩니다. Cepstrum이라는 단어의 어원은 spectrum의 'spec'을 거꾸로 해서 'ceps'로 만든 것으로 1963년 B. P. Bogert, M. J. Healy, J. W. Tukey의 논문에서 지진의 반사 신호 등에 대한 분석을 위해 처음 제시되었습니다. 단어에서 볼 수 있듯이 cepstrum은 spectrum과 연관이 있는데, 일종의 spectrum의 spectrum과 같은 의미입니다. Cepstrum의 수학적 정의는 다음과 같습니다.

<br>
<center> $ C(x(t)) = \mathcal{F}^{-1}[\log(\mathcal{F}[x(t)])]  $ </center>
<br>

여기서 $\mathcal{F}[\cdot]$과 $\mathcal{F}^{-1}[\cdot]$은 각각 Fourier transform과 inverse Fourier transform을 나타내고, $x(t)$는 time domain에서의 신호입니다. 아래 그림과 같이 어떤 신호의 log power spectrum에 다시 inverse Fourier transform을 적용해서 얻은 것이 cepstrum이 되는 것입니다.

<p align="center">
  <img src="https://i.ibb.co/p0pSPqg/cepstrum-making.png" alt="cepstrum" border="0">
</p>

마지막 cepstrum의 그림에서 x축은 quefrency라고 되어 있고 단위는 시간의 단위와 같습니다. Quefrency는 frequency의 글자 순서를 바꾼 것으로, 주파수와 관련이 있는 개념이라는 것을 알 수 있습니다. 왜 이러한 새로운 용어들을 만들고 사용하는지 음성 생성의 관점에서 이해를 해보겠습니다.

<br><br>

## 음성 생성의 관점에서 cepstral analysis 이해하기

음성은 기본적으로 발성 기관에서 발생한 파형이 입술 및 구강에서의 변형을 통해 형성됩니다. 발성기관에서 만들어진 목소리가 몸 밖으로 나오기까지 거쳐 가는 공간을 성도(vocal tract)라고 하는데, vocal tract는 고정된 형태가 아니라 유동적으로 변하는 공간이며 그 형태에 따라 소리의 형태가 크게 영향을 받습니다.

<p align="center">
  <img src="https://i.ibb.co/Wfp50xB/vocal-tract.png" alt="vocal tract" border="0">
</p>

음성 생성을 크게 두 가지 과정으로 나누면 먼저 성대가 주기적으로 열리고 닫히면서 glottal pulse가 생성되고, 입술 및 구강에서의 형태 변화를 통해 vocal tract가 filter의 역할을 하며 소리를 변형시킵니다. 이때 glottal pulse의 주기성과 형태는 음성의 음높이(pitch)를 결정하는 기본 주파수에 영향을 미치고 vocal tract filter는 주로 음성의 음색(timbre)과 발음에 큰 영향을 줍니다.

일반적으로 음성의 특성을 결정하는 것은 vocal tract filter에 대한 요소이기 때문에 음성 신호를 분석할 때 glottal pulse와 vocal tract filter 두 요소를 분리해내는 것이 필요합니다. 음성 생성의 과정을 시스템에 대한 자극(excitation)인 glottal pulse $e(t)$과 vocal tract filter response $h(t)$의 convolution으로 표현하면 다음 식과 같습니다.

<br>
<center> $ x(t) = e(t) * h(t) $ </center>
<br>

여기서 $x(t)$는 음성 신호입니다. Convolution theorem에 의하면 time domain에서 두 신호의 convolution은 frequency domain에서 각각의 Fourier transform의 곱과 같습니다. 따라서 frequency domain에서의 식을 표현하면 다음과 같습니다.

<br>
<center> $ X(f) = E(f) \cdot H(f) $ </center>
<br>

양변에 로그를 취해주면 로그 내부의 곱연산이 각각의 로그에 대한 합연산과 같기 때문에 다음 식이 성립합니다.

<br>
<center> $ \log{X(f)} = \log{E(f)} + \log{H(f)} $ </center>
<br>

이것을 그림으로 나타내면 다음과 같습니다.

<p align="center">
  <img src="https://i.ibb.co/MsS8MxQ/speech-separation.png" alt="speech signal separation" border="0">
</p>

Inverse discrete Fourier transform(IDFT)을 적용하면 선형적으로 결합된 glottal pulse와 vocal tract frequency response를 분리할 수 있습니다. Frequency domain의 스펙트럼에 IDFT를 적용하면 time domain으로 돌아가지만, 이 경우에는 로그 스펙트럼에 IDFT를 적용하는 것이기 때문에 time domain과는 구분되는 다른 개념이 필요합니다. 따라서 quefrency domain의 cepstrum이라는 개념이 나오게 된 것입니다. Frequency domain에서 빠르게 변화하는 glottal pulse의 요소는 quefrency domain에서 값이 큰 오른쪽에 피크가 위치하고, frequency domain에서 느리게 변화하는 vocal tract frequency response의 요소는 quefrency domain에서 값이 작은 왼쪽에 피크가 위치합니다.

<p align="center">
  <img src="https://i.ibb.co/VjMzBgv/rhamonic.png" alt="rhamonic in quefrency domain" border="0">
</p>

위의 그림은 0 ms 근처에서 vocal tract frequency response에 해당하는 낮은 quefrency의 피크가 나타나고 3 ms 근처에서 glottal pulse에 해당하는 quefrency의 피크가 나타나는 것을 보여줍니다. 그리고 이 첫 번째 피크를 1st rhamonic이라고 합니다. Frequency domain의 1st harmonic에 대응하는 개념입니다. 이러한 quefrency domain의 cepstrum에서 vocal tract frequency respose의 요소만 추출하기 위해서는 low-pass liftering을 적용해줘야 합니다. 이것은 low-pass filtering에 대응하는 개념입니다.

<br><br>

## MFCC를 생성하는 방법

Cepstrum을 얻는 과정에서 약간의 변형을 통해 MFCC를 생성할 수 있습니다. 신호에 DFT를 적용한 뒤 로그를 취해 로그 스펙트럼을 얻는 것까지는 동일하지만, 이후에 바로 IDFT를 하는 것이 아니라 mel scaling을 해주고 discrete cosine transform(DCT)을 적용하면 MFCC를 얻게 됩니다. IDFT가 아닌 DCT를 사용하는 이유는 MFCC를 얻을 때에는 실수 정보만 있어도 되는데 실수 계산에 대해 DCT가 더 효율적으로 수행될 수 있기 때문입니다. 또한 DCT는 입력 신호를 대부분의 에너지가 몰려 있는 작은 수의 계수로 압축하기 때문에 중요한 정보를 보다 적은 수의 계수로 효과적으로 표현할 수 있게 됩니다.

일반적으로 음성 신호에서는 주로 13개에서 40개 정도의 MFCC 개수를 사용하는 것이 흔하지만 사용되는 데이터의 특성, 응용 분야, 모델의 요구 사항에 따라 조절될 수 있습니다. MFCC는 각각 신호의 다른 특성을 나타내며 첫 번째 계수가 가장 많은 정보를 포함하고 있습니다. 또한 MFCC 시퀀스에서 각 계수의 변화율을 나타내는 delta MFCC와 두 번 미분한 double delta MFCC도 신호의 특성을 나타내는데 사용됩니다.

<br><br>

## MFCC의 시각화

MFCC는 x축이 시간, y축이 계수의 인덱스인 히트맵으로 시각화됩니다. 이때 y축은 맨 아래의 행이 첫 번째 계수이고 맨 위에 있는 행이 마지막 계수입니다. 아래의 그림은 계수의 개수가 13개일 때 음악 신호의 MFCC 스펙트로그램 예시입니다.

<p align="center">
  <img src="https://i.ibb.co/5MYrCsB/mfcc.png" alt="mfcc" border="0">
</p>

## Reference

[[Youtube] Valerio Velardo - The Sound of AI, "Mel-Frequency Cepstral Coefficients Explained Easily"](https://youtu.be/4_SH2nfbQZ8?feature=shared)

[[Course materials] Amrita University, "Cepstral Analysis of Speech" (2011)](https://vlab.amrita.edu/?sub=3&brch=164&sim=615&cnt=1)
