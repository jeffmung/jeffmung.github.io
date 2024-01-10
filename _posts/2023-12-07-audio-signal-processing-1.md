---
layout: post
title: "[오디오 신호 처리] 1. 소리의 구성 요소"
image: "https://i.ibb.co/Q61k2dd/thumbnail.png"
date: 2023-12-07
tags:
categories: Audio-Signal-Processing
use_math: true
---

<br><br>

## 소리란 무엇인가?

물리학적인 측면에서 소리는 공기, 물, 또는 고체와 같은 매질을 통해 음향파로 전파되는 진동으로 정의됩니다. 진동의 결과로 발생하는 압력 변화를 우리의 귀가 감지하고 뇌에서 그것을 소리라고 인식합니다. 따라서 소리의 특성을 알기 위해서는 소리를 파동의 형태로 이해하고 그 구성 요소를 파악하는 것이 중요합니다.

<p align="center">
  <img src="https://i.ibb.co/kQTWsmg/soundwave.png" alt="sound wave">
</p>

<br><br>

## 소리의 유형

가장 간단한 소리는 단일 사인파 형태를 가집니다. 하지만 일상에서 우리가 듣는 소리들은 이러한 단순한 하나의 사인파가 아닙니다. 이러한 소리의 진동은 주기적으로 반복되는 형태일 수도 있고 불규칙할 수도 있습니다.

주기적인 진동의 소리는 여러 개의 사인파들이 더해진 것으로 구성될 수 있습니다. 다시 말해 어떠한 주기적인 파형도 간단한 사인파들의 조합으로 분석될 수 있습니다. 반면, 주기적이지 않은 형태의 소리도 존재하는데 바람에 흔들리는 물체와 같이 무작위적이거나 두 물체가 충돌하는 경우와 같이 진동이 한 번만 발생하는 경우의 소리가 이에 해당합니다. 따라서 주기적이지 않은 파형에는 연속적인 비주기적 파형(Noise)과 일시적인 비주기적 파형(Pulse) 두 가지가 있습니다.

<p align="center">
  <img src="https://i.ibb.co/2j40f8B/waveforms.png" alt="waveforms">
</p>

<br><br>

## 주파수 (Frequency)

주파수는 초당 진동수를 나타내며 소리의 높낮이를 표현합니다. Hz 단위로 측정되는 주파수가 클수록 높은 소리, 작을수록 낮은 소리를 나타냅니다.

인간의 청각 체계는 주로 20 Hz에서 20,000 Hz(또는 20 kHz) 범위 내의 주파수를 감지할 수 있습니다. 이 범위를 초과하는 소리는 초음파로 분류되고 인간은 이를 인지하지 못합니다. 반면에 20 Hz 미만의 주파수는 저주파로 분류되며, 이 또한 인간의 귀로는 듣기 어렵습니다.

<br><br>

## 파장 (Wavelength)

소리의 파장은 공기 중에서의 한 주기 진동의 길이입니다. 따라서 파장의 크기는 공기 중에서의 소리의 속도(344 m/s)를 주파수로 나눈 값과 같습니다.

예를 들어, 가청 주파수 중 가장 높은 20,000 Hz의 소리의 파장을 계산해보면 약 1.7 cm입니다. 즉, 만약 어떠한 스피커로 20,000 Hz의 고음을 만드려고 한다면 1.7 cm의 작은 크기의 공기를 움직일 수 있는 힘만 있으면 됩니다. 큰 에너지보다는 작은 드라이버로 빠르게 움직이게 하는 것이 더 필요한 것입니다. 그리고 수 cm의 장애물만으로도 소리가 전파되는 것을 막을 수 있습니다.

반대로 가청 주파수 중 가장 낮은 20 Hz의 소리의 파장은 약 17.2 m입니다. 이러한 초저음을 만들기 위해서는 17.2 m의 공기를 움직일 수 있는 큰 에너지가 필요합니다. 따라서 이러한 저음이 재생될 때에는 물체가 떨리는 현상이 나타나는 것입니다. 그렇다면 공간의 크기가 17.2 m보다 작을 때에는 어떻게 될까요? 이럴 때에는 소리가 퍼져 나가는 것이 아니라 공간 전체를 흔들기 때문에 저음의 소리가 훨씬 잘들리게 됩니다. 마찬가지로 우리가 작은 이어폰으로 재생되는 저음을 잘 들을 수 있는 이유도 이어폰과 고막 사이의 밀폐된 귀 내부 공간이 매우 작기 때문입니다.

<br><br>

## 진폭 (Amplitude), 강도 (Intensity), 음량 (Loudness)

소리의 진폭과 강도, 음량은 모두 소리의 크기와 관련이 있습니다. 진폭은 공기와 같은 매질이 안정 상태일 때부터 최대 변위까지 움직였을 때의 크기를 말합니다. 따라서 진폭이 클수록 소리의 압력이 크고 소리의 크기도 큽니다.

<p align="center">
  <img src="https://i.ibb.co/k3VHgBC/amplitude.png" alt="amplitude">
</p>


소리의 강도는 단위면적당 단위시간에 전달되는 에너지(단위면적당 파워)를 말하고 W/m<sup>2</sup> 단위로 표시됩니다. 인간이 일반적으로 들을 수 있는 가장 작은 소리의 강도는 약 10<sup>-12</sup> W/m<sup>2</sup>입니다. 그리고 인간이 고통을 느끼기 시작하는 강도는 10 W/m<sup>2</sup>입니다. 즉, 인간은 굉장히 작은 강도의 소리를 인지할 수 있고, 인간이 듣는 소리의 강도는 매우 큰 범위에 걸쳐 있습니다. 따라서 소리의 강도를 표현할 때에는 보통 로그 스케일로 표현하고 절대적인 크기가 아닌 상대적인 크기로 나타냅니다. 이러한 소리의 강도를 표현하는 단위가 dB입니다. dB의 정의는 들을 수 있는 가장 작은 소리(Threshold of hearing, TOH)를 기준으로 다음의 식과 같습니다.

<br>
<center> $dB(I_{TOH}) = 10 \cdot \log_{10}(\frac{I}{I_{TOH}})$ </center>
<br>

$I$에 $I_{TOH}$를 넣으면 0이 되므로 인간이 들을 수 있는 가장 작은 소리의 강도인 10<sup>-12</sup> W/m<sup>2</sup>가 0 dB이 됩니다. dB은 로그 스케일이기 때문에 아래 표와 같이 소리의 강도가 10배가 되면 10 dB씩 증가합니다.

<p align="center">
  <img src="https://i.ibb.co/PxKpDjY/db.png" alt="db">
</p>

강도와 달리 음량(Loudness)은 소리의 크기를 인지하는 주관적인 수치입니다. 음량은 소리의 지속시간, 주파수에 따라 달라지며 듣는 사람의 나이에 따라서도 달라지는 값입니다. 음량의 단위로는 phon을 사용합니다.

<p align="center">
  <img src="https://i.ibb.co/g66K3d0/equal-loudness.png" alt="equal loudness">
</p>

위의 그래프는 동일한 음량을 갖는 주파수와 강도를 선으로 표시한 것입니다. 같은 dB일 때 저주파와 고주파보다는 1 kHz 부근에서 더 큰 음량을 나타내는 것을 볼 수 있습니다. 예를 들어 60 dB일 때 50 Hz 소리의 음량은 약 20 phon이지만 1 kHz 소리의 음량은 약 60 phon입니다. 인간은 수백에서 수천 Hz 근처의 범위에서 소리를 가장 잘 인지할 수 있기 때문입니다.

<br><br>

## 위상 (Phase)

같은 주파수를 갖는 두 음파가 있을 때 같은 변위를 갖는 시작점이 어떤 시간만큼 차이를 갖는 것을 위상이 다르다고 표현합니다. 위상은 절대적인 위치가 아니라 상대적인 차이만을 나타내는 개념입니다. 일반적으로 위상의 차이를 각도로 표현하는데, 한 주기를 $2\pi(360^\circ)$로 하여 1/2주기는 $\pi(180^\circ)$가 됩니다.

<p align="center">
  <img src="https://i.ibb.co/xLQsNtM/phase.png" alt="phase">
</p>

<br><br>

## 음색 (Timbre)

음색은 같은 강도와 주파수, 길이를 갖는 두 소리를 다른 악기로 연주했을 때 구분될 수 있도록 하는 특성입니다. 음색은 주파수나 진폭 등과 같이 한 가지의 물리량이 아닌 여러 차원의 요소들이 결합하여 결정하게 됩니다.

### Envelope

소리의 envelope는 시간에 따른 소리의 변화를 설명합니다. 가장 일반적인 envelope 모델에는 attack, decay, sustain, release의 네 가지 요소가 있으며 이를 ADSR이라고 부릅니다.

<p align="center">
  <img src="https://i.ibb.co/6ndPHrQ/envelope.jpg" alt="envelope">
</p>

예를 들어, 피아노 건반을 치는 소리는 attack이 매우 짧지만 바이올린과 같은 현악기의 소리는 attack이 긴 envelope를 갖습니다. 특히 인간이 소리를 구분할 때 가장 큰 영향을 주는 부분은 attack입니다. 따라서 attack을 ID of sound라고도 합니다.

### Harmonic Content

하나의 주파수만을 갖는 사인파를 simple wave라고 합니다. 하지만 우리가 듣는 악기의 소리나 자연에서의 소리 등은 모두 이러한 simple waveform이 아닌 complex wave입니다. Complex wave는 서로 다른 주파수를 갖는 사인파들의 조합으로 만들어지며 각각의 사인파들을 partial이라고 합니다.

이 중 그 소리의 음을 나타내는 가장 낮은 주파수를 fundamental frequency라고 하며, 이것의 정수 배가 되는 주파수의 사인파들을 harmonic partial이라고 합니다. 그리고 정수 배가 아닌 주파수의 사인파들은 inharmonic partial이라고 합니다. 그리고 fundamental frequency 위에 존재하는 모든 주파수들을 합쳐서 overtone이라고 합니다.

예를 들어 아래의 그림은 피아노로 라(A4) 음을 쳤을 때의 주파수 스펙트럼입니다. Fundamental frequency인 440 Hz 위로 정수 배인 880 Hz, 1320 Hz 등의 harmonic partial 피크가 크게 나타나고 그 외의 inharmonic partial들도 존재하는 것을 볼 수 있습니다.

<br>
<p align="center">
  <img src="https://i.ibb.co/HFYTQtT/partial.png" alt="harmonic contents">
</p>
<br>

### Modulation

Frequency나 amplitude의 modulation도 음색을 결정하는 요소들 중 하나입니다. Frequency modulation은 현악기의 비브라토와 같이 주파수를 위아래로 빠르게 움직이는 것입니다. 또한 amplitude modulation은 볼륨이 빠르게 커졌다 작아졌다 하면서 독특한 음색을 만들어냅니다.

<p align="center">
  <img src="https://i.ibb.co/HLvKdxW/modulation.png" alt="modulation">
</p>

<br><br>

## Reference

[[Youtube] Valerio Velardo - The Sound of AI, "Sound and Waveforms"](https://youtu.be/bnHHVo3j124?feature=shared)
<br>
[[Youtube] Valerio Velardo - The Sound of AI, "Intensity, Loudness, and Timbre"](https://youtu.be/Jkoysm1fHUw?feature=shared)
<br>
[[Youtube] 김도현 대림대교수, "소리의 6 요소! - 주파수, 파장, 진폭, 위상, 배음, 속도"](https://youtu.be/RGbsTdCQR6U?feature=shared)