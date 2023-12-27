---
layout: post
title: "[오디오 신호 처리] 6. 푸리에 변환 (Fourier Transform)"
image: https://drive.google.com/uc?export=view&id=1-PEO2yJXH0DAgiVeaDC8wd2Imk1N8ho4
date: 2023-12-26
tags: 
categories: Audio-Signal-Processing
use_math: true
---

<br><br>

## 직관적으로 Fourier Transform 이해하기

주기성을 갖는 신호는 서로 다른 주파수를 갖는 사인파들의 조합으로 이루어져 있습니다. 이러한 사인파 성분들은 각각 크기(magnitude)와 위상(phase)을 가지고 신호를 구성합니다. 먼저 예시를 통해 살펴보겠습니다. 아래 그림은 피아노의 도(C) 음 소리에 대한 파형입니다.

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1xvZAoaMeTEUDDZ9Rait_ASqZxlSPAmLZ" alt="piano waveform">
</p>

이 소리 신호는 어떤 사인파들의 조합으로 이루어져 있을까요? 이것을 알기 위해 수행하는 것이 Fourier transform입니다. 직관적인 이해를 위해 먼저 결과부터 보겠습니다.

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1Ix6JwdvcAdOBYX4Xewd43zUT44FfE4WT" alt="piano sound fft result">
</p>

500 Hz 근처의 fundamental frequency와 그 배음(harmonic partial)에 해당하는 주파수의 사인파 성분들이 있는 것을 알 수 있습니다. 그리고 각각의 주파수 성분들은 magnitude를 갖고 있습니다. 이 magnitude가 높다는 것은 원래의 신호와 높은 유사성을 갖는다는 의미입니다. 527 Hz의 사인파와 원래의 신호를 한 번 비교해보겠습니다.

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1QAG9RUcamuDPXv1aIOebVwvv__yAjoNw" alt="sine wave and piano wave">
</p>

빨간색의 사인파는 $y = \sin(2 \pi f t)$의 식을 갖는 $f = 527 \, \text{Hz}$의 파동입니다. 원래의 신호와 상당히 유사한 모양을 갖지만 위상이 맞지 않는 것을 볼 수 있습니다. 이제 phase $\varphi$도 고려한 사인파를 생각해보겠습니다. 그 함수는 $y = \sin(2 \pi \cdot (ft - \varphi))$가 됩니다.

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1cW2wxfILsF3s-sJJODICY2UuZYyrbRXc" alt="sine wave with phase">
</p>

$\varphi = 0.5$일 때의 결과입니다. 이제 정말로 원래의 신호와 비슷해 보입니다. 따라서 이 신호를 Fourier transform 했을 때 527 Hz의 주파수 성분은 약 0.5의 phase에서 높은 magnitude를 갖게 된다는 것을 알 수 있습니다. Fourier transform의 과정은 이처럼 주파수 성분마다 phase를 최적화(optimize)하고 magnitude를 계산하는 것입니다.

그렇다면 유사한 정도를 정량적으로 어떻게 판단할 수 있을까요? 한 가지 방법은 사인파와 원래 신호를 곱한 그래프가 x축과 이루는 면적을 보는 것입니다. 앞의 예시를 통해 시각적으로 확인해보겠습니다.

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1GWoj4Zxj6TjfksrFw7iFhprnibK4a17O" alt="sine wave with phase">
</p>

주황색 부분은 x축과 $y = s(t) \cdot \sin(2 \pi \cdot (ft - \varphi))$ 사이를 칠한 것입니다. 여기서 $s(t)$는 원래 신호입니다. 원래 신호와 유사한 $\varphi = 0.5$의 사인파는 주황색 부분의 면적이 양수로 크고, $\varphi = 0$의 사인파에 대해서는 주황색 부분이 작아서 음의 방향으로 있는 것을 볼 수 있습니다. 이를 수식으로 나타내면 어떤 주파수 $f$에 대한 최적의 phase $\varphi_{f}$는 다음과 같이 얻을 수 있습니다.

<br>
<center> $ \varphi_{f} = \arg\max_{\varphi \in [0, 1)} \left( \int s(t) \cdot \sin(2 \pi (ft - \varphi)) dt \right) $ </center>
<br>

마찬가지로 주파수 $f$에 대한 magnitude $d_{f}$의 값은 다음과 같습니다.

<br>
<center> $ d_{f} = \max_{\varphi \in [0, 1)} \left( \int s(t) \cdot \sin(2 \pi (ft - \varphi)) dt \right) $ </center>
<br>

<br><br>

## 복소수를 사용한 Fourier transform

위에서 Fourier transform을 직관적으로 이해하기는 했지만 엄밀하게 정의한 것은 아닙니다. Fourier transform을 수학적으로 정의하기 위해서는 복소수를 사용하는데, Fourier transform으로 얻어지는 두 요소인 phase와 magnitude를 표현하기에 복소수가 매우 편리하기 때문입니다. 먼저 복소수에 대한 이해를 위해 극좌표(polar coordinates)에서 시각적으로 복소수 $c$를 표현해보겠습니다.

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1UBe3jjEQ1i4Vd5Eh-awXn5G2CB4COx1Z" alt="complex number in polar coordinates">
</p>

복소 평면(complex plane)에서 x축은 실수(real number), y축은 허수(imaginary number)입니다. 그림에서도 볼 수 있듯이 $c$의 크기는 $ \vert c \vert $이며, 실수부와 허수부로 나눠서 표기하면 $c = \vert c \vert \cdot (\cos \gamma + i \sin \gamma)$가 됩니다. 그리고 오일러 공식(Euler formula)에 의해 $ e^{i \gamma} = cos \gamma + i \sin \gamma $이므로 $ c = \vert c \vert \cdot e^{i \gamma} $입니다.

다르게 표현하면 복소수 $ c $는 크기를 나타내는 $ \vert c \vert $와 방향을 나타내는 $ e^{i \gamma} $로 구성되어 있다고 할 수 있습니다. 이제 다시 Fourier transform으로 돌아가보겠습니다. Fourier transform으로 얻어지는 요소인 magnitude와 phase는 각각 크기와 방향의 개념에 해당합니다. 따라서 복소수로 magnitude와 phase를 한 번에 다음과 같이 표현할 수 있습니다.

<br>
<center> $ c_{f} = \frac{d_{f}}{\sqrt{2}} \cdot e^{-i 2 \pi \varphi_{f}} $ </center>
<br>

여기서 $c_{f}$를 Fourier coefficient라고 합니다. $ d_{f} $를 $ \sqrt 2 $로 나눠주는 이유는 Fourier transform으로 얻은 주파수 도메인의 에너지가 시간 도메인에서의 에너지와 같게 보존되도록 정규화를 해주는 것이 필요하기 때문입니다. 그리고 지수항에 마이너스 부호가 붙는 것은 phase를 시계 방향의 각도로 표현한다는 의미입니다.

시간 도메인의 신호 $ g(t) $에 대해 Fourier transform을 적용하면 주파수 도메인의 $ \hat{g}(f) $를 얻게 됩니다. $ g(t) $는 x축의 시간 $t$에 대해 y축의 amplitude 값 $ g(t) $가 대응됩니다. 시간과 amplitude 모두 실수이므로 $ g: \mathbb{R} \rightarrow \mathbb{R} $입니다. 반면 $ \hat{g}(f) $는 x축의 주파수 $f$에 대해 magnitude와 phase의 정보를 모두 담고 있는 Fourier coefficient $ c_{f} $가 대응됩니다. 따라서 $ \hat{g}: \mathbb{R} \rightarrow \mathbb{C} $입니다.

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1oMdnzjaC9q4mVrtwzw8IPhShIRopNG0M" alt="time and frequency domains">
</p>

위의 그림은 실수 평면에 시간 도메인의 신호를 시각화하듯이 극좌표의 복소 평면에 Fourier coefficient들을 시각화한 것입니다. 이와 같이 Fourier transform의 결과로 얻어진 $ \hat{g}(f) = c_{f} $는 복소 평면에서 시각화해보면 그 의미를 이해하는 데에 도움이 될 수 있습니다. 먼저 Fourier transform의 수학적 정의를 보면 다음과 같습니다.

<br>
<center> $ \hat{g}(f) = \int g(t) \cdot e^{-i 2 \pi f t }dt $ </center>
<br>

여기서 뒤의 $ e^{- 2 \pi f t} $ 부분만 보면 $ t $가 커짐에 따라서 복소 평면의 원을 시계 방향으로 회전하는 것과 같습니다. 그리고 주파수 $ f $가 커질수록 회전하는 속도는 더 빨라집니다.

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=18vvXJhjTHcjyQjNbAXcH1erxnZ3ZB89l" alt="Fourier transform visualization">
</p>

이제 앞의 $ g(t) $까지 곱한 결과는 복소 평면에 어떻게 나타나는지 몇 가지 예시를 통해 살펴보겠습니다. 아래 그림은 각각 주파수가 1, 2, 3 Hz인 사인파들을 합친 신호의 파형을 나타냅니다.

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=17egyiVwJiGHcEkHbkHg_VCW4AaHyn4M_" alt="signal example">
</p>

이 신호 $ g(t) $에 대해 주파수 $ f $를 바꿔가며 $ g(t) \cdot e^{-i 2 \pi f t } $를 극좌표계에 그려보면 다음과 같습니다.

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=18YKA-O0cYJr-XI1yM94uMdsJYBicPCBl" alt="signal example">
</p>