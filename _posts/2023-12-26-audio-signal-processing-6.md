---
layout: post
title: "[오디오 신호 처리] 6. 푸리에 변환 (Fourier Transform)"
image: https://i.ibb.co/gwDVvF4/thumbnail.png
date: 2023-12-26
tags: 
categories: audio-signal-processing
use_math: true
---

<br><br>

## 직관적으로 Fourier Transform 이해하기
<!-- excerpt-start -->
주기성을 갖는 신호는 서로 다른 주파수를 갖는 사인파들의 조합으로 이루어져 있습니다. 이러한 사인파 성분들은 각각 크기(magnitude)와 위상(phase)을 가지고 신호를 구성합니다. 먼저 예시를 통해 살펴보겠습니다. 아래 그림은 피아노의 도(C) 음 소리에 대한 파형입니다.

<p align="center">
  <img src="https://i.ibb.co/dLrXz0X/piano-c-waveform.png" alt="piano waveform">
</p>

이 소리 신호는 어떤 사인파들의 조합으로 이루어져 있을까요? 이것을 알기 위해 수행하는 것이 Fourier transform입니다. 직관적인 이해를 위해 먼저 결과부터 보겠습니다.

<p align="center">
  <img src="https://i.ibb.co/SQ3q0yP/piano-c-fft.png" alt="piano sound fft result">
</p>

500 Hz 근처의 fundamental frequency와 그 배음(harmonic partial)에 해당하는 주파수의 사인파 성분들이 있는 것을 알 수 있습니다. 그리고 각각의 주파수 성분들은 magnitude를 갖고 있습니다. 이 magnitude가 높다는 것은 원래의 신호와 높은 유사성을 갖는다는 의미입니다. 527 Hz의 사인파와 원래의 신호를 한 번 비교해보겠습니다.

<p align="center">
  <img src="https://i.ibb.co/xJ2sGY2/sine-wave.png" alt="sine wave and piano wave">
</p>

빨간색의 사인파는 $y = \sin(2 \pi f t)$의 식을 갖는 $f = 527 \, \text{Hz}$의 파동입니다. 원래의 신호와 상당히 유사한 모양을 갖지만 위상이 맞지 않는 것을 볼 수 있습니다. 이제 phase $\varphi$도 고려한 사인파를 생각해보겠습니다. 그 함수는 $y = \sin(2 \pi \cdot (ft - \varphi))$가 됩니다.

<p align="center">
  <img src="https://i.ibb.co/c25KnhL/sine-wave-phase.png" alt="sine wave with phase">
</p>

$\varphi = 0.5$일 때의 결과입니다. 이제 정말로 원래의 신호와 비슷해 보입니다. 따라서 이 신호를 Fourier transform 했을 때 527 Hz의 주파수 성분은 약 0.5의 phase에서 높은 magnitude를 갖게 된다는 것을 알 수 있습니다. Fourier transform의 과정은 이처럼 주파수 성분마다 phase를 최적화(optimize)하고 magnitude를 계산하는 것입니다.

그렇다면 유사한 정도를 정량적으로 어떻게 판단할 수 있을까요? 한 가지 방법은 사인파와 원래 신호를 곱한 그래프가 x축과 이루는 면적을 보는 것입니다. 앞의 예시를 통해 시각적으로 확인해보겠습니다.

<p align="center">
  <img src="https://i.ibb.co/rxjrFFw/integral.png" alt="integral">
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
  <img src="https://i.ibb.co/k3wWS2F/complex-number.png" alt="complex number in polar coordinates">
</p>

복소 평면(complex plane)에서 x축은 실수(real number), y축은 허수(imaginary number)입니다. 그림에서도 볼 수 있듯이 $c$의 크기는 $ \vert c \vert $이며, 실수부와 허수부로 나눠서 표기하면 $c = \vert c \vert \cdot (\cos \gamma + i \sin \gamma)$가 됩니다. 그리고 오일러 공식(Euler formula)에 의해 $ e^{i \gamma} = cos \gamma + i \sin \gamma $이므로 $ c = \vert c \vert \cdot e^{i \gamma} $입니다.

다르게 표현하면 복소수 $ c $는 크기를 나타내는 $ \vert c \vert $와 방향을 나타내는 $ e^{i \gamma} $로 구성되어 있다고 할 수 있습니다. 이제 다시 Fourier transform으로 돌아가보겠습니다. Fourier transform으로 얻어지는 요소인 magnitude와 phase는 각각 크기와 방향의 개념에 해당합니다. 따라서 복소수로 magnitude와 phase를 한 번에 다음과 같이 표현할 수 있습니다.

<br>
<center> $ c_{f} = \frac{d_{f}}{\sqrt{2}} \cdot e^{-i 2 \pi \varphi_{f}} $ </center>
<br>

여기서 $c_{f}$를 Fourier coefficient라고 합니다. $ d_{f} $를 $ \sqrt 2 $로 나눠주는 이유는 Fourier transform으로 얻은 주파수 도메인의 에너지가 시간 도메인에서의 에너지와 같게 보존되도록 정규화를 해주는 것이 필요하기 때문입니다. 그리고 지수항에 마이너스 부호가 붙는 것은 phase를 시계 방향의 각도로 표현한다는 의미입니다.

시간 도메인의 신호 $ g(t) $에 대해 Fourier transform을 적용하면 주파수 도메인의 $ \hat{g}(f) $를 얻게 됩니다. $ g(t) $는 x축의 시간 $t$에 대해 y축의 amplitude 값 $ g(t) $가 대응됩니다. 시간과 amplitude 모두 실수이므로 $ g: \mathbb{R} \rightarrow \mathbb{R} $입니다. 반면 $ \hat{g}(f) $는 x축의 주파수 $f$에 대해 magnitude와 phase의 정보를 모두 담고 있는 Fourier coefficient $ c_{f} $가 대응됩니다. 따라서 $ \hat{g}: \mathbb{R} \rightarrow \mathbb{C} $입니다.

<p align="center">
  <img src="https://i.ibb.co/KwGpmrQ/time-frequency.png" alt="time and frequency domains">
</p>

위의 그림은 실수 평면에 시간 도메인의 신호를 시각화하듯이 극좌표의 복소 평면에 Fourier coefficient들을 시각화한 것입니다. 이와 같이 Fourier transform의 결과로 얻어진 $ \hat{g}(f) = c_{f} $는 복소 평면에서 시각화해보면 그 의미를 이해하는 데에 도움이 될 수 있습니다. 먼저 Fourier transform의 수학적 정의를 보면 다음과 같습니다.

<br>
<center> $ \hat{g}(f) = \int g(t) \cdot e^{-i 2 \pi f t }dt $ </center>
<br>

여기서 뒤의 $ e^{- 2 \pi f t} $ 부분만 보면 $ t $가 커짐에 따라서 복소 평면의 원을 시계 방향으로 회전하는 것과 같습니다. 그리고 주파수 $ f $가 커질수록 회전하는 속도는 더 빨라집니다.

<p align="center">
  <img src="https://i.ibb.co/h1p2cyP/fourier-complex-vis.png" alt="Fourier transform visualization">
</p>

이제 앞의 $ g(t) $까지 곱한 결과는 복소 평면에 어떻게 나타나는지 몇 가지 예시를 통해 살펴보겠습니다. 아래 그림은 각각 주파수가 1, 2, 3 Hz인 사인파들을 합친 신호의 파형을 나타냅니다.

<p align="center">
  <img src="https://i.ibb.co/FWVSHgk/example-signal.png" alt="signal example">
</p>

이 신호 $ g(t) $에 대해 주파수 $ f $를 바꿔가며 $ g(t) \cdot e^{-i 2 \pi f t } $를 극좌표계에 그려보면 다음과 같습니다.

<p align="center">
  <img src="https://i.ibb.co/nDPJjyr/complex-signal.png" alt="signal example in polar coordinates">
</p>

그림을 보면 신호의 구성 주파수인 1, 2 Hz에서는 모양이 아래쪽으로 치우쳐 있는 반면, 그 외의 1.1이나 1.2 Hz 같은 주파수에서는 원점을 중심으로 대칭성이 있는 것을 알 수 있습니다. 따라서 직관적으로 이것을 t에 대해 적분하면 주파수가 1, 2 Hz와 같은 신호의 구성 성분이 아닐 때에는 서로 상쇄되어 0이 될 것이라고 생각됩니다. 적분을 하기 전에 먼저 물리적인 평균에 해당하는 무게중심(center of gravity)의 위치를 확인해보겠습니다.

<p align="center">
  <img src="https://i.ibb.co/sJWBm5w/center-of-gravity.png" alt="center of gravity">
</p>

실제로 주파수가 1.1, 1.2 Hz와 같은 경우에는 원점을 중심으로 서로 상쇄되어 값이 0이 되는 것을 볼 수 있습니다. 이 경우 최종적으로 적분까지 한 결과로 나오는 값인 Fourier coefficient도 역시 0이 될 것입니다. 주파수가 1, 2, 3 Hz인 경우에는 Fourier transform을 통해 위 그림의 원점이 아닌 점과 같이 복소수 값이 나오게 되고, 이를 통해 크기와 각도에 해당하는 magnitude와 phase를 얻을 수 있습니다.

<br><br>

## Inverse Fourier Transform

신호에 Fourier transform을 적용해서 얻은 구성 주파수 성분들의 magnitude와 phase 정보가 있으면 역으로 원래의 신호를 복원하는 것도 가능합니다. 이 과정을 푸리에 역변환(inverse Fourier transform, IFT)이라고 합니다. Inverse Fourier transform의 식은 다음과 같습니다.

<br>
<center> $ g(t) = \int c_{f} \cdot e^{i 2 \pi f t }df $ </center>
<br>

이 식의 의미는 각각의 주파수를 갖는 기본적인 사인파(pure tone)에 magnitude를 곱하고 phase를 더해준 뒤 모두 합쳐주라는 것입니다. 이러한 방식으로 소리를 만드는 것은 기본적인 가산 합성(additive synthesis)의 원리이기도 합니다.

<p align="center">
  <img src="https://i.ibb.co/gJZsT1d/additive-synthesizer.png" alt="additive synthesizer">
</p>

<br><br>

## Discrete Fourier Transform

오디오 신호의 Fourier transform은 실제로는 연속적인 아날로그 신호가 아닌 디지털 신호에 대해 연산이 이루어져야 합니다. 이 경우 무한한 범위의 시간이나 주파수를 처리할 수 없기 때문에 위에서 설명한 Fourier transform 공식의 적분 연산을 그대로 수행하는 것이 불가능합니다. 따라서 시간 도메인의 유한한 샘플 $ x_{n} $을 가지고 주파수 도메인의 유한한 Fourier coefficient $ X_{k} $로 변환시키는 discrete Fourier transform 연산이 필요합니다.

<br>
<center> $ X_{k} = \sum_{n=0}^{N-1} x_{n} \cdot e^{-i 2 \pi (\frac{k}{N}) n} $ </center>
<br>

여기서 $N$은 처리하는 신호의 샘플 개수입니다. 즉, $ x_{n} := x_{0}, x_{1}, \ldots, x_{N-1} $입니다. 그리고 일반적으로 주파수 도메인의 Fourier coefficient도 $ X_{k} := X_{0}, X_{1}, \ldots, X_{N-1} $로 N개의 개수를 갖도록 합니다. 그렇게 하면 Fourier transform과 inverse Fourier transform이 효율적인 양방향의 행렬(matrix) 연산으로 이루어질 수 있기 때문입니다.

또한 주파수의 범위는 일반적으로 $0$부터 sampling rate까지로 설정합니다. 예를 들어 sampling rate가 $22050$ Hz이면 k번째 Fourier coefficient의 주파수는 $\frac{k}{N} \cdot 22050$ Hz가 됩니다.

<p align="center">
  <img src="https://i.ibb.co/0yXydjg/discrete-sinwave.png" alt="discrete sine wave">
</p>

위의 그림은 주파수가 각각 1, 4 Hz인 사인파를 합친 신호의 예시입니다. 이때 sampling rate는 64 Hz입니다. 총 신호의 길이가 2초이므로 샘플의 개수 $N$은 128이 됩니다. 이 신호에 대해 discrete Fourier transform을 적용하면 0부터 64 Hz까지의 주파수 범위에서 128개의 Fourier coefficient가 얻어질 것입니다. Discrete Fourier transform을 적용하여 얻은 복소수의 Fourier coefficient들에 절대값을 취해서 magnitude 값을 계산하면 다음 그림과 같이 나옵니다.

<p align="center">
  <img src="https://i.ibb.co/GdJqs1B/dft-magnitude.png" alt="discrete sine wave">
</p>

결과를 보면 예상대로 1 Hz, 4 Hz에서 magnitude가 높게 나오는데, 오른쪽 끝의 60 Hz와 63 Hz에서도 magnitude 피크가 보입니다. 그리고 이는 32 Hz를 중심으로 대칭적인 모양입니다. Fourier transform의 결과는 이처럼 대칭적인 형태로 나오기 때문에 중복적인 정보를 제외하고 처음 절반의 값들만 사용합니다. 대칭의 중심이 되는 주파수는 sampling rate의 1/2인 Nyquist frequency에 해당합니다. 따라서 정보의 손실 없이 원래의 신호를 복원하려면 반드시 필요한 최고 주파수의 두 배 이상으로 sampling rate를 설정해야 합니다.

<br><br>

## Reference

[[Youtube] Valerio Velardo - The Sound of AI, "Demystifying the Fourier Transform: The Intuition"](https://youtu.be/XQ45IgG6rJ4?feature=shared)

[[Youtube] Valerio Velardo - The Sound of AI, "Complex Numbers for Audio Signal Processing"](https://youtu.be/DgF4m0AWCgA?feature=shared)

[[Youtube] Valerio Velardo - The Sound of AI, "Defining the Fourier Transform with Complex Numbers"](https://youtu.be/KxRmbtJWUzI?feature=shared)

[[Youtube] Valerio Velardo - The Sound of AI, "Discrete Fourier Transform Explained Easily"](https://youtu.be/ZUi_jdOyxIQ?feature=shared)

[[Youtube] 3Blue1Brown, "But what is the Fourier Transform? A visual introduction."](https://youtu.be/spUNpyF58BY?feature=shared)
