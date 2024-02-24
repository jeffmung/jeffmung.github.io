---
layout: post
title: "[논문 리뷰] MIDI-DDSP: Detailed Control of Musical Performance via Hierarchical Modeling"
image: https://i.ibb.co/gD3dznW/thumbnail.png
date: 2024-02-23
tags: 
categories: Paper-Review
use_math: true
---

<br><br>

## 논문 개요

MIDI-DDSP는 DDSP를 [(Jesse Engel et al., 2020)](https://openreview.net/forum?id=B1x1ma4tDr) 기반으로 미디 노트 시퀀스로부터 다양한 악기 소리와 연주의 표현을 반영한 실제에 가까운 음악을 생성해내는 모델입니다. DDSP의 장점인 음높이, 음량 등의 특징들을 미분 가능한 연산들로 오디오 합성에 활용하여 사용자가 제어할 수 있도록 한다는 것에 더해서 비브라토, 크레센도 등의 연주 표현에 관련된 특징들도 제어하는 것이 가능합니다. 논문에서는 아래 그림과 같이 기존 연구들이 제어 가능성과 자연스러운 소리 둘 중 한 가지에 주로 집중했다면 MIDI-DDSP는 두 가지 관점 모두에서 높은 성능을 보여준다는 것을 강조합니다.

<p align="center">
<img src="https://i.ibb.co/ykcWzzB/mididdsp-strengths.png" alt="mididdsp-strengths" border="0">
</p>

이러한 강점을 나타낼 수 있도록 MIDI-DDSP 모델은 계층적인 구조로 설계되어 있습니다. 가장 아래 모델은 DDSP로 오디오로부터 음높이, 진폭, 하모닉 분포, 노이즈 등의 소리 합성에 필요한 파라미터들을 추론하고 반대로 다시 소리를 합성하는 오토인코더입니다. 중간 레벨에는 합성 파라미터와 연주 표현 특징들 사이에서 추출과 생성을 학습하는 모델이 있습니다. 가장 위 레벨에는 음표(note)로부터 연주 표현을 생성해내는 방법을 학습하는 모델이 있습니다. 이러한 계층 구조는 아래 그림에 표현되어 있습니다.

<p align="center">
<img src="https://i.ibb.co/b7LRGLm/hierearchy.png" alt="hierearchy" border="0">
</p>

<br><br>

## DDSP Synthesizer

