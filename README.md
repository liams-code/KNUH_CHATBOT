---
title: KNUH 칠곡 경북대학교병원 규정집 & 노동조합 단체협약서 AI Agent
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.25.1"
app_file: app.py
pinned: false
---

# KNUH 칠곡 경북대학교병원 규정집 & 노동조합 단체협약서 AI Agent

칠곡 경북대학교병원 규정집 & 노동조합 단체협약서를 분석하고 질문에 답변하는 챗봇 시스템입니다.

## 기능

- PDF 문서 업로드 및 분석
- 자연어 질문에 대한 답변
- 관리자 페이지를 통한 문서 관리
- 사용자 친화적인 인터페이스

## 설치 방법

1. 저장소를 클론합니다:
```bash
git clone https://github.com/liams-code/KNUH_CHATBOT.git
cd KNUH_CHATBOT
```

2. 필요한 패키지를 설치합니다:
```bash
pip install -r requirements.txt
```

3. 환경 변수 설정:
- `.env` 파일을 생성하고 OpenAI API 키를 설정합니다:
