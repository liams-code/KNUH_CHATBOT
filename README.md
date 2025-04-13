# KNUH 칠곡 경북대학교병원 내규 및 노동조합 챗봇

병원 내규와 노동조합 관련 문서를 분석하고 질문에 답변하는 챗봇 시스템입니다.

## 기능

- PDF 문서 업로드 및 분석
- 자연어 질문에 대한 답변
- 관리자 페이지를 통한 문서 관리
- 사용자 친화적인 인터페이스

## 설치 방법

1. 저장소를 클론합니다:
```bash
git clone [repository-url]
cd [repository-name]
```

2. 필요한 패키지를 설치합니다:
```bash
pip install -r requirements.txt
```

3. 환경 변수 설정:
- `.env` 파일을 생성하고 OpenAI API 키를 설정합니다:
```
OPENAI_API_KEY=your-api-key
```

## 실행 방법

```bash
python app.py
```

## 배포 방법 (Hugging Face Spaces)

1. Hugging Face 계정을 만듭니다.
2. 새로운 Space를 생성합니다.
3. GitHub 저장소를 연결하거나 파일을 직접 업로드합니다.
4. Space의 설정에서 환경 변수를 설정합니다.

## 사용 방법

- 사용자 페이지: `/`
- 관리자 페이지: `/admin`

## 라이센스

MIT License "# KNUH_CHATBOT" 
