# 블랙잭 AI 게임 앱

인터랙티브 블랙잭 게임 앱입니다. AI 모델이 실시간으로 Hit/Stay 확률을 예측하여 추천합니다.

## 실행 방법

1. 필요한 패키지 설치:
```bash
uv pip install streamlit
```

또는
```bash
pip install streamlit
```

2. 앱 실행:
```bash
streamlit run blackjack_app.py
```

## 기능

- 🎮 인터랙티브 블랙잭 게임
- 📊 실시간 AI 예측 (Hit 시 버스트 확률, Stay 시 딜러 승리 확률)
- 💡 AI 추천 행동
- 🎯 10덱 사용
- 🔄 여러 모델 선택 가능

## 사용법

1. 사이드바에서 모델 선택
2. "새 게임 시작" 버튼 클릭
3. Hit 또는 Stay 버튼으로 게임 진행
4. 우측 패널에서 AI 예측 확인

## 모델 파일

앱을 실행하기 전에 `main_probabilities.ipynb`를 실행하여 모델을 학습시켜야 합니다.
모델 파일은 `blackjack_model_probabilities_{model_key}.pth` 형식으로 저장됩니다.

