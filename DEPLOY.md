# 배포 순서

## 1. Render 백엔드

Render에서 저장소를 연결하고 `render.yaml`을 사용해 Web Service를 생성합니다.

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- 모델 파일(`models/*.pt`, `models/*.pkl`)이 저장소에 포함되어 있어야 합니다.

배포 후 `https://<render-service>.onrender.com` 주소를 복사합니다.

## 2. Streamlit Community Cloud

Streamlit Community Cloud에서 `app.py`를 메인 파일로 선택해 배포합니다.

앱의 Settings → Secrets에 아래처럼 입력합니다.

```toml
BACKEND_URL = "https://<render-service>.onrender.com"
```

앱은 이 주소의 `/predict` API를 호출합니다. 코드에는 로컬 개발용 기본값으로 `http://localhost:8000`이 남아 있습니다.
