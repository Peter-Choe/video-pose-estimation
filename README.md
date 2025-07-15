# 🎯 Video Pose Estimation API

영상 파일을 입력받아 프레임 단위로 분리하고, Triton Inference Server를 통해 2D skeleton keypoints를 추출하는 비동기 기반 포즈 추정 API 시스템입니다.  
FastAPI 서버, Celery 비동기 워커, Redis 브로커, Triton 모델 서버로 구성되며, ONNX 변환용 컨테이너도 함께 제공됩니다.

---

## 🧩 주요 구성 요소

| 구성 요소         | 설명 |
|------------------|------|
| **FastAPI**      | 영상 업로드 및 추론 요청을 처리하는 REST API |
| **Celery + Redis** | 비동기 영상 처리 작업 분산 수행 |
| **Triton Inference Server** | ONNX 모델을 고성능으로 서빙 |
| **ONNX Export 컨테이너** | MMPose 모델을 ONNX 포맷으로 변환하는 도구 제공 |

---

## 📁 프로젝트 구조 요약

```
video-pose-estimation-api/
├── docker-compose.yml
├── docker-compose.local.yml          # Triton + Redis만 실행
├── fastapi_pipeline/
│   ├── Dockerfile
│   ├── app/
│   │   ├── main.py                   # FastAPI 서버 진입점
│   │   ├── tasks.py                  # Celery 비동기 태스크
│   │   ├── infer_client.py           # Triton 추론 요청 모듈
│   │   └── core/                     # 설정, 로깅
├── onnx_export/                      # MMPose → ONNX 변환 도구
│   ├── Dockerfile
│   ├── export_mmpose_to_onnx.py
│   └── models/
├── models/                           # Triton 서빙 모델 디렉토리
├── resources/
│   ├── video/                        # 입력 비디오
│   └── output_video/                # 출력 비디오
├── logs/                             # 추론 로그
├── test_api.py                       # API 테스트 스크립트
└── run_local_inference_visual.py     # 로컬 Triton 테스트 및 시각화
```

---

## 🧪 로컬 개발 환경 실행 방법

### 1. Triton + Redis 컨테이너 실행

```bash
docker compose -f docker-compose.local.yml up
```

> `localhost:8000` (Triton), `localhost:6379` (Redis) 노출됨

### 2. FastAPI 서버 실행 (루트 디렉토리 기준 가상환경에서)

```bash
uvicorn fastapi_pipeline.app.main:app --reload --port 5000
```

### 3. Celery 워커 실행

```bash
celery -A fastapi_pipeline.app.core.celery_app worker --loglevel=info
```

> `.env.local` 또는 `.env.docker` 등을 통해 환경변수 설정이 필요할 수 있습니다.

---

## 🚀 전체 컨테이너로 실행 (배포 또는 통합 테스트용)

```bash
docker compose up --build
```

FastAPI + Celery + Redis + Triton Inference Server가 한 번에 실행됩니다.

---

## 🔄 MMPose → ONNX 변환 방법

### 1. 컨테이너 이미지 빌드

```bash
cd onnx_export
docker build -t mmpose2onnx .
```

### 2. 변환 스크립트 실행

```bash
docker run --rm -v $(pwd):/workspace mmpose2onnx python export_mmpose_to_onnx.py
```

- 변환된 모델: `onnx_export/models/mobilenetv2_pose/model.onnx`
- Triton에서 서빙 시, `models/mobilenetv2_pose/1/model.onnx` 위치에 배치 필요

---

## 📡 API 사용 예시

### ✅ 1. 전체 시스템 실행 후 테스트

```bash
python test_api.py
```

- `resources/video/`의 mp4 영상 업로드
- Celery task 등록 및 task_id 반환
- 결과 로그는 `logs/`에 저장 또는 콘솔 출력 확인 가능

| 로그 파일                          | 설명 |
|----------------------------------|------|
| `logs/video_pose_inference.log` | 추론 단계별 로그 |
| `logs/video_inference_times.log`| 처리 시간 기록 |

---

### ✅ 2. 로컬 Triton 추론 테스트 (GUI 시각화)

```bash
ENV=local python run_local_inference_visual.py
```

- Triton 서버에 직접 추론 요청
- 시각화된 keypoint를 실시간 출력 (`DISPLAY` 환경 필요)
- 결과는 `resources/output_video/`에 mp4 파일로 저장

---

## 🧰 주요 기술 스택

- **Backend**: Python, FastAPI, Celery
- **Inference**: Triton Inference Server, ONNX, Torch
- **Pose Model**: MMPose (Top-down Heatmap 기반 MobileNetV2)
- **Infra**: Docker, Redis, Docker Compose

---

## 📝 라이선스

MIT License
