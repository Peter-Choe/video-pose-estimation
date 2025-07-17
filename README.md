# 🎯 Video Pose Estimation

FastAPI, Celery, Redis, Triton 기반의 2D Skeleton 추론 파이프라인입니다.  
실제 동영상에서 keypoint를 추출하고, 병렬 처리 및 GPU 성능을 실험한 결과를 포함합니다.

---
## 🎥 영상 데모

- ▶️ [output_pose.mp4  (15초 슬로우모션 결과)](https://drive.google.com/file/d/1T3WjgtZQ7LvWhscR1ontnD3E2aMBUDIp/view?usp=drive_link)
- ▶️ [output_pose_Djokovic_compressed.mp4 (5분 슬로우모션 결과)](https://drive.google.com/file/d/1BHyoYVqeUC3mfSRxbsDVUL8qJQQEDz94/view?usp=drive_link)
- ▶️ [Djokovic_forehand_slow_motion.mp4 (입력 영상)](https://drive.google.com/file/d/1cu2y3wpreLL0c_2U4F8Jr8hrPD1dJt5F/view?usp=drive_link)

✅ 두 결과 모두 슬로우모션 영상 기반이며,
15초 영상은 배경이 단순하고 인물 중심의 구도로 포즈 추정 정확도가 상대적으로 높게 나타났습니다.
반면, 5분 테니스 영상은 관중석·소품 등 복잡한 배경이 존재하여 일부 keypoint 누락이나 왜곡이 관찰됩니다.
이는 Top-down 기반 MMPose 모델이 배경이 단순할수록 높은 정확도를 보인다는 점을 보여주는 사례입니다.

---


## 🧩 주요 구성 요소

| 구성 요소             | 설명 |
|----------------------|------|
| **FastAPI**          | 영상 업로드 및 추론 요청을 처리하는 REST API |
| **Celery + Redis**   | 비동기 영상 처리 작업을 큐로 분산 실행 |
| **NVIDIA Triton Server**    | ONNX 변환된 MMPose 모델을 고성능으로 추론 서빙 |
| **ONNX Export 컨테이너** | MMPose 학습 모델을 ONNX로 변환 |
| **Docker Compose**   | 전 구성 요소를 컨테이너 기반으로 통합 실행 |

---

## ⚙️ 시스템 구조 및 실행

```
Client → FastAPI → Celery → Triton → 결과 저장 및 시각화
```

- 전체 구성 컨테이너화: `docker compose up --build`
- 로컬 단독 실행도 지원: FastAPI, Celery 별도 실행 가능

---

## 📊 추론 성능 비교 요약

> 테스트 환경: WSL2 + Ubuntu 22.04 + Triton Inference Server  
> GPU: **NVIDIA GeForce RTX 3060 Laptop (6GB VRAM)**  
> ⚠️ 노트북 GPU 성능은 데스크탑 대비 약 30% 낮을 수 있음

| 조건                   | 평균 처리 시간 | GPU 사용률 | 비고                             |
| -------------------- | -------- | ------- | ------------------------------ |
| Non-batch (1개 동영상)   | 약 85.5초  | ~50%   | 기준 성능                          |
| Non-batch (동시 3건 요청) | 약 127.5초 | ~55%   | 요청이 몰리며 다소 지연 발생 (Triton 큐 대기) |
| Batch-16 × 4건 요청     | 약 86.5초  | ~80%   | 성능 회복, GPU 활용도 향상              |
| Batch-32 × 4건 요청     | 약 91.1초  | ~87%   | 배치 준비·큐 대기로 소폭 증가 가능성          |

⚠️ 위 결과는 약 5분 길이의 **슬로우모션 테니스 영상**(`Djokovic_forehand_slow_motion.mp4`) 기준으로 측정되었습니다.

---

## 🧪 Batching 기반 병렬 추론 실험

- **Batch Size** 증가 시 GPU 활용률이 향상되며 처리 성능이 개선됨
- **Celery + Triton** 구조에서 다수 영상 동시 처리 시 효율성 증가
- **GPU 사용률**: Batching 적용 시 최대 87%까지 도달

---

## 🔍 사용 모델 구조: MMPose 기반 MobileNetV2

| 구성       | 설명                                         |
|------------|----------------------------------------------|
| Backbone   | MobileNetV2 – 경량화된 CNN 구조               |
| Head       | HeatmapHead – 17개 관절에 대해 Gaussian Heatmap 예측 |
| 입력 크기  | 192×256                                      |
| 출력 크기  | 48×64 × 17 keypoints                          |

> Gaussian Heatmap은 각 keypoint 좌표를 2D 정규분포로 표현하여 학습 안정성과 정확도를 향상시킵니다.  
> COCO format의 17개 관절을 예측하며, ONNX 변환 후 Triton에서 서빙됩니다.

---

## 🔄 YOLO 연동 확장성

- 현재는 **전체 프레임을 192×256으로 resize**하여 포즈를 추론합니다.
- 사용한 MMPose 모델은 **사람 감지 기능 없이**, 인물이 포함된 이미지 전체를 입력으로 받습니다.
- 이로 인해 **인물이 화면 중심에 위치하지 않거나 배경이 복잡할 경우**, 포즈 추정 정확도가 저하될 수 있습니다.
- 추후 YOLO 등의 detector와 연동하면, **여러 사람의 bounding box를 자동 추출**하여 인물 중심 crop 후 개별 포즈를 정확히 추론할 수 있습니다.
- 이는 **실시간 행동 인식**, **AI 광고 분석**, **멀티 인물 트래킹** 시스템 등으로의 확장에 활용될 수 있습니다.

---

## 📁 프로젝트 구조 요약

```
video-pose-estimation-api/
├── docker-compose.yml
├── fastapi_pipeline/
│   ├── app/
│   │   ├── main.py
│   │   ├── tasks.py
│   │   ├── infer_client.py
│   │   └── core/
├── onnx_export/
│   ├── export_mmpose_to_onnx.py
│   └── models/
├── models/
├── resources/video/
├── resources/output_video/
├── logs/
├── test_api.py
└── run_local_inference_visual.py
```

---

## 🧪 로컬 개발 환경 실행 방법

### 1. Triton + Redis 컨테이너 실행
```bash
docker compose -f docker-compose.local.yml up
```

### 2. FastAPI 서버 실행
```bash
uvicorn fastapi_pipeline.app.main:app --reload --port 5000
```

### 3. Celery 워커 실행
```bash
celery -A fastapi_pipeline.app.core.celery_app worker --loglevel=info
```

> `.env.local` 또는 `.env.docker`를 통한 환경변수 설정 필요

---

## 🚀 전체 컨테이너로 실행

```bash
docker compose up --build
```

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
- Triton에서 서빙 시: `models/mobilenetv2_pose/1/model.onnx`

---

## 📡 API 사용 예시

### 1. 전체 시스템 실행 후 테스트
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

### 2. 로컬 Triton 추론 테스트 (GUI 시각화)

```bash
ENV=local python run_local_inference_visual.py
```

- Triton 서버에 직접 추론 요청
- 시각화된 keypoint를 실시간 출력 (`DISPLAY` 환경 필요)
- 결과는 `resources/output_video/`에 mp4 파일로 저장

---
## ⚠️ 모델 학습 및 성능 관련 참고

- 본 프로젝트는 MMPose에서 제공하는 **사전 학습 모델(weight)**을 사용하여 추론 파이프라인을 구성하였습니다.
- **Fine-tuning**이나 **정량적 평가**는 수행하지 않았으며, 입력 품질에 따라 성능이 달라질 수 있습니다.
- 향후 개선 방향으로는 사용자 데이터 기반 학습, keypoint 기반 동작 분류 모델 연동, OKS 등 평가 지표 적용 등이 있습니다.

---

## 🧰 기술 스택 요약

- **Backend**: Python, FastAPI, Celery
- **Inference**: Triton Inference Server, ONNX, Torch
- **Pose Model**: MMPose (Top-down, MobileNetV2)
- **Infra**: Docker, Redis, Docker Compose
- **테스트 환경**: WSL2 + **RTX 3060 Laptop (6GB VRAM)**
