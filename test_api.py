import requests
from time import sleep

import requests
import os

import requests
import os

API_URL = "http://localhost:5000/infer/"
video_path = "resources/video/Djokovic_forehand_slow_motion.mp4"

# Memory-safe: file streamed internally by requests
with open(video_path, "rb") as f:
    response = requests.post(API_URL, files={"file": (os.path.basename(video_path), f)})

if response.status_code == 200:
    task_id = response.json().get("task_id")
    print("Task submitted:", task_id)
else:
    print("Failed:", response.status_code, response.text)
    exit(1)


# 비동기 작업 상태 조회용
# task_status_url = f"http://localhost:8000/celery-status/{task_id}"

# # 작업 상태를 주기적으로 확인
# for _ in range(10):
#     sleep(5)  # 5초 대기 후 상태 확인
#     try:
#         task_status_response = requests.get(task_status_url)
#         task_status_response.raise_for_status()  # Check if the request was successful
#         task_status = task_status_response.json()
#         print(f"Task Status: {task_status}")
#     except requests.exceptions.RequestException as e:
#         print(f"작업 상태 요청 실패. 에러: {e}")
#         break
#     except ValueError as e:
#         print("응답에서 JSON을 디코딩할 수 없습니다. 응답 본문:", task_status_response.text)
#         break

#     if task_status.get('status') == 'SUCCESS':
#         print("작업 성공적으로 완료됨!")
#         break
#     elif task_status.get('status') == 'FAILURE':
#         print("작업 실패")
#         break
