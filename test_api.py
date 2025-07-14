import requests
from time import sleep

# FastAPI 서버 URL
API_URL = "http://localhost:5000/infer/"

# 테스트용 비디오 파일 경로 (테스트할 비디오 파일을 적절히 지정하세요)
video_path = "resources/video/Djokovic_forehand_slow_motion.mp4"
#video_path = "resources/video/video-for-pose-detection-1-1080x1920-30fps.mp4"
# 비디오 파일을 API에 업로드
with open(video_path, "rb") as video_file:
    files = {'file': video_file}
    try:
        response = requests.post(API_URL, files=files)
        response.raise_for_status()  # Check if the request was successful (status code 200)
    except requests.exceptions.RequestException as e:
        print(f"API 요청 실패. 에러: {e}")
        exit(1)  # Exit if request fails

# API 응답 확인
if response.status_code == 200:
    print("API 요청 성공! Task ID:", response.json().get("task_id", "No task ID found"), "for video:", video_path)
    task_id = response.json().get("task_id")
else:
    print("API 요청 실패. 상태 코드:", response.status_code)
    exit(1)

# # 비동기 작업 상태 확인 (Celery 작업을 확인하려면 일정 시간이 필요할 수 있습니다)
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
