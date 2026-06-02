from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
MODEL_PATH = Path(__file__).with_name("pose_landmarker_full.task")

POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
)


def ensure_model() -> None:
  if MODEL_PATH.exists():
    return

  print(f"Downloading pose model to {MODEL_PATH}...")
  urlretrieve(MODEL_URL, MODEL_PATH)


def create_landmarker():
  base_options = mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH))
  options = mp.tasks.vision.PoseLandmarkerOptions(
      base_options=base_options,
      running_mode=mp.tasks.vision.RunningMode.VIDEO,
      num_poses=1,
      min_pose_detection_confidence=0.5,
      min_pose_presence_confidence=0.5,
      min_tracking_confidence=0.5,
  )
  return mp.tasks.vision.PoseLandmarker.create_from_options(options)


def draw_pose(image, landmarks) -> None:
  height, width = image.shape[:2]
  points = []

  for landmark in landmarks:
    x = int(landmark.x * width)
    y = int(landmark.y * height)
    points.append((x, y))

  for start, end in POSE_CONNECTIONS:
    cv2.line(image, points[start], points[end], (0, 255, 0), 2)

  for point in points:
    cv2.circle(image, point, 4, (0, 0, 255), -1)


def main() -> None:
  ensure_model()

  cap = cv2.VideoCapture(0)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

  if not cap.isOpened():
    raise RuntimeError("Could not open webcam index 0.")

  timestamp_ms = 0

  with create_landmarker() as landmarker:
    while True:
      success, frame = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        continue

      rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
      result = landmarker.detect_for_video(mp_image, timestamp_ms)
      timestamp_ms += 33

      if result.pose_landmarks:
        draw_pose(frame, result.pose_landmarks[0])

      cv2.imshow("MediaPipe Pose", cv2.flip(frame, 1))
      if cv2.waitKey(5
                     ) & 0xFF == 27:
        break

  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()
