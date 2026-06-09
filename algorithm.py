from pathlib import Path
from urllib.request import urlretrieve
import time

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
        if start < len(points) and end < len(points):
            cv2.line(image, points[start], points[end], (0, 255, 0), 2)

    for point in points:
        cv2.circle(image, point, 4, (0, 0, 255), -1)


def get_shoulder_y(landmarks):
    """
    Tính vị trí y trung bình của đường nối 2 vai.
    Landmark:
    11 = left shoulder
    12 = right shoulder
    """
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    return (left_shoulder.y + right_shoulder.y) / 2


def get_shoulder_width(landmarks):
    """
    Tính khoảng cách ngang giữa 2 vai (landmark 11 và 12).
    Khi người dùng ngả ra sau, chiều rộng vai trên khung hình giảm
    → shoulder_width_ratio < 1.
    """
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    return abs(left_shoulder.x - right_shoulder.x)


def get_face_width(landmarks):
    """
    Tính khoảng cách ngang giữa 2 tai:
      landmark 7 = right ear
      landmark 8 = left ear
    Khi người dùng ngả ra sau, chiều rộng khuôn mặt trên khung hình giảm
    → face_width_ratio < 1.
    """
    right_ear = landmarks[7]
    left_ear = landmarks[8]
    return abs(right_ear.x - left_ear.x)


def get_nose_to_shoulder_distance(landmarks):
    nose = landmarks[0]
    shoulder_y = get_shoulder_y(landmarks)
    return abs(nose.y - shoulder_y)


def get_shoulder_points(frame, landmarks):
    """
    Trả về tọa độ pixel của 2 vai để vẽ đường baseline và current shoulder line.
    """
    height, width = frame.shape[:2]

    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]

    left_point = (
        int(left_shoulder.x * width),
        int(left_shoulder.y * height)
    )

    right_point = (
        int(right_shoulder.x * width),
        int(right_shoulder.y * height)
    )

    return left_point, right_point


def get_nose_point(frame, landmarks):
    height, width = frame.shape[:2]
    nose = landmarks[0]
    return (int(nose.x * width), int(nose.y * height))


def classify_posture(
    current_shoulder_y,
    baseline_shoulder_y,
    current_nose_to_shoulder,
    baseline_nose_to_shoulder,
    current_shoulder_width=None,
    baseline_shoulder_width=None,
    current_face_width=None,
    baseline_face_width=None,
):
    """
    Classify the user's posture.

    Lean-backward detection:
      - shoulder_width_ratio = current_shoulder_width / baseline_shoulder_width
      - face_width_ratio      = current_face_width      / baseline_face_width
      When the user leans backward they move away from the camera, so both
      projected widths shrink → ratios drop below 1.

    Forward-hunch detection:
      - shoulder_drop  : how much the shoulders have fallen (normalised y)
      - nose_ratio     : current nose-to-shoulder distance / baseline
    """
    shoulder_drop = current_shoulder_y - baseline_shoulder_y

    if baseline_nose_to_shoulder == 0:
        nose_ratio = 1.0
    else:
        nose_ratio = current_nose_to_shoulder / baseline_nose_to_shoulder

    # --- Width ratios ---
    if baseline_shoulder_width and baseline_shoulder_width > 0:
        shoulder_width_ratio = current_shoulder_width / baseline_shoulder_width
    else:
        shoulder_width_ratio = 1.0

    if baseline_face_width and baseline_face_width > 0:
        face_width_ratio = current_face_width / baseline_face_width
    else:
        face_width_ratio = 1.0

    # --- Lean-backward thresholds ---
    # Strong: both shoulders AND face appear significantly narrower
    lean_backward_strong = (
        shoulder_width_ratio < 0.82 and face_width_ratio < 0.82
    )
    # Mild: either one is somewhat narrower
    lean_backward_mild = (
        shoulder_width_ratio < 0.90 or face_width_ratio < 0.90
    )

    # --- Forward-hunch thresholds ---
    shoulder_dropped_strong = shoulder_drop > 0.06
    shoulder_dropped_mild = shoulder_drop > 0.035
    nose_close_strong = nose_ratio < 0.75
    nose_close_mild = nose_ratio < 0.85

    if lean_backward_strong:
        status = "Leaning backward"
        color = (0, 128, 255)   # orange in BGR
    elif lean_backward_mild:
        status = "Slightly leaning backward"
        color = (0, 200, 255)
    elif shoulder_dropped_strong and nose_close_strong:
        status = "Likely hunch/slouching"
        color = (0, 0, 255)
    elif shoulder_dropped_strong:
        status = "Likely hunch: shoulders dropped"
        color = (0, 0, 255)
    elif nose_close_strong:
        status = "Likely hunch: nose close to shoulder"
        color = (0, 0, 255)
    elif shoulder_dropped_mild or nose_close_mild:
        status = "Mild slouching"
        color = (0, 255, 255)
    else:
        status = "Good posture"
        color = (0, 255, 0)

    return status, shoulder_drop, nose_ratio, shoulder_width_ratio, face_width_ratio, color


def draw_baseline(frame, baseline_shoulder_y):
    """
    Vẽ đường baseline ngang qua vị trí vai chuẩn.
    """
    height, width = frame.shape[:2]
    baseline_y_pixel = int(baseline_shoulder_y * height)

    cv2.line(
        frame,
        (0, baseline_y_pixel),
        (width, baseline_y_pixel),
        (255, 0, 0),
        2
    )

    cv2.putText(
        frame,
        "Baseline shoulder line",
        (30, baseline_y_pixel - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2
    )


def main() -> None:
    ensure_model()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam index 0.")

    timestamp_ms = 0

    # Calibration settings
    calibration_seconds = 3
    calibration_start_time = time.time()
    calibration_shoulder_values = []
    calibration_nose_shoulder_values = []
    calibration_shoulder_width_values = []
    calibration_face_width_values = []
    baseline_shoulder_y = None
    baseline_nose_to_shoulder = None
    baseline_shoulder_width = None
    baseline_face_width = None

    with create_landmarker() as landmarker:
        while True:
            success, frame = cap.read()

            if not success:
                print("Ignoring empty camera frame.")
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            timestamp_ms += 33

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                draw_pose(frame, landmarks)

                current_shoulder_y = get_shoulder_y(landmarks)
                current_nose_to_shoulder = get_nose_to_shoulder_distance(landmarks)
                current_shoulder_width = get_shoulder_width(landmarks)
                current_face_width = get_face_width(landmarks)
                left_shoulder_point, right_shoulder_point = get_shoulder_points(frame, landmarks)
                nose_point = get_nose_point(frame, landmarks)

                # Vẽ đường nối 2 vai hiện tại
                cv2.line(
                    frame,
                    left_shoulder_point,
                    right_shoulder_point,
                    (255, 255, 0),
                    3
                )
                cv2.circle(frame, nose_point, 7, (255, 0, 255), -1)

                # Giai đoạn calibration
                if baseline_shoulder_y is None or baseline_nose_to_shoulder is None:
                    elapsed_time = time.time() - calibration_start_time

                    calibration_shoulder_values.append(current_shoulder_y)
                    calibration_nose_shoulder_values.append(current_nose_to_shoulder)
                    calibration_shoulder_width_values.append(current_shoulder_width)
                    calibration_face_width_values.append(current_face_width)

                    cv2.putText(
                        frame,
                        "Sit straight for calibration...",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 255),
                        2
                    )
                    cv2.putText(
                        frame,
                        f"Calibrating: {calibration_seconds - int(elapsed_time)}s",
                        (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2
                    )

                    if elapsed_time >= calibration_seconds:
                        baseline_shoulder_y = sum(calibration_shoulder_values) / len(
                            calibration_shoulder_values
                        )
                        baseline_nose_to_shoulder = sum(
                            calibration_nose_shoulder_values
                        ) / len(calibration_nose_shoulder_values)
                        baseline_shoulder_width = sum(
                            calibration_shoulder_width_values
                        ) / len(calibration_shoulder_width_values)
                        baseline_face_width = sum(
                            calibration_face_width_values
                        ) / len(calibration_face_width_values)
                        print("Baseline shoulder y:", baseline_shoulder_y)
                        print("Baseline nose to shoulder:", baseline_nose_to_shoulder)
                        print("Baseline shoulder width:", baseline_shoulder_width)
                        print("Baseline face width:", baseline_face_width)

                # Sau khi đã calibration xong
                else:
                    draw_baseline(frame, baseline_shoulder_y)

                    status, shoulder_drop, nose_ratio, shoulder_width_ratio, face_width_ratio, color = classify_posture(
                        current_shoulder_y,
                        baseline_shoulder_y,
                        current_nose_to_shoulder,
                        baseline_nose_to_shoulder,
                        current_shoulder_width,
                        baseline_shoulder_width,
                        current_face_width,
                        baseline_face_width,
                    )

                    cv2.putText(
                        frame, status, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
                    )
                    cv2.putText(
                        frame, f"Shoulder drop: {shoulder_drop:.3f}", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                    )
                    cv2.putText(
                        frame, f"Nose ratio: {nose_ratio:.2f}", (30, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                    )
                    cv2.putText(
                        frame, f"Shoulder W ratio: {shoulder_width_ratio:.2f}", (30, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                    )
                    cv2.putText(
                        frame, f"Face W ratio: {face_width_ratio:.2f}", (30, 195),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                    )

            else:
                cv2.putText(
                    frame,
                    "No pose detected",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2
                )

            cv2.imshow("Shoulder Line Posture Detection", frame)

            key = cv2.waitKey(5) & 0xFF

            # ESC để thoát
            if key == 27:
                break

            # Nhấn R để calibration lại
            if key == ord("r"):
                baseline_shoulder_y = None
                baseline_nose_to_shoulder = None
                baseline_shoulder_width = None
                baseline_face_width = None
                calibration_shoulder_values = []
                calibration_nose_shoulder_values = []
                calibration_shoulder_width_values = []
                calibration_face_width_values = []
                calibration_start_time = time.time()
                print("Recalibrating...")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
