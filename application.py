from pathlib import Path
import sys
import threading
import time
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
from PySide6.QtCore import QSettings, Qt, QThread, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QAction, QCloseEvent, QImage, QPixmap
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsBlurEffect,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStyle,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)

MODEL_PATH = Path(__file__).with_name("pose_landmarker_full.task")

POSE_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (27, 29),
    (27, 31),
    (29, 31),
    (24, 26),
    (26, 28),
    (28, 30),
    (28, 32),
    (30, 32),
)

STATUS_COLORS = {
    "idle": "#94a3b8",
    "starting": "#38bdf8",
    "calibrating": "#facc15",
    "good": "#22c55e",
    "mild": "#facc15",
    "mild_backward": "#fb923c",
    "hunch": "#ef4444",
    "backward": "#f97316",
    "no_pose": "#fb7185",
    "error": "#ef4444",
}


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
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]

    return (left_shoulder.y + right_shoulder.y) / 2


def get_nose_to_shoulder_distance(landmarks):
    nose = landmarks[0]
    shoulder_y = get_shoulder_y(landmarks)

    return abs(nose.y - shoulder_y)


def get_shoulder_points(frame, landmarks):
    height, width = frame.shape[:2]
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]

    left_point = (int(left_shoulder.x * width), int(left_shoulder.y * height))
    right_point = (int(right_shoulder.x * width), int(right_shoulder.y * height))

    return left_point, right_point


def get_nose_point(frame, landmarks):
    height, width = frame.shape[:2]
    nose = landmarks[0]

    return (int(nose.x * width), int(nose.y * height))


def get_shoulder_width(landmarks):
    """
    Horizontal distance between shoulders (landmarks 11 and 12).
    Decreases when the user leans backward away from the camera.
    """
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    return abs(left_shoulder.x - right_shoulder.x)


def get_face_width(landmarks):
    """
    Horizontal distance between ears:
      landmark 7 = right ear
      landmark 8 = left ear
    Decreases when the user leans backward away from the camera.
    """
    right_ear = landmarks[7]
    left_ear = landmarks[8]
    return abs(right_ear.x - left_ear.x)


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
    shoulder_drop = current_shoulder_y - baseline_shoulder_y

    if baseline_nose_to_shoulder == 0:
        nose_ratio = 1.0
    else:
        nose_ratio = current_nose_to_shoulder / baseline_nose_to_shoulder

    # --- Width ratios (lean-backward detection) ---
    # shoulder_width_ratio = current_shoulder_width / baseline_shoulder_width
    # face_width_ratio      = current_face_width      / baseline_face_width
    # Both shrink when the user leans backward (moves away from camera).
    if baseline_shoulder_width and baseline_shoulder_width > 0:
        shoulder_width_ratio = current_shoulder_width / baseline_shoulder_width
    else:
        shoulder_width_ratio = 1.0

    if baseline_face_width and baseline_face_width > 0:
        face_width_ratio = current_face_width / baseline_face_width
    else:
        face_width_ratio = 1.0

    # Lean-backward thresholds
    lean_backward_strong = shoulder_width_ratio < 0.82 and face_width_ratio < 0.82
    lean_backward_mild   = shoulder_width_ratio < 0.90 or  face_width_ratio < 0.90

    # Forward-hunch thresholds
    shoulder_dropped_strong = shoulder_drop > 0.06
    shoulder_dropped_mild   = shoulder_drop > 0.035
    nose_close_strong = nose_ratio < 0.75
    nose_close_mild   = nose_ratio < 0.85

    if lean_backward_strong:
        status = "Leaning backward"
        color = (0, 128, 255)
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
    height, width = frame.shape[:2]
    baseline_y_pixel = int(baseline_shoulder_y * height)

    cv2.line(frame, (0, baseline_y_pixel), (width, baseline_y_pixel), (255, 0, 0), 2)
    cv2.putText(
        frame,
        "Baseline shoulder line",
        (30, baseline_y_pixel - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2,
    )


def frame_to_qimage(frame) -> QImage:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb_frame.shape
    bytes_per_line = channels * width

    return QImage(
        rgb_frame.data,
        width,
        height,
        bytes_per_line,
        QImage.Format.Format_RGB888,
    ).copy()


class PostureWorker(QThread):
    frame_ready = Signal(QImage)
    metrics_ready = Signal(dict)
    camera_started = Signal()
    camera_stopped = Signal()
    camera_error = Signal(str)

    def __init__(self, camera_index=0, calibration_seconds=3, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.calibration_seconds = calibration_seconds
        self._recalibration_requested = threading.Event()

    def request_recalibration(self) -> None:
        self._recalibration_requested.set()

    def run(self) -> None:
        capture = None

        try:
            ensure_model()
            capture = cv2.VideoCapture(self.camera_index)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            if not capture.isOpened():
                raise RuntimeError(f"Could not open webcam index {self.camera_index}.")

            self.camera_started.emit()
            self._process_frames(capture)
        except Exception as error:
            self.camera_error.emit(str(error))
        finally:
            if capture is not None:
                capture.release()
            self.camera_stopped.emit()

    def _process_frames(self, capture) -> None:
        timestamp_ms = 0
        baseline_shoulder_y = None
        baseline_nose_to_shoulder = None
        baseline_shoulder_width = None
        baseline_face_width = None
        calibration_start_time = None
        calibration_shoulder_values = []
        calibration_nose_shoulder_values = []
        calibration_shoulder_width_values = []
        calibration_face_width_values = []

        with create_landmarker() as landmarker:
            while not self.isInterruptionRequested():
                if self._recalibration_requested.is_set():
                    self._recalibration_requested.clear()
                    baseline_shoulder_y = None
                    baseline_nose_to_shoulder = None
                    baseline_shoulder_width = None
                    baseline_face_width = None
                    calibration_start_time = None
                    calibration_shoulder_values = []
                    calibration_nose_shoulder_values = []
                    calibration_shoulder_width_values = []
                    calibration_face_width_values = []
                    self.metrics_ready.emit(
                        self._build_metrics(
                            "Recalibration requested. Sit straight.",
                            "calibrating",
                        )
                    )

                success, frame = capture.read()

                if not success:
                    self.metrics_ready.emit(
                        self._build_metrics("Camera frame unavailable", "error")
                    )
                    time.sleep(0.05)
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
                    left_shoulder_point, right_shoulder_point = get_shoulder_points(
                        frame, landmarks
                    )
                    nose_point = get_nose_point(frame, landmarks)

                    cv2.line(
                        frame,
                        left_shoulder_point,
                        right_shoulder_point,
                        (255, 255, 0),
                        3,
                    )
                    cv2.circle(frame, nose_point, 7, (255, 0, 255), -1)

                    if (
                        baseline_shoulder_y is None
                        or baseline_nose_to_shoulder is None
                    ):
                        if calibration_start_time is None:
                            calibration_start_time = time.monotonic()

                        elapsed_time = time.monotonic() - calibration_start_time
                        calibration_shoulder_values.append(current_shoulder_y)
                        calibration_nose_shoulder_values.append(
                            current_nose_to_shoulder
                        )
                        calibration_shoulder_width_values.append(current_shoulder_width)
                        calibration_face_width_values.append(current_face_width)
                        remaining_time = max(
                            0, self.calibration_seconds - int(elapsed_time)
                        )

                        cv2.putText(
                            frame,
                            "Sit straight for calibration...",
                            (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 255),
                            2,
                        )
                        cv2.putText(
                            frame,
                            f"Calibrating: {remaining_time}s",
                            (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255),
                            2,
                        )
                        self.metrics_ready.emit(
                            self._build_metrics(
                                f"Calibrating: sit straight ({remaining_time}s)",
                                "calibrating",
                                current_shoulder_y=current_shoulder_y,
                            )
                        )

                        if elapsed_time >= self.calibration_seconds:
                            baseline_shoulder_y = sum(
                                calibration_shoulder_values
                            ) / len(calibration_shoulder_values)
                            baseline_nose_to_shoulder = sum(
                                calibration_nose_shoulder_values
                            ) / len(calibration_nose_shoulder_values)
                            baseline_shoulder_width = sum(
                                calibration_shoulder_width_values
                            ) / len(calibration_shoulder_width_values)
                            baseline_face_width = sum(
                                calibration_face_width_values
                            ) / len(calibration_face_width_values)
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
                        status_kind = self._status_kind(status)

                        cv2.putText(
                            frame,
                            status,
                            (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            color,
                            2,
                        )
                        self.metrics_ready.emit(
                            self._build_metrics(
                                status,
                                status_kind,
                                shoulder_drop,
                                nose_ratio,
                                baseline_shoulder_y,
                                current_shoulder_y,
                            )
                        )
                else:
                    if (
                        baseline_shoulder_y is None
                        or baseline_nose_to_shoulder is None
                    ):
                        calibration_start_time = None
                        calibration_shoulder_values = []
                        calibration_nose_shoulder_values = []
                        calibration_shoulder_width_values = []
                        calibration_face_width_values = []

                    cv2.putText(
                        frame,
                        "No pose detected",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        2,
                    )
                    self.metrics_ready.emit(
                        self._build_metrics(
                            "No pose detected",
                            "no_pose",
                            baseline_shoulder_y=baseline_shoulder_y,
                        )
                    )

                self.frame_ready.emit(frame_to_qimage(frame))

    @staticmethod
    def _status_kind(status: str) -> str:
        if status == "Good posture":
            return "good"
        if status == "Mild slouching":
            return "mild"
        if status == "Slightly leaning backward":
            return "mild_backward"
        if status == "Leaning backward":
            return "backward"
        return "hunch"

    @staticmethod
    def _build_metrics(
        status,
        status_kind,
        shoulder_drop=None,
        nose_ratio=None,
        shoulder_width_ratio=None,
        face_width_ratio=None,
        baseline_shoulder_y=None,
        current_shoulder_y=None,
    ):
        return {
            "status": status,
            "status_kind": status_kind,
            "shoulder_drop": shoulder_drop,
            "nose_ratio": nose_ratio,
            "shoulder_width_ratio": shoulder_width_ratio,
            "face_width_ratio": face_width_ratio,
            "baseline_shoulder_y": baseline_shoulder_y,
            "current_shoulder_y": current_shoulder_y,
        }


class SettingsDialog(QDialog):
    def __init__(self, settings: QSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("Posture Monitor Settings")
        self.setMinimumWidth(460)

        layout = QVBoxLayout(self)
        actions_group = QGroupBox("Actions for sustained hunched posture")
        actions_layout = QVBoxLayout(actions_group)

        self.sound_checkbox = QCheckBox("Play alert music")
        self.blur_checkbox = QCheckBox("Blur the primary screen")
        required_action_label = QLabel("An alert window is always shown.")
        required_action_label.setStyleSheet("color: #64748b;")
        actions_layout.addWidget(required_action_label)
        actions_layout.addWidget(self.sound_checkbox)
        actions_layout.addWidget(self.blur_checkbox)
        layout.addWidget(actions_group)

        behavior_group = QGroupBox("Alert timing")
        behavior_layout = QFormLayout(behavior_group)
        self.delay_spinbox = QSpinBox()
        self.delay_spinbox.setRange(1, 300)
        self.delay_spinbox.setSuffix(" seconds")
        behavior_layout.addRow("Alert after continuous detection:", self.delay_spinbox)
        layout.addWidget(behavior_group)

        sound_group = QGroupBox("Alert music")
        sound_layout = QVBoxLayout(sound_group)
        self.audio_path_label = QLabel()
        self.audio_path_label.setWordWrap(True)
        sound_layout.addWidget(self.audio_path_label)
        choose_audio_button = QPushButton("Choose audio file...")
        choose_audio_button.clicked.connect(self.choose_audio_file)
        sound_layout.addWidget(choose_audio_button)
        fallback_label = QLabel(
            "If no audio file is selected, the app uses a repeating system beep."
        )
        fallback_label.setWordWrap(True)
        fallback_label.setStyleSheet("color: #64748b;")
        sound_layout.addWidget(fallback_label)
        layout.addWidget(sound_group)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._load_values()

    def _load_values(self) -> None:
        self.sound_checkbox.setChecked(
            self.settings.value("actions/sound", False, type=bool)
        )
        self.blur_checkbox.setChecked(
            self.settings.value("actions/blur", False, type=bool)
        )
        self.delay_spinbox.setValue(self.settings.value("alert_delay_seconds", 5, type=int))
        self._set_audio_path(self.settings.value("audio_file", "", type=str))

    def choose_audio_file(self) -> None:
        audio_file, _ = QFileDialog.getOpenFileName(
            self,
            "Choose alert audio",
            self.audio_path,
            "Audio files (*.mp3 *.wav *.ogg *.m4a);;All files (*)",
        )
        if audio_file:
            self._set_audio_path(audio_file)

    def _set_audio_path(self, audio_path: str) -> None:
        self.audio_path = audio_path
        self.audio_path_label.setText(audio_path or "No audio file selected")

    def _validate_and_accept(self) -> None:
        self.settings.setValue("actions/sound", self.sound_checkbox.isChecked())
        self.settings.setValue("actions/blur", self.blur_checkbox.isChecked())
        self.settings.setValue("alert_delay_seconds", self.delay_spinbox.value())
        self.settings.setValue("audio_file", self.audio_path)
        self.settings.sync()
        self.accept()


class PostureAlertDialog(QDialog):
    dismiss_requested = Signal()

    def __init__(self, alert_type="hunch", parent=None):
        super().__init__(parent)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)

        if alert_type == "backward":
            self.setWindowTitle("Posture Alert — Leaning Backward")
            title = QLabel("Leaning backward detected")
            title.setStyleSheet("font-size: 18px; font-weight: 700; color: #f97316;")
            message = QLabel(
                "You appear to be leaning backward.\n"
                "Sit upright with your back against the chair and feet flat on the floor."
            )
        else:
            self.setWindowTitle("Posture Alert")
            title = QLabel("Sustained hunched posture detected")
            title.setStyleSheet("font-size: 18px; font-weight: 700; color: #dc2626;")
            message = QLabel("Please adjust your sitting position.")

        message.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(message)

        dismiss_button = QPushButton("Dismiss")
        dismiss_button.clicked.connect(self.dismiss_requested)
        dismiss_button.clicked.connect(self.close)
        layout.addWidget(dismiss_button)


class BlurOverlay(QWidget):
    dismiss_requested = Signal()

    def __init__(self, screen):
        super().__init__(None)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setGeometry(screen.geometry())
        background = QLabel(self)
        background.setGeometry(self.rect())
        background.setPixmap(screen.grabWindow(0))
        background.setScaledContents(True)
        blur_effect = QGraphicsBlurEffect(background)
        blur_effect.setBlurRadius(28)
        background.setGraphicsEffect(blur_effect)


    def closeEvent(self, event: QCloseEvent) -> None:
        self.dismiss_requested.emit()
        event.accept()


class PostureDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings("KyphosisDetection", "PostureMonitor")
        self.worker = None
        self.current_pixmap = None
        self.has_camera_error = False
        self.hunch_started_at = None
        self.alert_active = False
        self.alert_acknowledged = False
        self._current_alert_type = "hunch"
        self.alert_dialog = None
        self.blur_overlay = None
        self.shutting_down = False
        self.tray_notice_shown = False

        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(0.65)
        self.media_player = QMediaPlayer(self)
        self.media_player.setAudioOutput(self.audio_output)
        self.beep_timer = QTimer(self)
        self.beep_timer.setInterval(1500)
        self.beep_timer.timeout.connect(QApplication.beep)

        self.setWindowTitle("Kyphosis Detection Dashboard")
        self.resize(1280, 760)
        self._build_ui()
        self._setup_tray_icon()
        self._refresh_settings_summary()
        self._set_status("Camera is idle", "idle")
        self._set_controls(camera_running=False)
        QTimer.singleShot(0, self.start_camera)

    def _build_ui(self) -> None:
        central_widget = QWidget()
        root_layout = QHBoxLayout(central_widget)
        root_layout.setContentsMargins(20, 20, 20, 20)
        root_layout.setSpacing(20)

        self.video_label = QLabel("Camera preview will appear here")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(720, 480)
        self.video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.video_label.setStyleSheet(
            "background-color: #020617; color: #94a3b8; "
            "border: 1px solid #334155; border-radius: 12px;"
        )
        root_layout.addWidget(self.video_label, stretch=3)

        panel = QFrame()
        panel.setFixedWidth(340)
        panel.setStyleSheet(
            "QFrame { background-color: #0f172a; border-radius: 12px; }"
            "QLabel { color: #e2e8f0; }"
        )
        panel_layout = QVBoxLayout(panel)

        panel_layout.setContentsMargins(20, 20, 20, 20)
        panel_layout.setSpacing(16)

        title = QLabel("Posture Monitor")
        title.setStyleSheet("font-size: 24px; font-weight: 700; color: #f8fafc;")
        panel_layout.addWidget(title)

        subtitle = QLabel("Runs in the background and reminds you to sit upright.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #94a3b8; font-size: 13px;")
        panel_layout.addWidget(subtitle)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumHeight(82)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        panel_layout.addWidget(self.status_label)

        self.alert_progress_label = QLabel("Waiting for calibration")
        self.alert_progress_label.setWordWrap(True)
        self.alert_progress_label.setStyleSheet("color: #cbd5e1;")
        panel_layout.addWidget(self.alert_progress_label)

        settings_title = QLabel("Active reminder settings")
        settings_title.setStyleSheet("font-size: 16px; font-weight: 700;")
        panel_layout.addWidget(settings_title)

        self.settings_summary_label = QLabel()
        self.settings_summary_label.setWordWrap(True)
        self.settings_summary_label.setStyleSheet("color: #94a3b8;")
        panel_layout.addWidget(self.settings_summary_label)
        panel_layout.addStretch()

        self.start_button = QPushButton("Start Monitoring")
        self.stop_button = QPushButton("Stop Monitoring")
        self.recalibrate_button = QPushButton("Recalibrate")
        self.settings_button = QPushButton("Settings")
        self.hide_button = QPushButton("Hide to Background")
        self.exit_button = QPushButton("Exit Application")

        for button in (
            self.start_button,
            self.stop_button,
            self.recalibrate_button,
            self.settings_button,
            self.hide_button,
            self.exit_button,
        ):
            button.setMinimumHeight(42)
            button.setCursor(Qt.CursorShape.PointingHandCursor)

        self.start_button.setStyleSheet(self._button_style("#0284c7", "#0369a1"))
        self.stop_button.setStyleSheet(self._button_style("#dc2626", "#b91c1c"))
        self.recalibrate_button.setStyleSheet(self._button_style("#475569", "#334155"))
        self.settings_button.setStyleSheet(self._button_style("#475569", "#334155"))
        self.hide_button.setStyleSheet(self._button_style("#334155", "#1e293b"))
        self.exit_button.setStyleSheet(self._button_style("#991b1b", "#7f1d1d"))

        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        self.recalibrate_button.clicked.connect(self.recalibrate)
        self.settings_button.clicked.connect(self.open_settings)
        self.hide_button.clicked.connect(self.hide_to_background)
        self.exit_button.clicked.connect(self.quit_application)

        panel_layout.addWidget(self.start_button)
        panel_layout.addWidget(self.stop_button)
        panel_layout.addWidget(self.recalibrate_button)
        panel_layout.addWidget(self.settings_button)
        panel_layout.addWidget(self.hide_button)
        panel_layout.addWidget(self.exit_button)
        root_layout.addWidget(panel)

        central_widget.setStyleSheet("background-color: #020617;")
        self.setCentralWidget(central_widget)

    def _setup_tray_icon(self) -> None:
        self.tray_icon = None
        if not QSystemTrayIcon.isSystemTrayAvailable():
            self.hide_button.setEnabled(False)
            self.hide_button.setToolTip("The system tray is not available.")
            return

        self.tray_icon = QSystemTrayIcon(self)
        icon = self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
        self.tray_icon.setIcon(icon)
        self.setWindowIcon(icon)

        tray_menu = QMenu(self)
        show_action = QAction("Show Dashboard", self)
        dismiss_action = QAction("Dismiss Alert", self)
        quit_action = QAction("Quit", self)
        show_action.triggered.connect(self.show_dashboard)
        dismiss_action.triggered.connect(self.dismiss_alert)
        quit_action.triggered.connect(self.quit_application)
        tray_menu.addAction(show_action)
        tray_menu.addAction(dismiss_action)
        tray_menu.addSeparator()
        tray_menu.addAction(quit_action)
        self.tray_menu = tray_menu
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.activated.connect(self._on_tray_activated)
        self.tray_icon.show()

    @staticmethod
    def _button_style(background: str, hover: str) -> str:
        return (
            "QPushButton {"
            f"background-color: {background}; color: white; border: none; "
            "border-radius: 7px; font-size: 14px; font-weight: 700;"
            "}"
            f"QPushButton:hover {{ background-color: {hover}; }}"
            "QPushButton:disabled { background-color: #1e293b; color: #64748b; }"
        )

    @Slot()
    def start_camera(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return

        self.has_camera_error = False
        self._reset_hunch_detection()
        self._set_status("Starting camera...", "starting")
        self._set_controls(camera_running=True, camera_ready=False)

        worker = PostureWorker(camera_index=0, calibration_seconds=3, parent=self)
        worker.frame_ready.connect(self.update_frame)
        worker.metrics_ready.connect(self.update_metrics)
        worker.camera_started.connect(self.on_camera_started)
        worker.camera_error.connect(self.on_camera_error)
        worker.camera_stopped.connect(self.on_camera_stopped)
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(lambda: self._clear_worker(worker))
        self.worker = worker
        worker.start()

    @Slot()
    def stop_camera(self) -> None:
        if self.worker is None or not self.worker.isRunning():
            return

        self._reset_hunch_detection()
        self._set_status("Stopping camera...", "idle")
        self._set_controls(camera_running=True, camera_ready=False)
        self.stop_button.setEnabled(False)
        self.worker.requestInterruption()

    @Slot()
    def recalibrate(self) -> None:
        if self.worker is None or not self.worker.isRunning():
            return

        self._reset_hunch_detection()
        self._set_status("Recalibration requested. Sit straight.", "calibrating")
        self.alert_progress_label.setText("Waiting for calibration")
        self.worker.request_recalibration()

    @Slot()
    def open_settings(self) -> None:
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._reset_hunch_detection()
            self._refresh_settings_summary()

    @Slot()
    def on_camera_started(self) -> None:
        self._set_status("Calibrating: sit straight", "calibrating")
        self.alert_progress_label.setText("Waiting for calibration")
        self._set_controls(camera_running=True, camera_ready=True)

    @Slot(str)
    def on_camera_error(self, message: str) -> None:
        self.has_camera_error = True
        self._reset_hunch_detection()
        self._set_status(f"Camera error: {message}", "error")
        self.alert_progress_label.setText("Monitoring is unavailable")

    @Slot()
    def on_camera_stopped(self) -> None:
        self._reset_hunch_detection()
        if not self.has_camera_error:
            self._set_status("Camera is idle", "idle")
            self.alert_progress_label.setText("Monitoring is stopped")
        self._set_controls(camera_running=False)

    def _clear_worker(self, worker) -> None:
        if self.worker is worker:
            self.worker = None

    @Slot(QImage)
    def update_frame(self, image: QImage) -> None:
        self.current_pixmap = QPixmap.fromImage(image)
        self._refresh_video_frame()

    @Slot(dict)
    def update_metrics(self, metrics: dict) -> None:
        status_kind = metrics["status_kind"]
        self._set_status(metrics["status"], status_kind)
        self._track_hunch_status(status_kind)

    def _track_hunch_status(self, status_kind: str) -> None:
        if status_kind not in ("hunch", "backward"):
            self._reset_hunch_detection()
            if status_kind == "good":
                self.alert_progress_label.setText("Posture looks good")
            elif status_kind == "mild":
                self.alert_progress_label.setText("Mild slouching: sit upright")
            elif status_kind == "mild_backward":
                self.alert_progress_label.setText("Slightly leaning backward: adjust your position")
            elif status_kind == "no_pose":
                self.alert_progress_label.setText("Move into view to resume monitoring")
            return

        now = time.monotonic()
        if self.hunch_started_at is None:
            self.hunch_started_at = now
            self.alert_acknowledged = False
            self._current_alert_type = status_kind  # track what triggered the alert

        delay_seconds = self.settings.value("alert_delay_seconds", 5, type=int)
        elapsed_seconds = now - self.hunch_started_at
        remaining_seconds = max(0, int(delay_seconds - elapsed_seconds + 0.999))

        if elapsed_seconds >= delay_seconds and not self.alert_acknowledged:
            self.trigger_alert(self._current_alert_type)
        elif self.alert_active:
            self.alert_progress_label.setText("Posture reminder is active")
        elif self.alert_acknowledged:
            self.alert_progress_label.setText(
                "Reminder dismissed. Sit upright to reset monitoring."
            )
        else:
            label = "leaning backward" if status_kind == "backward" else "hunched posture"
            self.alert_progress_label.setText(
                f"{label.capitalize()} detected. Reminder in {remaining_seconds}s"
            )

    def trigger_alert(self, alert_type="hunch") -> None:
        self.alert_active = True
        self.alert_acknowledged = True
        self.alert_progress_label.setText("Posture reminder is active")

        if self.settings.value("actions/blur", False, type=bool):
            self._show_blur_overlay()
        if self.settings.value("actions/sound", False, type=bool):
            self._start_alert_sound()
        self._show_alert_dialog(alert_type)

        if self.tray_icon is not None:
            if alert_type == "backward":
                tray_msg = "Leaning backward detected. Please sit upright with back support."
            else:
                tray_msg = "Sustained hunched posture detected. Please sit upright."
            self.tray_icon.showMessage(
                "Posture reminder",
                tray_msg,
                QSystemTrayIcon.MessageIcon.Warning,
                5000,
            )

    @Slot()
    def dismiss_alert(self) -> None:
        self.alert_active = False
        self.media_player.stop()
        self.beep_timer.stop()

        alert_dialog = self.alert_dialog
        self.alert_dialog = None
        if alert_dialog is not None:
            alert_dialog.close()

        blur_overlay = self.blur_overlay
        self.blur_overlay = None
        if blur_overlay is not None:
            blur_overlay.close()

        if self.hunch_started_at is not None:
            self.alert_progress_label.setText(
                "Reminder dismissed. Sit upright to reset monitoring."
            )

    def _show_alert_dialog(self, alert_type="hunch") -> None:
        if self.alert_dialog is not None:
            return

        self.alert_dialog = PostureAlertDialog(alert_type, self)
        self.alert_dialog.dismiss_requested.connect(self.dismiss_alert)
        self.alert_dialog.finished.connect(self._clear_alert_dialog)
        self.alert_dialog.show()
        self.alert_dialog.raise_()
        self.alert_dialog.activateWindow()

    def _show_blur_overlay(self) -> None:
        if self.blur_overlay is not None:
            return

        screen = QApplication.primaryScreen()
        if screen is None:
            return

        self.blur_overlay = BlurOverlay(screen)
        self.blur_overlay.dismiss_requested.connect(self.dismiss_alert)
        self.blur_overlay.destroyed.connect(self._clear_blur_overlay)
        self.blur_overlay.showFullScreen()

    def _start_alert_sound(self) -> None:
        audio_file = self.settings.value("audio_file", "", type=str)
        if audio_file and Path(audio_file).is_file():
            self.media_player.setSource(QUrl.fromLocalFile(audio_file))
            self.media_player.setLoops(QMediaPlayer.Loops.Infinite)
            self.media_player.play()
            return

        QApplication.beep()
        self.beep_timer.start()

    def _clear_alert_dialog(self) -> None:
        self.alert_dialog = None
        if self.alert_active:
            self.dismiss_alert()

    def _clear_blur_overlay(self) -> None:
        self.blur_overlay = None

    def _reset_hunch_detection(self) -> None:
        self.hunch_started_at = None
        self.alert_acknowledged = False
        self.dismiss_alert()

    def _refresh_settings_summary(self) -> None:
        actions = ["alert window"]
        if self.settings.value("actions/sound", False, type=bool):
            actions.append("music")
        if self.settings.value("actions/blur", False, type=bool):
            actions.append("screen blur")

        delay_seconds = self.settings.value("alert_delay_seconds", 5, type=int)
        self.settings_summary_label.setText(
            f"After {delay_seconds}s of continuous detection: {', '.join(actions)}"
        )

    def _set_status(self, text: str, status_kind: str) -> None:
        color = STATUS_COLORS.get(status_kind, STATUS_COLORS["idle"])
        self.status_label.setText(text)
        self.status_label.setStyleSheet(
            f"background-color: {color}; color: #020617; border-radius: 8px; "
            "padding: 12px; font-size: 16px; font-weight: 700;"
        )

    def _set_controls(self, camera_running: bool, camera_ready: bool = False) -> None:
        self.start_button.setEnabled(not camera_running)
        self.stop_button.setEnabled(camera_running)
        self.recalibrate_button.setEnabled(camera_ready)

    def _refresh_video_frame(self) -> None:
        if self.current_pixmap is None:
            return

        scaled_pixmap = self.current_pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled_pixmap)

    @Slot()
    def show_dashboard(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    @Slot()
    def hide_to_background(self) -> None:
        if self.tray_icon is None:
            return

        self.hide()
        if not self.tray_notice_shown:
            self.tray_icon.showMessage(
                "Posture Monitor",
                "Monitoring continues in the background. Use the tray icon to reopen.",
                QSystemTrayIcon.MessageIcon.Information,
                4000,
            )
            self.tray_notice_shown = True

    @Slot(QSystemTrayIcon.ActivationReason)
    def _on_tray_activated(self, reason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.show_dashboard()

    @Slot()
    def quit_application(self) -> None:
        self.shutting_down = True
        self.dismiss_alert()

        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption()
            if not self.worker.wait(5000):
                self.shutting_down = False
                self.show_dashboard()
                self._set_status("Waiting for the camera to stop...", "error")
                return

        if self.tray_icon is not None:
            self.tray_icon.hide()
        QApplication.instance().quit()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh_video_frame()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.tray_icon is not None and not self.shutting_down:
            self.hide_to_background()
            event.ignore()
            return

        self.quit_application()
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    window = PostureDashboard()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
