"""
evaluate_dataset.py
====================
Evaluate classify_posture accuracy on static images in Dataset/.

File naming convention:
  {person}_{label}.jpg
  e.g. anhdahong_goodposture.jpg, anhdahong_mildslouching.jpg

Steps:
  1. For each person, find the goodposture image as the baseline.
  2. Extract MediaPipe landmarks from the baseline image.
  3. For each remaining test image of that person, run classify_posture
     and compare against the ground-truth label encoded in the filename.
  4. Print a detailed results table + overall accuracy.
"""

import sys
import io

# Fix Windows console encoding
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
import cv2
import mediapipe as mp
from algorithm import (
    ensure_model,
    get_shoulder_y,
    get_shoulder_width,
    get_face_width,
    get_nose_to_shoulder_distance,
    classify_posture,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_DIR = Path(__file__).parent / "Dataset"

# Mapping from filename label suffix -> canonical class name
LABEL_MAP: dict[str, str] = {
    "goodposture":             "good",
    "goodposture2":            "good",
    "mildslouching":           "mild",
    "likelyhunch":             "hunch",
    "leaningbackward":         "backward",
    "slightlyleaningbackward": "mild_backward",
}

# Mapping from algorithm status string -> canonical class name
STATUS_TO_CLASS: dict[str, str] = {
    "Good posture":                         "good",
    "Mild slouching":                       "mild",
    "Likely hunch/slouching":               "hunch",
    "Likely hunch: shoulders dropped":      "hunch",
    "Likely hunch: nose close to shoulder": "hunch",
    "Leaning backward":                     "backward",
    "Slightly leaning backward":            "mild_backward",
}


# ---------------------------------------------------------------------------
# MediaPipe - IMAGE mode (static photos, not VIDEO)
# ---------------------------------------------------------------------------
def create_image_landmarker():
    ensure_model()
    model_path = Path(__file__).with_name("pose_landmarker_full.task")
    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    return mp.tasks.vision.PoseLandmarker.create_from_options(options)


def detect_landmarks(landmarker, image_path: Path):
    """Read image and return (landmarks, bgr_frame) or (None, None)."""
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        print(f"  [!] Cannot read image: {image_path.name}")
        return None, None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    if not result.pose_landmarks:
        return None, bgr
    return result.pose_landmarks[0], bgr


# ---------------------------------------------------------------------------
# Filename parser
# ---------------------------------------------------------------------------
def parse_filename(path: Path):
    """
    Extract (person, label_key) from filename.
    e.g. anhdahong_mildslouching.jpg -> ('anhdahong', 'mildslouching')
    Returns None if no matching label suffix found.
    """
    stem = path.stem.lower()
    for label_key in sorted(LABEL_MAP, key=len, reverse=True):
        if stem.endswith("_" + label_key):
            person = stem[: -(len(label_key) + 1)]
            return person, label_key
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    images = sorted(DATASET_DIR.glob("*.jpg")) + sorted(DATASET_DIR.glob("*.png"))
    if not images:
        print(f"No images found in {DATASET_DIR}")
        sys.exit(1)

    # Group images by person
    persons: dict[str, dict[str, Path]] = {}
    skipped = []
    for img_path in images:
        parsed = parse_filename(img_path)
        if parsed is None:
            skipped.append(img_path.name)
            continue
        person, label_key = parsed
        persons.setdefault(person, {})[label_key] = img_path

    if skipped:
        print(f"[!] Skipped (cannot parse label): {skipped}\n")

    print("=" * 80)
    print(f"{'FILE':<45} {'TRUE LABEL':<18} {'PREDICTED STATUS':<28} RESULT")
    print("=" * 80)

    total = 0
    correct = 0
    results_by_class: dict[str, dict[str, int]] = {}

    with create_image_landmarker() as landmarker:
        for person, label_map in sorted(persons.items()):

            # --- Find baseline image (goodposture) --------------------------
            baseline_key = None
            for k in ("goodposture", "goodposture2"):
                if k in label_map:
                    baseline_key = k
                    break

            if baseline_key is None:
                print(f"\n[!] {person}: no goodposture image found -- skipping\n")
                continue

            print(f"\n-- {person.upper()} " + "-" * 56)

            baseline_lm, _ = detect_landmarks(landmarker, label_map[baseline_key])
            if baseline_lm is None:
                print("  [!] No pose detected in baseline image -- skipping person")
                continue

            baseline_shoulder_y       = get_shoulder_y(baseline_lm)
            baseline_nose_to_shoulder = get_nose_to_shoulder_distance(baseline_lm)
            baseline_shoulder_width   = get_shoulder_width(baseline_lm)
            baseline_face_width       = get_face_width(baseline_lm)

            print(
                f"  Baseline: shoulder_y={baseline_shoulder_y:.3f}  "
                f"nose_to_shoulder={baseline_nose_to_shoulder:.3f}  "
                f"shoulder_w={baseline_shoulder_width:.3f}  "
                f"face_w={baseline_face_width:.3f}"
            )

            # --- Evaluate each test image -----------------------------------
            for label_key, img_path in sorted(label_map.items()):
                true_class = LABEL_MAP[label_key]

                test_lm, _ = detect_landmarks(landmarker, img_path)
                if test_lm is None:
                    print(f"  {'[no pose detected]':<43} {true_class:<18} {'--':<28} SKIP")
                    continue

                current_shoulder_y       = get_shoulder_y(test_lm)
                current_nose_to_shoulder = get_nose_to_shoulder_distance(test_lm)
                current_shoulder_width   = get_shoulder_width(test_lm)
                current_face_width       = get_face_width(test_lm)

                status, shoulder_drop, nose_ratio, sw_ratio, fw_ratio, _ = classify_posture(
                    current_shoulder_y,
                    baseline_shoulder_y,
                    current_nose_to_shoulder,
                    baseline_nose_to_shoulder,
                    current_shoulder_width,
                    baseline_shoulder_width,
                    current_face_width,
                    baseline_face_width,
                )

                pred_class = STATUS_TO_CLASS.get(status, "unknown")
                is_correct = pred_class == true_class
                total += 1
                if is_correct:
                    correct += 1

                results_by_class.setdefault(true_class, {})
                results_by_class[true_class][pred_class] = (
                    results_by_class[true_class].get(pred_class, 0) + 1
                )

                result_str = "OK  " if is_correct else "FAIL"
                print(f"  {img_path.name:<43} {true_class:<18} {status:<28} {result_str}")
                print(
                    f"      drop={shoulder_drop:+.3f}  nose={nose_ratio:.2f}  "
                    f"sw_ratio={sw_ratio:.2f}  fw_ratio={fw_ratio:.2f}"
                )

    # --- Summary ------------------------------------------------------------
    print("\n" + "=" * 80)
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"TOTAL: {correct}/{total} correct  ->  Accuracy = {accuracy:.1f}%")

    if results_by_class:
        all_classes = sorted(
            set(k for v in results_by_class.values() for k in v)
            | set(results_by_class.keys())
        )
        print("\nConfusion matrix (rows=true label, cols=predicted label):")
        header = f"{'':20}" + "".join(f"{c:>16}" for c in all_classes)
        print(header)
        print("-" * len(header))
        for true_c in all_classes:
            row = results_by_class.get(true_c, {})
            row_total = sum(row.values())
            cells = "".join(f"{row.get(pred_c, 0):>16}" for pred_c in all_classes)
            print(f"{true_c:<20}{cells}   (n={row_total})")

    print("=" * 80)


if __name__ == "__main__":
    main()
