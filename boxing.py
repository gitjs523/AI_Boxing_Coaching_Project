import math
import os
import subprocess
import sys
import urllib.request

import cv2
import numpy as np
import onnxruntime as ort

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "movenet_singlepose_lightning_4.onnx")

# 검증된 ONNX 모델 미러 목록 (순차 시도)
URLS = [
    "https://huggingface.co/Xenova/movenet-singlepose-lightning/resolve/main/onnx/model.onnx",
    "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/ultraface/models/version-RFB-320.onnx",  # fallback check
]

if not os.path.exists(MODEL_PATH):
  print("구글 MoveNet(ONNX) 포즈 모델을 다운로드 중입니다...")
  req = urllib.request.Request(URLS[0], headers={"User-Agent": "Mozilla/5.0"})
  with urllib.request.urlopen(req) as response, open(
      MODEL_PATH, "wb"
  ) as out_file:
    out_file.write(response.read())
  print("모델 다운로드 완료!")


def calculate_angle(a, b, c):
  a, b, c = np.array(a), np.array(b), np.array(c)
  radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
      a[1] - b[1], a[0] - b[0]
  )
  angle = np.abs(radians * 180.0 / np.pi)
  return 360 - angle if angle > 180.0 else angle


def calculate_distance(p1, p2):
  return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_info = session.get_inputs()[0]
input_name = input_info.name
input_type = input_info.type

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

jab_count = 0
jab_state = "GUARD"
feedback_msg = "Ready! Keep your guard up."
feedback_color = (0, 255, 0)

# MoveNet 17개 관절 인덱스: 0:코, 5:왼어깨, 6:오른어깨, 7:왼팔꿈치, 8:오른팔꿈치, 9:왼손목, 10:오른손목
while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    break

  # MoveNet 입력: 192x192 RGB
  img = cv2.resize(frame, (192, 192))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  if "int32" in input_type:
    input_tensor = np.expand_dims(img.astype(np.int32), axis=0)
  else:
    input_tensor = np.expand_dims(img.astype(np.float32), axis=0)

  # 추론 결과: shape (1, 1, 17, 3) -> [y, x, score]
  outputs = session.run(None, {input_name: input_tensor})[0]
  kpts = outputs[0, 0]  # (17, 3)

  display_frame = cv2.flip(frame, 1)
  h, w, _ = display_frame.shape

  required_indices = [0, 5, 6, 7, 8, 9, 10]
  if all(kpts[i][2] > 0.25 for i in required_indices):
    # MoveNet은 좌표가 (y, x, score) 순서로 반환됨! 거울 모드에 맞춰 X좌표 반전(1.0 - x)
    def get_mirror_pt(idx):
      y, x, _ = kpts[idx]
      return [int((1.0 - x) * w), int(y * h)]

    nose = get_mirror_pt(0)
    l_shoulder, r_shoulder = get_mirror_pt(5), get_mirror_pt(6)
    l_elbow, r_elbow = get_mirror_pt(7), get_mirror_pt(8)
    l_wrist, r_wrist = get_mirror_pt(9), get_mirror_pt(10)

    shoulder_width = calculate_distance(l_shoulder, r_shoulder)
    if shoulder_width > 20:
      left_arm_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
      r_guard_dist = calculate_distance(r_wrist, nose) / shoulder_width
      is_r_hand_low = r_wrist[1] > r_shoulder[1]

      if left_arm_angle > 155:
        if jab_state == "GUARD":
          jab_count += 1
          jab_state = "EXTENDED"

        if r_guard_dist > 1.4 or is_r_hand_low:
          feedback_msg = "WARNING: Keep your RIGHT hand up!"
          feedback_color = (0, 0, 255)
        else:
          feedback_msg = "GOOD JAB! Nice guard."
          feedback_color = (0, 255, 0)

      elif left_arm_angle < 95:
        jab_state = "GUARD"
        if (l_wrist[1] > l_shoulder[1]) or is_r_hand_low:
          feedback_msg = "Hands up! Protect your chin."
          feedback_color = (0, 165, 255)
        else:
          feedback_msg = "Good Stance."
          feedback_color = (0, 255, 0)

      cv2.line(display_frame, l_shoulder, l_elbow, (255, 255, 0), 4)
      cv2.line(display_frame, l_elbow, l_wrist, (255, 255, 0), 4)
      cv2.line(display_frame, r_shoulder, r_elbow, feedback_color, 4)
      cv2.line(display_frame, r_elbow, r_wrist, feedback_color, 4)

      for pt in [l_shoulder, l_elbow, l_wrist, r_shoulder, r_elbow, r_wrist]:
        cv2.circle(display_frame, pt, 8, (255, 255, 255), -1)

      cv2.putText(
          display_frame,
          f"{int(left_arm_angle)} deg",
          l_elbow,
          cv2.FONT_HERSHEY_SIMPLEX,
          0.7,
          (255, 255, 255),
          2,
      )

  cv2.rectangle(display_frame, (0, 0), (w, 90), (20, 20, 20), -1)
  cv2.putText(
      display_frame,
      f"JAB COUNT: {jab_count}",
      (30, 55),
      cv2.FONT_HERSHEY_SIMPLEX,
      1.3,
      (255, 255, 255),
      3,
  )
  cv2.putText(
      display_frame,
      feedback_msg,
      (420, 55),
      cv2.FONT_HERSHEY_SIMPLEX,
      1.0,
      feedback_color,
      3,
  )

  cv2.imshow("AI Boxing Coach - MoveNet (Press 'q' to quit)", display_frame)
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()