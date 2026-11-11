import math
import os
import subprocess
import sys
import time
import urllib.request

for pkg_name, import_name in [
    ("opencv-python", "cv2"),
    ("numpy", "numpy"),
    ("onnxruntime", "onnxruntime"),
]:
  try:
    __import__(import_name)
  except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])

import cv2
import numpy as np
import onnxruntime as ort

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "movenet_singlepose_lightning_4.onnx")
MODEL_URL = "https://huggingface.co/Xenova/movenet-singlepose-lightning/resolve/main/onnx/model.onnx"

if not os.path.exists(MODEL_PATH):
  print("구글 MoveNet(ONNX) 포즈 모델을 다운로드 중입니다...")
  req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
  with urllib.request.urlopen(req) as response, open(
      MODEL_PATH, "wb"
  ) as out_file:
    out_file.write(response.read())


def calculate_angle(a, b, c):
  a, b, c = np.array(a), np.array(b), np.array(c)
  radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
      a[1] - b[1], a[0] - b[0]
  )
  angle = np.abs(radians * 180.0 / np.pi)
  return 360 - angle if angle > 180.0 else angle


def calculate_distance(p1, p2):
  return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


# ★ [핵심 보정 1] 16:9 화면이 찌그러지지 않도록 정사각형 패딩(Letterbox) 후 추론하는 함수
def detect_movenet_letterbox(session, input_name, input_type, frame):
  h, w, _ = frame.shape
  scale = min(192 / w, 192 / h)
  nw, nh = int(w * scale), int(h * scale)
  resized = cv2.resize(frame, (nw, nh))

  # 192x192 검은색 정사각형 캔버스 중앙에 배치 (비율 왜곡 방지!)
  canvas = np.zeros((192, 192, 3), dtype=np.uint8)
  dw, dh = (192 - nw) // 2, (192 - nh) // 2
  canvas[dh : dh + nh, dw : dw + nw] = resized

  rgb_img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
  input_tensor = np.expand_dims(
      rgb_img.astype(np.int32 if "int32" in input_type else np.float32), axis=0
  )

  outputs = session.run(None, {input_name: input_tensor})[0]
  raw_kpts = outputs[0, 0]  # (17, 3) -> [y, x, score] (0.0 ~ 1.0)

  # 패딩된 좌표를 원본 화면 비율(0.0 ~ 1.0)로 정확히 역변환
  real_kpts = np.zeros_like(raw_kpts)
  for i in range(17):
    py = raw_kpts[i, 0] * 192.0
    px = raw_kpts[i, 1] * 192.0
    real_kpts[i, 0] = (py - dh) / nh  # 원본 y 비율
    real_kpts[i, 1] = (px - dw) / nw  # 원본 x 비율
    real_kpts[i, 2] = raw_kpts[i, 2]  # 신뢰도(score)
  return real_kpts


session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_info = session.get_inputs()[0]
input_name, input_type = input_info.name, input_info.type

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
  cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

jab_count, cross_count, duck_count = 0, 0, 0
jab_state, cross_state, duck_state = "GUARD", "GUARD", "STAND"
jab_start_time, cross_start_time, last_punch_speed = 0.0, 0.0, 0.0

feedback_msg = "Step back slightly so your upper body is visible!"
feedback_color = (0, 255, 255)
last_valid_wrists = {}  # 손목(9, 10번)만 잔상 보완 (어깨는 고정하지 않음!)

try:
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break

    # 비율 왜곡 없는 정밀 관절 추출
    kpts = detect_movenet_letterbox(session, input_name, input_type, frame)

    display_frame = cv2.flip(frame, 1)
    h, w, _ = display_frame.shape

    def get_mirror_pt(idx):
      y, x, _ = kpts[idx]
      return [int((1.0 - x) * w), int(y * h)]

    # ★ [핵심 보정 2] 코(0)와 양쪽 어깨(5, 6)는 신뢰도가 0.35 이상으로 확실할 때만 인정!
    if kpts[0][2] > 0.35 and kpts[5][2] > 0.35 and kpts[6][2] > 0.35:
      nose = get_mirror_pt(0)
      l_shoulder = get_mirror_pt(5)
      r_shoulder = get_mirror_pt(6)

      shoulder_width = calculate_distance(l_shoulder, r_shoulder)
      shoulder_center_y = int((l_shoulder[1] + r_shoulder[1]) / 2)

      # ★ [핵심 보정 3] 인중/입을 어깨로 착각하는 현상 방지 (기하학적 검증)
      # 조건 A: 양 어깨 사이의 너비가 화면 가로의 10% 이상이어야 함 (입꼬리 간격 배제)
      # 조건 B: 어깨 중심이 코보다 확실히 아래(최소 40픽셀 이상 아래)에 있어야 함
      is_valid_body = (shoulder_width > w * 0.10) and (
          (shoulder_center_y - nose[1]) > 35
      )

      if not is_valid_body:
        feedback_msg = (
            "Too close! Step back so your SHOULDERS are clearly visible."
        )
        feedback_color = (0, 165, 255)
      else:
        # 팔꿈치/손목이 빠른 펀치로 흐려질 때만 직전 손목 좌표로 보완
        for idx in [7, 8, 9, 10]:
          if kpts[idx][2] > 0.20:
            last_valid_wrists[idx] = get_mirror_pt(idx)
          elif idx not in last_valid_wrists:
            last_valid_wrists[idx] = get_mirror_pt(idx)

        l_elbow, r_elbow = last_valid_wrists[7], last_valid_wrists[8]
        l_wrist, r_wrist = last_valid_wrists[9], last_valid_wrists[10]

        left_arm_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        right_arm_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)

        r_guard_dist = calculate_distance(r_wrist, nose) / shoulder_width
        l_guard_dist = calculate_distance(l_wrist, nose) / shoulder_width

        l_reach_ratio = calculate_distance(l_shoulder, l_wrist) / shoulder_width
        r_reach_ratio = calculate_distance(r_shoulder, r_wrist) / shoulder_width

        is_r_hand_low = r_wrist[1] > r_shoulder[1]
        is_l_hand_low = l_wrist[1] > l_shoulder[1]

        l_arm_color = (255, 255, 0)
        r_arm_color = (0, 255, 0)

        # -------------------------------------------------------------
        # 1. 더킹/위빙(회피) 실시간 상태 반영
        # -------------------------------------------------------------
        duck_ratio = (shoulder_center_y - nose[1]) / shoulder_width
        duck_trigger_y = int(shoulder_center_y - (shoulder_width * 0.38))
        cv2.line(
            display_frame,
            (l_shoulder[0] - 30, duck_trigger_y),
            (r_shoulder[0] + 30, duck_trigger_y),
            (255, 255, 0),
            2,
        )

        if duck_ratio < 0.38:
          if duck_state == "STAND":
            duck_count += 1
            duck_state = "DUCKED"
          feedback_msg = f"DUCKING! (Down: {duck_ratio:.2f})"
          feedback_color = (255, 255, 0)
        elif duck_ratio >= 0.43:
          if duck_state == "DUCKED":
            duck_state = "STAND"
            feedback_msg = "Returned to Guard! Ready."
            feedback_color = (0, 255, 0)

        # -------------------------------------------------------------
        # 2. 왼손 잽(Jab) 판정
        # -------------------------------------------------------------
        if (
            left_arm_angle > 95
            and jab_state == "GUARD"
            and jab_start_time == 0.0
        ):
          jab_start_time = time.time()

        is_left_extended = (left_arm_angle > 135) or (
            left_arm_angle > 115 and l_reach_ratio > 1.15
        )
        if is_left_extended:
          if jab_state == "GUARD":
            jab_count += 1
            jab_state = "EXTENDED"
            if jab_start_time > 0:
              last_punch_speed = round(
                  max(0.05, time.time() - jab_start_time), 2
              )
              jab_start_time = 0.0

          if r_guard_dist > 1.4 or is_r_hand_low:
            feedback_msg = "WARNING: Keep RIGHT hand up on Jab!"
            feedback_color = (0, 0, 255)
            r_arm_color = (0, 0, 255)
          else:
            feedback_msg = f"GOOD JAB! ({last_punch_speed}s)"
            feedback_color = (0, 255, 0)
        elif left_arm_angle < 110 and l_reach_ratio < 1.0:
          jab_state = "GUARD"
          jab_start_time = 0.0

        # -------------------------------------------------------------
        # 3. 오른손 스트레이트(Cross) 판정
        # -------------------------------------------------------------
        if (
            right_arm_angle > 95
            and cross_state == "GUARD"
            and cross_start_time == 0.0
        ):
          cross_start_time = time.time()

        is_right_extended = (right_arm_angle > 135) or (
            right_arm_angle > 115 and r_reach_ratio > 1.15
        )
        if is_right_extended:
          if cross_state == "GUARD":
            cross_count += 1
            cross_state = "EXTENDED"
            if cross_start_time > 0:
              last_punch_speed = round(
                  max(0.05, time.time() - cross_start_time), 2
              )
              cross_start_time = 0.0

          if l_guard_dist > 1.4 or is_l_hand_low:
            feedback_msg = "WARNING: Keep LEFT hand up on Cross!"
            feedback_color = (0, 0, 255)
            l_arm_color = (0, 0, 255)
          else:
            feedback_msg = f"GOOD CROSS! ({last_punch_speed}s)"
            feedback_color = (0, 255, 0)
        elif right_arm_angle < 110 and r_reach_ratio < 1.0:
          cross_state = "GUARD"
          cross_start_time = 0.0

        # -------------------------------------------------------------
        # 4. 평상시 스탠스 체크
        # -------------------------------------------------------------
        if (
            jab_state == "GUARD"
            and cross_state == "GUARD"
            and duck_state == "STAND"
        ):
          if is_l_hand_low or is_r_hand_low:
            feedback_msg = "Hands up! Protect your chin."
            feedback_color = (0, 165, 255)
          else:
            feedback_msg = "Good Stance (Ready)."
            feedback_color = (0, 255, 0)

        # 어깨선(분홍색) + 양팔 스켈레톤 그리기
        cv2.line(display_frame, l_shoulder, r_shoulder, (255, 0, 255), 3)
        cv2.line(display_frame, l_shoulder, l_elbow, l_arm_color, 4)
        cv2.line(display_frame, l_elbow, l_wrist, l_arm_color, 4)
        cv2.line(display_frame, r_shoulder, r_elbow, r_arm_color, 4)
        cv2.line(display_frame, r_elbow, r_wrist, r_arm_color, 4)

        cv2.circle(display_frame, nose, 7, (0, 255, 0), -1)
        for pt in [l_shoulder, r_shoulder]:
          cv2.circle(
              display_frame, pt, 9, (255, 0, 255), -1
          )  # 어깨는 보라색 큰 점으로 표시
        for pt in [l_elbow, r_elbow]:
          cv2.circle(display_frame, pt, 7, (255, 255, 255), -1)

        cv2.circle(display_frame, l_wrist, 14, (255, 200, 0), -1)
        cv2.circle(display_frame, r_wrist, 14, (0, 140, 255), -1)

        cv2.putText(
            display_frame,
            f"{int(left_arm_angle)}d ({jab_state[0]})",
            l_elbow,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            display_frame,
            f"{int(right_arm_angle)}d ({cross_state[0]})",
            r_elbow,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
    else:
      feedback_msg = (
          "Please show your upper body (head & shoulders) to the camera!"
      )
      feedback_color = (0, 165, 255)

    # 상단 대시보드 UI 출력
    cv2.rectangle(display_frame, (0, 0), (w, 105), (20, 20, 20), -1)
    stats_text = (
        f"JAB: {jab_count}  |  CROSS: {cross_count}  |  DUCK: {duck_count}"
        f" ({duck_state})  |  SPEED: {last_punch_speed:.2f}s"
    )
    cv2.putText(
        display_frame,
        stats_text,
        (30, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        display_frame,
        feedback_msg,
        (30, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        feedback_color,
        3,
    )

    cv2.imshow(
        "AI Boxing Coach - Team 1 Task (Press 'q' to quit)", display_frame
    )

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
      break
    elif key == ord("r"):
      jab_count, cross_count, duck_count, last_punch_speed = 0, 0, 0, 0.0

except KeyboardInterrupt:
  print("\n프로그램이 정상 종료되었습니다.")
finally:
  cap.release()
  cv2.destroyAllWindows()