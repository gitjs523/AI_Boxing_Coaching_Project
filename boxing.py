from collections import deque
import math
import os
import sys
import time
import urllib.request
import cv2
import numpy as np
import onnxruntime as ort

# ==========================================
# [모델 로드 & ONNX CPU 스레드 최적화]
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "movenet_singlepose_lightning_4.onnx")

URLS = [
    "https://huggingface.co/Xenova/movenet-singlepose-lightning/resolve/main/onnx/model.onnx",
]

if not os.path.exists(MODEL_PATH):
  print("구글 MoveNet(ONNX) 포즈 모델을 다운로드 중입니다...")
  req = urllib.request.Request(URLS[0], headers={"User-Agent": "Mozilla/5.0"})
  with (
      urllib.request.urlopen(req) as response,
      open(MODEL_PATH, "wb") as out_file,
  ):
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


def draw_outlined_text(img, text, pos, font, scale, color, thickness):
  x, y = pos
  for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
    cv2.putText(
        img,
        text,
        (x + dx, y + dy),
        font,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )
  cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


# CPU 점유율 폭등 방지를 위한 ONNX 스레드 수 제한
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 2
sess_options.inter_op_num_threads = 1
session = ort.InferenceSession(
    MODEL_PATH, sess_options=sess_options, providers=["CPUExecutionProvider"]
)

input_info = session.get_inputs()[0]
input_name = input_info.name
input_type = input_info.type

# ==========================================
# [웹캠 연결 (DSHOW 우선 탐색)]
# ==========================================
cap = None
for idx in [0, 1, 2]:
  temp_cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
  if temp_cap.isOpened():
    ret, test_frame = temp_cap.read()
    if ret and test_frame is not None and test_frame.size > 0:
      cap = temp_cap
      print(f"웹캠 연결 성공! (DSHOW 인덱스: {idx})")
      break
    temp_cap.release()

if cap is None:
  for idx in [0, 1, 2]:
    temp_cap = cv2.VideoCapture(idx)
    if temp_cap.isOpened():
      ret, test_frame = temp_cap.read()
      if ret and test_frame is not None and test_frame.size > 0:
        cap = temp_cap
        print(f"웹캠 연결 성공! (기본 인덱스: {idx})")
        break
      temp_cap.release()

if cap is None or not cap.isOpened():
  print("\n[카메라 연결 실패] 웹캠 프레임을 읽어올 수 없습니다.")
  sys.exit()

jab_count = 0
jab_state = "GUARD"
feedback_msg = "Ready! Keep your guard up."
feedback_color = (0, 255, 0)

# ==========================================
# [팀원 2 파트 변수: EMA 필터 및 주먹 궤적]
# ==========================================
prev_kpts = None
alpha = 0.35  # EMA 평활화 계수 (0.1 ~ 0.5)
ema_enabled = True  # 'f' 키로 필터 ON/OFF 전환

left_wrist_trail = deque(maxlen=14)
right_wrist_trail = deque(maxlen=14)

while cap.isOpened():
  loop_start = time.perf_counter()

  ret, frame = cap.read()
  if not ret or frame is None or frame.size == 0:
    continue

  # MoveNet 입력 변환: 192x192 RGB
  img = cv2.resize(frame, (192, 192))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  if "int32" in input_type:
    input_tensor = np.expand_dims(img.astype(np.int32), axis=0)
  else:
    input_tensor = np.expand_dims(img.astype(np.float32), axis=0)

  outputs = session.run(None, {input_name: input_tensor})[0]
  raw_kpts = outputs[0, 0]  # (17, 3) -> [y, x, score]

  # ==========================================
  # [팀원 2 기능 1] 관절 좌표 EMA 저역통과 필터 (떨림 보정)
  # ==========================================
  kpts = raw_kpts.copy()
  if ema_enabled and prev_kpts is not None:
    kpts[:, :2] = alpha * raw_kpts[:, :2] + (1 - alpha) * prev_kpts[:, :2]
  prev_kpts = kpts.copy()

  display_frame = cv2.flip(frame, 1)
  h, w, _ = display_frame.shape

  required_indices = [0, 5, 6, 7, 8, 9, 10]
  if all(kpts[i][2] > 0.25 for i in required_indices):

    def get_mirror_pt(idx):
      y, x, _ = kpts[idx]
      return [int((1.0 - x) * w), int(y * h)]

    nose = get_mirror_pt(0)
    l_shoulder, r_shoulder = get_mirror_pt(5), get_mirror_pt(6)
    l_elbow, r_elbow = get_mirror_pt(7), get_mirror_pt(8)
    l_wrist, r_wrist = get_mirror_pt(9), get_mirror_pt(10)

    # 인식 튐(갑작스러운 좌표 점프) 방지 후 궤적 추가
    if (
        left_wrist_trail
        and calculate_distance(left_wrist_trail[-1], l_wrist) > w * 0.25
    ):
      left_wrist_trail.clear()
    if (
        right_wrist_trail
        and calculate_distance(right_wrist_trail[-1], r_wrist) > w * 0.25
    ):
      right_wrist_trail.clear()

    left_wrist_trail.append(tuple(l_wrist))
    right_wrist_trail.append(tuple(r_wrist))

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

      # ==========================================
      # [팀원 2 기능 2] 주먹 궤적 잔상 시각화 (Motion Trail)
      # ==========================================
      for i in range(1, len(left_wrist_trail)):
        ratio = i / float(len(left_wrist_trail))
        thickness = max(1, int(ratio * 8))
        trail_color = (int(255 * ratio), int(255 * ratio), 0)
        cv2.line(
            display_frame,
            left_wrist_trail[i - 1],
            left_wrist_trail[i],
            trail_color,
            thickness,
            cv2.LINE_AA,
        )

      for i in range(1, len(right_wrist_trail)):
        ratio = i / float(len(right_wrist_trail))
        thickness = max(1, int(ratio * 8))
        trail_color = (0, int(165 * ratio), int(255 * ratio))
        cv2.line(
            display_frame,
            right_wrist_trail[i - 1],
            right_wrist_trail[i],
            trail_color,
            thickness,
            cv2.LINE_AA,
        )

      # 관절 스켈레톤 라인 그리기
      cv2.line(
          display_frame, l_shoulder, l_elbow, (255, 255, 0), 4, cv2.LINE_AA
      )
      cv2.line(display_frame, l_elbow, l_wrist, (255, 255, 0), 4, cv2.LINE_AA)
      cv2.line(
          display_frame, r_shoulder, r_elbow, feedback_color, 4, cv2.LINE_AA
      )
      cv2.line(display_frame, r_elbow, r_wrist, feedback_color, 4, cv2.LINE_AA)

      for pt in [l_shoulder, l_elbow, l_wrist, r_shoulder, r_elbow, r_wrist]:
        cv2.circle(display_frame, pt, 7, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(display_frame, pt, 7, (30, 30, 30), 1, cv2.LINE_AA)

      # 관절 각도 텍스트
      draw_outlined_text(
          display_frame,
          f"{int(left_arm_angle)} deg",
          (l_elbow[0] + 12, l_elbow[1] + 5),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.6,
          (255, 255, 255),
          2,
      )
  else:
    left_wrist_trail.clear()
    right_wrist_trail.clear()

  # ==========================================
  # [반응형 상단 UI 헤더] 해상도/문구 길이에 맞춘 자동 정렬
  # ==========================================
  header_h = int(h * 0.13)
  scale_base = w / 640.0
  cv2.rectangle(display_frame, (0, 0), (w, header_h), (20, 20, 20), -1)

  # 좌측: JAB COUNT
  jab_text = f"JAB COUNT: {jab_count}"
  jab_scale = 0.8 * scale_base
  jab_thick = max(2, int(2 * scale_base))
  (jab_w, jab_h), _ = cv2.getTextSize(
      jab_text, cv2.FONT_HERSHEY_SIMPLEX, jab_scale, jab_thick
  )
  jab_x = int(18 * scale_base)
  jab_y = int((header_h + jab_h) / 2)
  cv2.putText(
      display_frame,
      jab_text,
      (jab_x, jab_y),
      cv2.FONT_HERSHEY_SIMPLEX,
      jab_scale,
      (255, 255, 255),
      jab_thick,
      cv2.LINE_AA,
  )

  # 우측: 피드백 문구
  margin_x = int(18 * scale_base)
  gap_x = int(20 * scale_base)
  max_fb_w = w - (jab_x + jab_w + gap_x) - margin_x

  fb_scale = 0.65 * scale_base
  fb_thick = max(2, int(2 * scale_base))
  (fb_w, fb_h), _ = cv2.getTextSize(
      feedback_msg, cv2.FONT_HERSHEY_SIMPLEX, fb_scale, fb_thick
  )

  if fb_w > max_fb_w and max_fb_w > 0:
    fb_scale = fb_scale * (max_fb_w / float(fb_w))
    (fb_w, fb_h), _ = cv2.getTextSize(
        feedback_msg, cv2.FONT_HERSHEY_SIMPLEX, fb_scale, fb_thick
    )

  fb_x = w - fb_w - margin_x
  fb_y = int((header_h + fb_h) / 2)
  cv2.putText(
      display_frame,
      feedback_msg,
      (fb_x, fb_y),
      cv2.FONT_HERSHEY_SIMPLEX,
      fb_scale,
      feedback_color,
      fb_thick,
      cv2.LINE_AA,
  )

  # ==========================================
  # [하단 좌측 상태 배지]
  # ==========================================
  filter_label = (
      f"EMA Filter: {'ON' if ema_enabled else 'OFF'} (Press 'f' to toggle)"
  )
  filter_color = (0, 255, 200) if ema_enabled else (180, 180, 180)
  f_scale = 0.52 * scale_base
  f_thick = max(1, int(1.5 * scale_base))
  (fw, fh), _ = cv2.getTextSize(
      filter_label, cv2.FONT_HERSHEY_SIMPLEX, f_scale, f_thick
  )

  pad_x, pad_y = int(10 * scale_base), int(6 * scale_base)
  box_x1 = int(12 * scale_base)
  box_y2 = h - int(10 * scale_base)
  box_y1 = box_y2 - fh - pad_y * 2
  box_x2 = box_x1 + fw + pad_x * 2

  cv2.rectangle(
      display_frame, (box_x1, box_y1), (box_x2, box_y2), (20, 20, 20), -1
  )
  cv2.putText(
      display_frame,
      filter_label,
      (box_x1 + pad_x, box_y2 - pad_y),
      cv2.FONT_HERSHEY_SIMPLEX,
      f_scale,
      filter_color,
      f_thick,
      cv2.LINE_AA,
  )

  cv2.imshow(
      "AI Boxing Coach - DIP Filter & Motion Trail (Press 'q' to quit)",
      display_frame,
  )

  # 웹캠 30 FPS(33ms) 주기에 맞춰 루프 대기 시간 동기화
  elapsed_ms = (time.perf_counter() - loop_start) * 1000
  wait_ms = max(1, int(33 - elapsed_ms))
  key = cv2.waitKey(wait_ms) & 0xFF
  if key == ord("q"):
    break
  elif key == ord("f"):
    ema_enabled = not ema_enabled
    prev_kpts = None

cap.release()
cv2.destroyAllWindows()

