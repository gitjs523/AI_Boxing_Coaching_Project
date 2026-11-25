from collections import deque
from datetime import datetime
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
    ("Pillow", "PIL"),
]:
  try:
    __import__(import_name)
  except ImportError:
    print(f"[{pkg_name}] 자동 설치 중...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# [1. 모델 준비, 한글 폰트 로드 및 ONNX 설정]
# ==========================================
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
  print("모델 다운로드 완료!")


def find_korean_font_path():
  candidates = [
      "C:/Windows/Fonts/malgun.ttf",
      "C:/Windows/Fonts/malgunbd.ttf",
      "C:/Windows/Fonts/NanumGothic.ttf",
      os.path.join(SCRIPT_DIR, "NanumGothic.ttf"),
      os.path.join(SCRIPT_DIR, "malgun.ttf"),
  ]
  for path in candidates:
    if os.path.exists(path):
      return path
  return None


FONT_PATH = find_korean_font_path()
_FONT_CACHE = {}


def get_korean_font(size):
  px_size = max(10, int(round(size)))
  if px_size not in _FONT_CACHE:
    if FONT_PATH:
      _FONT_CACHE[px_size] = ImageFont.truetype(FONT_PATH, px_size)
    else:
      _FONT_CACHE[px_size] = ImageFont.load_default()
  return _FONT_CACHE[px_size]


def bgr_to_pil_rgb(color_bgr):
  return (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))


def measure_korean_text(text, font_size):
  font = get_korean_font(font_size)
  dummy = Image.new("RGB", (10, 10))
  draw = ImageDraw.Draw(dummy)
  bbox = draw.textbbox((0, 0), str(text), font=font)
  return max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])


def draw_korean_text(
    img_bgr, text, pos, font_size, color_bgr=(255, 255, 255), outline=True
):
  """OpenCV BGR 이미지 위에 한글(및 외곽선)을 렌더링합니다."""
  font = get_korean_font(font_size)
  rgb_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
  pil_img = Image.fromarray(rgb_img)
  draw = ImageDraw.Draw(pil_img)

  x, y = int(round(pos[0])), int(round(pos[1]))
  fill_rgb = bgr_to_pil_rgb(color_bgr)

  if outline:
    for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
      draw.text((x + dx, y + dy), str(text), font=font, fill=(0, 0, 0))

  draw.text((x, y), str(text), font=font, fill=fill_rgb)
  result_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
  np.copyto(img_bgr, result_bgr)


def calculate_angle(a, b, c):
  a, b, c = np.array(a), np.array(b), np.array(c)
  radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
      a[1] - b[1], a[0] - b[0]
  )
  angle = np.abs(radians * 180.0 / np.pi)
  return 360 - angle if angle > 180.0 else angle


def calculate_distance(p1, p2):
  return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def detect_movenet_letterbox(session, input_name, input_type, frame):
  """16:9 비율 왜곡을 막기 위해 192x192 Letterbox 패딩 후 원본 비율로 역변환합니다."""
  h, w, _ = frame.shape
  scale = min(192 / w, 192 / h)
  nw, nh = int(w * scale), int(h * scale)
  resized = cv2.resize(frame, (nw, nh))

  canvas = np.zeros((192, 192, 3), dtype=np.uint8)
  dw, dh = (192 - nw) // 2, (192 - nh) // 2
  canvas[dh : dh + nh, dw : dw + nw] = resized

  rgb_img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
  input_tensor = np.expand_dims(
      rgb_img.astype(np.int32 if "int32" in input_type else np.float32), axis=0
  )

  outputs = session.run(None, {input_name: input_tensor})[0]
  raw_kpts = outputs[0, 0]

  real_kpts = np.zeros_like(raw_kpts)
  for i in range(17):
    py, px = raw_kpts[i, 0] * 192.0, raw_kpts[i, 1] * 192.0
    real_kpts[i, 0] = (py - dh) / nh
    real_kpts[i, 1] = (px - dw) / nw
    real_kpts[i, 2] = raw_kpts[i, 2]
  return real_kpts


sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 2
sess_options.inter_op_num_threads = 1
session = ort.InferenceSession(
    MODEL_PATH, sess_options=sess_options, providers=["CPUExecutionProvider"]
)

input_info = session.get_inputs()[0]
input_name, input_type = input_info.name, input_info.type

# ==========================================
# [2. 웹캠 연결 (DSHOW 우선 탐색)]
# ==========================================
cap = None
for idx in [0, 1, 2]:
  temp_cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
  if temp_cap.isOpened():
    ret, test_frame = temp_cap.read()
    if ret and test_frame is not None and test_frame.size > 0:
      cap = temp_cap
      break
    temp_cap.release()

if cap is None:
  for idx in [0, 1, 2]:
    temp_cap = cv2.VideoCapture(idx)
    if temp_cap.isOpened():
      ret, test_frame = temp_cap.read()
      if ret and test_frame is not None and test_frame.size > 0:
        cap = temp_cap
        break
      temp_cap.release()

if cap is None or not cap.isOpened():
  print("\n[카메라 연결 실패] 웹캠 프레임을 읽어올 수 없습니다.")
  sys.exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ==========================================
# [3. 복싱 상태 머신, 필터, 궤적, HUD 변수]
# ==========================================
jab_count, cross_count, duck_count = 0, 0, 0
jab_state, cross_state, duck_state = "GUARD", "GUARD", "STAND"
jab_start_time, cross_start_time, last_punch_speed = 0.0, 0.0, 0.0
base_torso_height = 0.0

feedback_msg = "상체가 보이도록 조금만 뒤로 물러서 주세요!"
feedback_color = (0, 255, 255)

prev_kpts = None
last_valid_kpts = None
missing_frames = 0
MAX_MISSING_FRAMES = 5

FILTER_MODES = ["ADAPTIVE", "FIXED", "OFF"]
DUCK_NAMES_KR = {
    "STAND": "대기",
    "DUCKED": "회피중",
}
filter_mode_idx = 0
ALPHA_MIN = 0.22
ALPHA_MAX = 0.92

left_wrist_trail = deque(maxlen=14)
right_wrist_trail = deque(maxlen=14)
prev_l_wrist, prev_r_wrist = None, None

prev_loop_start = None
fps_ema = 30.0
infer_ms = 0.0
saved_notice_until = 0.0
saved_filename = ""

try:
  while cap.isOpened():
    loop_start = time.perf_counter()
    if prev_loop_start is not None:
      dt = max(loop_start - prev_loop_start, 1e-6)
      fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt)
    prev_loop_start = loop_start

    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
      continue

    infer_start = time.perf_counter()
    raw_kpts = detect_movenet_letterbox(session, input_name, input_type, frame)
    infer_ms = (time.perf_counter() - infer_start) * 1000.0

    kpts = raw_kpts.copy()
    current_mode = FILTER_MODES[filter_mode_idx]

    # 현재 프레임에서 적응형 필터가 어떻게 조절 중인지 확인
    adaptive_hint = ""
    if current_mode == "ADAPTIVE":
      if prev_kpts is not None:
        wrist_move = max(
            float(np.linalg.norm(kpts[9, :2] - prev_kpts[9, :2])),
            float(np.linalg.norm(kpts[10, :2] - prev_kpts[10, :2])),
        )
        adaptive_hint = " · 고속추적" if wrist_move >= 0.012 else " · 떨림억제"
      else:
        adaptive_hint = " · 자동대기"

    if prev_kpts is not None and current_mode != "OFF":
      raw_pos = raw_kpts[:, :2]
      conf = raw_kpts[:, 2]

      if current_mode == "FIXED":
        kpts[:, :2] = 0.35 * raw_pos + 0.65 * prev_kpts[:, :2]
      else:
        dist = np.linalg.norm(raw_pos - prev_kpts[:, :2], axis=1)
        dynamic_alpha = np.clip(
            ALPHA_MIN + (dist - 0.004) * ((ALPHA_MAX - ALPHA_MIN) / 0.021),
            ALPHA_MIN,
            ALPHA_MAX,
        )
        is_moving = dist >= 0.012
        rest_conf_weight = np.clip((conf - 0.20) / 0.45, 0.45, 1.0)
        effective_alpha = np.where(
            is_moving, dynamic_alpha, dynamic_alpha * rest_conf_weight
        )

        is_false_jump = (dist > 0.28) & (conf < 0.35)
        effective_alpha = np.where(is_false_jump, 0.10, effective_alpha)

        kpts[:, :2] = (
            effective_alpha[:, None] * raw_pos
            + (1.0 - effective_alpha[:, None]) * prev_kpts[:, :2]
        )

    prev_kpts = kpts.copy()

    display_frame = cv2.flip(frame, 1)
    h, w, _ = display_frame.shape

    core_visible = kpts[0][2] > 0.30 and (kpts[5][2] > 0.20 or kpts[6][2] > 0.20)
    limbs_visible = all(kpts[i][2] > 0.08 for i in [7, 8, 9, 10])
    is_detected = core_visible and limbs_visible

    if is_detected:
      missing_frames = 0
      last_valid_kpts = kpts.copy()
      active_kpts = kpts
    elif core_visible:
      missing_frames = 0
      if last_valid_kpts is not None:
        blended = kpts.copy()
        for idx in [7, 8, 9, 10]:
          if kpts[idx][2] <= 0.08:
            blended[idx] = last_valid_kpts[idx]
        last_valid_kpts = blended.copy()
        active_kpts = blended
        is_detected = True
      else:
        last_valid_kpts = kpts.copy()
        active_kpts = kpts
        is_detected = True
    else:
      missing_frames += 1
      active_kpts = (
          last_valid_kpts if missing_frames <= MAX_MISSING_FRAMES else None
      )

    if active_kpts is not None:
      def get_mirror_pt(idx):
        y, x, _ = active_kpts[idx]
        return [int((1.0 - x) * w), int(y * h)]

      nose = get_mirror_pt(0)
      l_shoulder, r_shoulder = get_mirror_pt(5), get_mirror_pt(6)
      l_elbow, r_elbow = get_mirror_pt(7), get_mirror_pt(8)
      l_wrist, r_wrist = get_mirror_pt(9), get_mirror_pt(10)

      shoulder_center = [
          int((l_shoulder[0] + r_shoulder[0]) / 2),
          int((l_shoulder[1] + r_shoulder[1]) / 2),
      ]
      torso_height = shoulder_center[1] - nose[1]

      left_arm_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
      right_arm_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
      l_arm_color = (255, 255, 0)
      r_arm_color = (0, 255, 0)

      if torso_height < 25 and base_torso_height == 0.0:
        feedback_msg = "가까워요! 양쪽 어깨가 보이게 뒤로 물러서세요."
        feedback_color = (0, 165, 255)
      else:
        if is_detected and duck_state == "STAND" and torso_height > 35:
          if base_torso_height == 0.0:
            base_torso_height = float(torso_height)
          else:
            base_torso_height = (
                0.95 * base_torso_height + 0.05 * torso_height
            )

        ref_height = max(base_torso_height, float(torso_height), 40.0)

        r_guard_dist = calculate_distance(r_wrist, nose) / ref_height
        l_guard_dist = calculate_distance(l_wrist, nose) / ref_height
        l_reach_ratio = calculate_distance(l_shoulder, l_wrist) / ref_height
        r_reach_ratio = calculate_distance(r_shoulder, r_wrist) / ref_height

        l_velocity = (
            calculate_distance(l_wrist, prev_l_wrist) / ref_height
            if prev_l_wrist
            else 0.0
        )
        r_velocity = (
            calculate_distance(r_wrist, prev_r_wrist) / ref_height
            if prev_r_wrist
            else 0.0
        )

        is_r_hand_low = r_wrist[1] > r_shoulder[1]
        is_l_hand_low = l_wrist[1] > l_shoulder[1]

        now = time.time()

        if is_detected:
          prev_l_wrist, prev_r_wrist = l_wrist.copy(), r_wrist.copy()

          l_norm_speed = 0.0
          if left_wrist_trail:
            dist_l = calculate_distance(
                (left_wrist_trail[-1][0], left_wrist_trail[-1][1]), l_wrist
            )
            if dist_l > w * 0.30:
              left_wrist_trail.clear()
            else:
              l_norm_speed = dist_l / ref_height

          r_norm_speed = 0.0
          if right_wrist_trail:
            dist_r = calculate_distance(
                (right_wrist_trail[-1][0], right_wrist_trail[-1][1]), r_wrist
            )
            if dist_r > w * 0.30:
              right_wrist_trail.clear()
            else:
              r_norm_speed = dist_r / ref_height

          left_wrist_trail.append((l_wrist[0], l_wrist[1], l_norm_speed))
          right_wrist_trail.append((r_wrist[0], r_wrist[1], r_norm_speed))

          # 1. 연속 더킹/위빙
          duck_ratio = torso_height / max(base_torso_height, 40.0)
          if duck_ratio < 0.68:
            if duck_state == "STAND":
              duck_count += 1
              duck_state = "DUCKED"
            feedback_msg = "더킹 / 위빙 동작 감지!"
            feedback_color = (255, 255, 0)
          elif duck_ratio > 0.82:
            if duck_state == "DUCKED":
              duck_state = "STAND"

          # 2. 왼손 잽(Jab) 판정
          if (left_arm_angle > 90 or l_velocity > 0.35) and jab_state == "GUARD":
            if jab_start_time == 0.0 or (now - jab_start_time > 0.8):
              jab_start_time = now

          is_left_extended = (
              (left_arm_angle > 125)
              or (left_arm_angle > 110 and l_reach_ratio > 1.65)
              or (
                  l_velocity > 0.55
                  and left_arm_angle > 105
                  and l_reach_ratio > 1.45
              )
          )

          if is_left_extended:
            if jab_state == "GUARD":
              jab_count += 1
              jab_state = "EXTENDED"
              if jab_start_time > 0:
                last_punch_speed = round(
                    max(0.04, min(0.80, now - jab_start_time)), 2
                )
                jab_start_time = 0.0

            if r_guard_dist > 1.8 or is_r_hand_low:
              feedback_msg = "경고: 오른손 가드를 올리세요!"
              feedback_color, r_arm_color = (0, 0, 255), (0, 0, 255)
            else:
              feedback_msg = f"좋은 잽! ({last_punch_speed:.2f}초)"
              feedback_color = (0, 255, 0)
          elif left_arm_angle < 115 and l_reach_ratio < 1.55:
            jab_state, jab_start_time = "GUARD", 0.0

          # 3. 오른손 스트레이트(Cross) 판정
          if (
              right_arm_angle > 90 or r_velocity > 0.35
          ) and cross_state == "GUARD":
            if cross_start_time == 0.0 or (now - cross_start_time > 0.8):
              cross_start_time = now

          is_right_extended = (
              (right_arm_angle > 125)
              or (right_arm_angle > 110 and r_reach_ratio > 1.65)
              or (
                  r_velocity > 0.55
                  and right_arm_angle > 105
                  and r_reach_ratio > 1.45
              )
          )

          if is_right_extended:
            if cross_state == "GUARD":
              cross_count += 1
              cross_state = "EXTENDED"
              if cross_start_time > 0:
                last_punch_speed = round(
                    max(0.04, min(0.80, now - cross_start_time)), 2
                )
                cross_start_time = 0.0

            if l_guard_dist > 1.8 or is_l_hand_low:
              feedback_msg = "경고: 왼손 가드를 올리세요!"
              feedback_color, l_arm_color = (0, 0, 255), (0, 0, 255)
            else:
              feedback_msg = f"좋은 스트레이트! ({last_punch_speed:.2f}초)"
              feedback_color = (0, 255, 0)
          elif right_arm_angle < 115 and r_reach_ratio < 1.55:
            cross_state, cross_start_time = "GUARD", 0.0

          # 4. 평상시 스탠스 체크
          if (
              jab_state == "GUARD"
              and cross_state == "GUARD"
              and duck_state == "STAND"
          ):
            if is_l_hand_low or is_r_hand_low:
              feedback_msg = "가드를 올려 턱을 보호하세요!"
              feedback_color = (0, 165, 255)
            elif (
                "경고" in feedback_msg
                or "가드를" in feedback_msg
                or "물러서" in feedback_msg
            ):
              feedback_msg = "자세 양호! 준비 완료."
              feedback_color = (0, 255, 0)

      # ==========================================
      # [4. 속도 반응형 주먹 궤적 잔상 & 스켈레톤]
      # ==========================================
      for i in range(1, len(left_wrist_trail)):
        ratio = i / float(len(left_wrist_trail))
        speed_boost = min(1.0, left_wrist_trail[i][2] / 0.35)
        thickness = max(1, int(ratio * (4 + 7 * speed_boost)))
        b_col = int((220 - 120 * speed_boost) * ratio)
        g_col = int((200 + 55 * speed_boost) * ratio)
        r_col = int((255 * speed_boost) * ratio)
        cv2.line(
            display_frame,
            (left_wrist_trail[i - 1][0], left_wrist_trail[i - 1][1]),
            (left_wrist_trail[i][0], left_wrist_trail[i][1]),
            (b_col, g_col, r_col),
            thickness,
            cv2.LINE_AA,
        )

      for i in range(1, len(right_wrist_trail)):
        ratio = i / float(len(right_wrist_trail))
        speed_boost = min(1.0, right_wrist_trail[i][2] / 0.35)
        thickness = max(1, int(ratio * (4 + 7 * speed_boost)))
        b_col = int((40 * speed_boost) * ratio)
        g_col = int((150 - 60 * speed_boost) * ratio)
        r_col = int((200 + 55 * speed_boost) * ratio)
        cv2.line(
            display_frame,
            (right_wrist_trail[i - 1][0], right_wrist_trail[i - 1][1]),
            (right_wrist_trail[i][0], right_wrist_trail[i][1]),
            (b_col, g_col, r_col),
            thickness,
            cv2.LINE_AA,
        )

      cv2.line(display_frame, nose, shoulder_center, (255, 0, 255), 3, cv2.LINE_AA)
      cv2.circle(display_frame, shoulder_center, 5, (255, 0, 255), -1, cv2.LINE_AA)
      cv2.line(display_frame, l_shoulder, r_shoulder, (255, 0, 255), 3, cv2.LINE_AA)

      cv2.line(display_frame, l_shoulder, l_elbow, l_arm_color, 4, cv2.LINE_AA)
      cv2.line(display_frame, l_elbow, l_wrist, l_arm_color, 4, cv2.LINE_AA)
      cv2.line(display_frame, r_shoulder, r_elbow, r_arm_color, 4, cv2.LINE_AA)
      cv2.line(display_frame, r_elbow, r_wrist, r_arm_color, 4, cv2.LINE_AA)

      for pt in [nose, l_shoulder, r_shoulder, l_elbow, r_elbow]:
        cv2.circle(display_frame, pt, 7, (255, 255, 255), -1, cv2.LINE_AA)
      cv2.circle(display_frame, l_wrist, 14, (255, 200, 0), -1, cv2.LINE_AA)
      cv2.circle(display_frame, r_wrist, 14, (0, 140, 255), -1, cv2.LINE_AA)

      angle_font_px = max(14, int(round(18 * (w / 1280.0))))
      draw_korean_text(
          display_frame,
          f"{int(left_arm_angle)}도",
          (l_elbow[0] + 10, l_elbow[1] - 8),
          angle_font_px,
          (255, 255, 255),
      )
      draw_korean_text(
          display_frame,
          f"{int(right_arm_angle)}도",
          (r_elbow[0] + 10, r_elbow[1] - 8),
          angle_font_px,
          (255, 255, 255),
      )
    else:
      left_wrist_trail.clear()
      right_wrist_trail.clear()
      prev_l_wrist, prev_r_wrist = None, None
      if missing_frames > 15:
        prev_kpts = None

    # ==========================================
    # [5. 반응형 상단 헤더 & 하단 한글 HUD]
    # ==========================================
    header_h = max(76, int(h * 0.145))
    ui_scale = w / 1280.0
    cv2.rectangle(display_frame, (0, 0), (w, header_h), (20, 20, 20), -1)

    duck_state_kr = DUCK_NAMES_KR.get(duck_state, duck_state)
    stats_text = (
        f"잽: {jab_count}  |  스트레이트: {cross_count}  |  더킹: {duck_count}"
        f" ({duck_state_kr})  |  속도: {last_punch_speed:.2f}초"
    )
    max_st_w = max(100, w - int(36 * ui_scale))
    stats_px = max(14, int(round(25 * ui_scale)))
    st_w, st_h = measure_korean_text(stats_text, stats_px)
    if st_w > max_st_w:
      stats_px = max(12, int(stats_px * (max_st_w / float(st_w))))
      st_w, st_h = measure_korean_text(stats_text, stats_px)

    st_x = int(18 * ui_scale)
    st_y = int(header_h * 0.14)
    draw_korean_text(
        display_frame,
        stats_text,
        (st_x, st_y),
        stats_px,
        (255, 255, 255),
        outline=False,
    )

    fb_px = max(15, int(round(27 * ui_scale)))
    fb_w, fb_h = measure_korean_text(feedback_msg, fb_px)
    if fb_w > max_st_w:
      fb_px = max(12, int(fb_px * (max_st_w / float(fb_w))))
      fb_w, fb_h = measure_korean_text(feedback_msg, fb_px)

    fb_x = int(18 * ui_scale)
    fb_y = int(header_h * 0.56)
    draw_korean_text(
        display_frame,
        feedback_msg,
        (fb_x, fb_y),
        fb_px,
        feedback_color,
        outline=False,
    )

    # 하단 좌측: 자동 작동 여부 강조 배지, 하단 우측: FPS & 추론 시간
    hud_px = max(12, int(round(17 * ui_scale)))
    pad_x, pad_y = max(6, int(10 * ui_scale)), max(4, int(6 * ui_scale))

    if time.perf_counter() < saved_notice_until:
      filter_label = f"저장 완료: {saved_filename}"
      filter_color = (0, 255, 120)
      box_border_color = (0, 255, 120)
    elif current_mode == "ADAPTIVE":
      filter_label = (
          f"[자동] 적응형 필터{adaptive_hint}  |  전환('f') · 초기화('r') ·"
          " 캡처('s')"
      )
      filter_color = (0, 255, 180)
      box_border_color = (0, 255, 180)
    elif current_mode == "FIXED":
      filter_label = (
          "[수동] 고정형 필터(0.35)  |  전환('f') · 초기화('r') · 캡처('s')"
      )
      filter_color = (0, 210, 255)
      box_border_color = (0, 210, 255)
    else:
      filter_label = (
          "[수동] 필터 꺼짐  |  전환('f') · 초기화('r') · 캡처('s')"
      )
      filter_color = (180, 180, 180)
      box_border_color = (110, 110, 110)

    fw, fh = measure_korean_text(filter_label, hud_px)
    l_box_x1 = int(12 * ui_scale)
    l_box_y2 = h - int(10 * ui_scale)
    l_box_y1 = l_box_y2 - fh - pad_y * 2
    l_box_x2 = l_box_x1 + fw + pad_x * 2

    cv2.rectangle(
        display_frame, (l_box_x1, l_box_y1), (l_box_x2, l_box_y2), (20, 20, 20), -1
    )
    cv2.rectangle(
        display_frame,
        (l_box_x1, l_box_y1),
        (l_box_x2, l_box_y2),
        box_border_color,
        2,
    )
    draw_korean_text(
        display_frame,
        filter_label,
        (l_box_x1 + pad_x, l_box_y1 + pad_y),
        hud_px,
        filter_color,
        outline=False,
    )

    perf_label = f"프레임: {fps_ema:.1f}  |  추론: {infer_ms:.1f}ms"
    pw, ph = measure_korean_text(perf_label, hud_px)
    r_box_x2 = w - int(12 * ui_scale)
    r_box_x1 = r_box_x2 - pw - pad_x * 2
    r_box_y2 = l_box_y2
    r_box_y1 = r_box_y2 - ph - pad_y * 2

    if r_box_x1 > l_box_x2 + int(8 * ui_scale):
      cv2.rectangle(
          display_frame,
          (r_box_x1, r_box_y1),
          (r_box_x2, r_box_y2),
          (20, 20, 20),
          -1,
      )
      draw_korean_text(
          display_frame,
          perf_label,
          (r_box_x1 + pad_x, r_box_y1 + pad_y),
          hud_px,
          (220, 220, 220),
          outline=False,
      )

    cv2.imshow(
        "AI Boxing Coach ('f': Filter, 'r': Reset, 's': Shot, 'q': Quit)",
        display_frame,
    )

    elapsed_ms = (time.perf_counter() - loop_start) * 1000
    wait_ms = max(1, int(33 - elapsed_ms))
    key = cv2.waitKey(wait_ms) & 0xFF
    if key == ord("q"):
      break
    elif key == ord("f"):
      filter_mode_idx = (filter_mode_idx + 1) % len(FILTER_MODES)
      prev_kpts = None
    elif key == ord("r"):
      jab_count, cross_count, duck_count = 0, 0, 0
      jab_state, cross_state, duck_state = "GUARD", "GUARD", "STAND"
      jab_start_time, cross_start_time, last_punch_speed = 0.0, 0.0, 0.0
      base_torso_height = 0.0
      left_wrist_trail.clear()
      right_wrist_trail.clear()
      prev_l_wrist, prev_r_wrist = None, None
      feedback_msg = "기록이 초기화되었습니다! 가드를 유지하세요."
      feedback_color = (0, 255, 0)
    elif key == ord("s"):
      saved_filename = f"capture_{datetime.now().strftime('%H%M%S')}.png"
      save_path = os.path.join(SCRIPT_DIR, saved_filename)
      is_ok, buf = cv2.imencode(".png", display_frame)
      if is_ok:
        buf.tofile(save_path)
        saved_notice_until = time.perf_counter() + 1.5
        print(f"스크린샷 저장 완료: {save_path}")

except KeyboardInterrupt:
  pass
finally:
  cap.release()
  cv2.destroyAllWindows()