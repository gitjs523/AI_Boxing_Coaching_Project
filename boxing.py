from collections import deque
from datetime import datetime
import math
import os
import sys
import time
import urllib.request
import cv2
import numpy as np
import onnxruntime as ort
from preprocessing import FramePreprocessor
from mitt_game import MittGame
from report import SessionStats
from glove_detector import detect_gloves, fuse_wrist
from ui import CoachUI

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

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
ui = CoachUI()
preprocessor = FramePreprocessor()
game = MittGame()
stats = SessionStats(version="sim-test")
hsv_enabled = False
clahe_mode = 0
clahe_active = False
jab_count = 0
cross_state = "WAIT_GUARD"
duck_state = "STAND"
base_torso_height = 0.0
prev_hands = {}
jab_state = "WAIT_GUARD"
feedback_msg = "가드를 올려 주세요"
feedback_color = (0, 255, 0)

# ==========================================
# [저지연 적응형 EMA 필터 & 스켈레톤 보간 변수]
# ==========================================
prev_kpts = None
last_valid_kpts = None
missing_frames = 0
MAX_MISSING_FRAMES = 5  # 순간적인 인식 누락 시 최대 5프레임(~0.16초) 스켈레톤 유지

# 필터 모드: "ADAPTIVE" -> "FIXED" (기존 0.35) -> "OFF"
FILTER_MODES = ["ADAPTIVE", "FIXED", "OFF"]
filter_mode_idx = 0

ALPHA_MIN = 0.22  # 정지/가드 시 미세 떨림 억제
ALPHA_MAX = 0.92  # 펀치 이동 시 지연 없는 즉각 추적

# (x, y, norm_speed) 형태로 저장
left_wrist_trail = deque(maxlen=14)
right_wrist_trail = deque(maxlen=14)

prev_loop_start = None
fps_ema = 30.0
infer_ms = 0.0
saved_notice_until = 0.0
saved_filename = ""

while cap.isOpened():
    loop_start = time.perf_counter()
    if prev_loop_start is not None:
        dt = max(loop_start - prev_loop_start, 1e-6)
        fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt)
    prev_loop_start = loop_start

    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        print("웹캠 프레임 읽기 실패")
        break

    stats.frames += 1
    if clahe_mode == 2:
        brightness = cv2.cvtColor(cv2.resize(frame, (80, 60)), cv2.COLOR_BGR2GRAY).mean()
        clahe_active = brightness < (95 if clahe_active else 75)
    else:
        clahe_active = clahe_mode == 1
    preprocessor.enabled = clahe_active
    stats.clahe_frames += int(clahe_active)
    frame = preprocessor.apply(frame)
    infer_start = time.perf_counter()
    raw_kpts = detect_movenet_letterbox(session, input_name, input_type, frame)
    infer_ms = (time.perf_counter() - infer_start) * 1000.0
    stats.infer_total += infer_ms

    # ==========================================
    # [저지연(Zero-Lag) 적응형 EMA 필터]
    # ==========================================
    kpts = raw_kpts.copy()
    current_mode = FILTER_MODES[filter_mode_idx]

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

            # 정지 상태에서만 낮은 신뢰도 떨림을 억제하고 이동 중에는 추적 속도 유지
            is_moving = dist >= 0.012
            rest_conf_weight = np.clip((conf - 0.20) / 0.45, 0.45, 1.0)
            effective_alpha = np.where(
                is_moving, dynamic_alpha, dynamic_alpha * rest_conf_weight
            )

            # 신뢰도가 매우 낮으면서 화면 28% 이상 순간이동하는 오인식만 억제
            is_false_jump = (dist > 0.28) & (conf < 0.35)
            effective_alpha = np.where(is_false_jump, 0.10, effective_alpha)

            kpts[:, :2] = (
                effective_alpha[:, None] * raw_pos
                + (1.0 - effective_alpha[:, None]) * prev_kpts[:, :2]
            )

    prev_kpts = kpts.copy()

    display_frame = cv2.flip(frame, 1)
    h, w, _ = display_frame.shape

    # Team threshold; only current observations qualify for scoring.
    core_visible = kpts[0,2] > .30 and all(kpts[i,2] > .20 for i in (5,6))
    fresh = {side: bool(core_visible and all(kpts[i,2] > .08 for i in ids))
             for side,ids in [('left',(7,9)),('right',(8,10))]}
    is_detected = core_visible and all(fresh.values())
    if core_visible:
        missing_frames = 0
        active_kpts = kpts.copy()
        if last_valid_kpts is not None:
            for i in (7,8,9,10):
                if kpts[i,2] <= .08: active_kpts[i] = last_valid_kpts[i]
        if is_detected: last_valid_kpts = kpts.copy()
    else:
        missing_frames += 1
        active_kpts = last_valid_kpts if missing_frames <= MAX_MISSING_FRAMES else None

    hands = {"left": None, "right": None}
    angles = {}
    shoulder_width = 0
    valid_pose = False
    extended = {}
    if active_kpts is not None:

        def get_mirror_pt(idx):
            y, x, _ = active_kpts[idx]
            return [int((1.0 - x) * w), int(y * h)]

        nose = get_mirror_pt(0)
        l_shoulder, r_shoulder = get_mirror_pt(5), get_mirror_pt(6)
        l_elbow, r_elbow = get_mirror_pt(7), get_mirror_pt(8)
        l_wrist, r_wrist = get_mirror_pt(9), get_mirror_pt(10)

        shoulder_width = calculate_distance(l_shoulder, r_shoulder)
        norm_base = max(shoulder_width, 1.0)

        # 실제 새 프레임이 감지되었을 때만 궤적 및 카운트 갱신
        if is_detected:
            l_norm_speed = 0.0
            if left_wrist_trail:
                dist_l = calculate_distance(
                    (left_wrist_trail[-1][0], left_wrist_trail[-1][1]), l_wrist
                )
                if dist_l > w * 0.30:
                    left_wrist_trail.clear()
                else:
                    l_norm_speed = dist_l / norm_base

            r_norm_speed = 0.0
            if right_wrist_trail:
                dist_r = calculate_distance(
                    (right_wrist_trail[-1][0], right_wrist_trail[-1][1]), r_wrist
                )
                if dist_r > w * 0.30:
                    right_wrist_trail.clear()
                else:
                    r_norm_speed = dist_r / norm_base

            left_wrist_trail.append((l_wrist[0], l_wrist[1], l_norm_speed))
            right_wrist_trail.append((r_wrist[0], r_wrist[1], r_norm_speed))

        if shoulder_width > 20:
            left_arm_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
            r_guard_dist = calculate_distance(r_wrist, nose) / shoulder_width
            is_r_hand_low = r_wrist[1] > r_shoulder[1]

            right_arm_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
            torso_height = (l_shoulder[1]+r_shoulder[1])/2-nose[1]
            if core_visible and torso_height > 35 and duck_state == "STAND":
                base_torso_height = (.95*base_torso_height+.05*torso_height) if base_torso_height else torso_height
            ref_height = max(base_torso_height, torso_height, 40.)
            points = {"left": l_wrist, "right": r_wrist}
            shoulders = {"left": l_shoulder, "right": r_shoulder}
            arm_angles = {"left": left_arm_angle, "right": right_arm_angle}
            extended = {}
            guard = {}
            for side in ("left", "right"):
                if not fresh[side]:
                    prev_hands.pop(side,None)
                    continue
                hand = points[side]
                angle = arm_angles[side]
                reach = calculate_distance(shoulders[side],hand)/ref_height
                velocity = calculate_distance(hand,prev_hands[side])/ref_height if side in prev_hands else 0.
                prev_hands[side] = hand.copy()
                extended[side] = angle > 125 or (angle > 110 and reach > 1.65) or (velocity > .55 and angle > 105 and reach > 1.45)
                guard[side] = calculate_distance(hand,nose)/ref_height <= 1.8 and hand[1] <= shoulders[side][1]
                hands[side] = hand
                angles[side] = angle
            valid_pose = any(fresh.values())
            if is_detected:
                stats.detected_frames += 1
                if not any(extended.values()):
                    stats.guard_samples += 1
                    stats.good_guard += int(all(guard.values()))
                stats.warnings += int(not all(guard.values()))
            for side in ("left", "right"):
                state = jab_state if side == "left" else cross_state
                other = "right" if side == "left" else "left"
                if not fresh[side]: state = "WAIT_GUARD"
                elif extended[side] and state == "GUARD":
                    state = "EXTENDED"
                    if side == "left": jab_count += 1
                    else: stats.cross_count += 1
                    if other in guard:
                        stats.evaluated_punches += 1
                        stats.good_punches += int(guard[other])
                        if side == "left": stats.good_jabs += int(guard[other])
                    feedback_msg = ("좋은 잽!" if side == "left" else "좋은 스트레이트!") if guard.get(other) else "반대 손 가드 확인"
                elif arm_angles[side] < 115 and calculate_distance(shoulders[side],points[side])/ref_height < 1.55:
                    state = "GUARD"
                if side == "left": jab_state = state
                else: cross_state = state
            if core_visible and base_torso_height > 0:
                ratio = torso_height/max(base_torso_height,40.)
                if ratio < .68 and duck_state == "STAND":
                    duck_state = "DUCKED"; stats.duck_count += 1
                elif ratio > .82: duck_state = "STAND"
            if hsv_enabled and valid_pose:
                gloves = detect_gloves(display_frame, hands['left'], hands['right'], shoulder_width)
                for side,color in [('left',(255,0,0)),('right',(0,0,255))]:
                    if hands[side] is not None and gloves[side] is not None:
                        if side == 'left': stats.hsv_left_frames += 1
                        else: stats.hsv_right_frames += 1
                        hands[side] = fuse_wrist(hands[side],gloves[side],shoulder_width)
                        cv2.circle(display_frame,gloves[side],9,color,2)
            feedback_color = (0,255,0) if valid_pose else (0,165,255)

            # ==========================================
            # [속도 반응형 주먹 궤적 잔상]
            # ==========================================
            for i in range(1, len(left_wrist_trail)):
                ratio = i / float(len(left_wrist_trail))
                speed_boost = min(1.0, left_wrist_trail[i][2] / 0.25)
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
                speed_boost = min(1.0, right_wrist_trail[i][2] / 0.25)
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

            # 관절 스켈레톤 라인 그리기
            cv2.line(
                display_frame, l_shoulder, l_elbow, (255, 255, 0), 4, cv2.LINE_AA
            )
            cv2.line(
                display_frame, l_elbow, l_wrist, (255, 255, 0), 4, cv2.LINE_AA
            )
            cv2.line(
                display_frame, r_shoulder, r_elbow, feedback_color, 4, cv2.LINE_AA
            )
            cv2.line(
                display_frame, r_elbow, r_wrist, feedback_color, 4, cv2.LINE_AA
            )

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
        if missing_frames > 15:
            prev_kpts = None

    # Game only uses freshly observed pose, never a held skeleton.
    if not valid_pose:
        jab_state = cross_state = "WAIT_GUARD"
        prev_hands.clear()
    game.update(display_frame.shape, hands, angles, shoulder_width, extended=extended)
    game.draw(display_frame)
    stats.jab_count = jab_count
    stats.sync_game(game)
    screen = ui.render(display_frame, stats, game, feedback_msg, valid_pose,
                       clahe_mode, clahe_active, hsv_enabled, current_mode, fps_ema, infer_ms)
    ui.present(screen)

    elapsed_ms = (time.perf_counter() - loop_start) * 1000
    wait_ms = max(1, int(33 - elapsed_ms))
    key = cv2.waitKey(wait_ms) & 0xFF
    try:
        if cv2.getWindowProperty(ui.window, cv2.WND_PROP_VISIBLE) < 1: break
    except cv2.error:
        break
    if key in (ord("?"), ord("/")):
        ui.help = not ui.help
    elif key == ord("q"):
        break
    elif key == ord("c"):
        clahe_mode = (clahe_mode + 1) % 3
    elif key == ord("g"):
        game.toggle()
    elif key == ord("h"):
        hsv_enabled = not hsv_enabled
    elif key == ord("p"):
        print("리포트 저장:", stats.save(os.path.join(SCRIPT_DIR, "reports")))
    elif key == ord("f"):
        filter_mode_idx = (filter_mode_idx + 1) % len(FILTER_MODES)
        prev_kpts = None
    elif key == ord("r"):
        jab_count = 0
        cross_state = "WAIT_GUARD"
        duck_state = "STAND"
        base_torso_height = 0.0
        prev_hands.clear()
        game = MittGame()
        stats = SessionStats(version="sim-test")
        jab_state = "WAIT_GUARD"
        left_wrist_trail.clear()
        right_wrist_trail.clear()
        feedback_msg = "기록 초기화 완료"
        feedback_color = (0, 255, 0)
    elif key == ord("s"):
        saved_filename = f"capture_{datetime.now().strftime('%H%M%S')}.png"
        save_path = os.path.join(SCRIPT_DIR, saved_filename)
        is_ok, buf = cv2.imencode(".png", screen)
        if is_ok:
            buf.tofile(save_path)
            saved_notice_until = time.perf_counter() + 1.5
            print(f"스크린샷 저장 완료: {save_path}")

cap.release()
paths = None
try:
    paths = stats.save(os.path.join(SCRIPT_DIR, "reports"))
    print("최종 리포트 저장:", paths)
    print(stats.to_dict())
except OSError as e:
    print("리포트 저장 오류:", e)
try:
    if cv2.getWindowProperty(ui.window, cv2.WND_PROP_VISIBLE) >= 1:
        ui.summary(stats, paths)
        while cv2.waitKey(50) < 0:
            if cv2.getWindowProperty(ui.window, cv2.WND_PROP_VISIBLE) < 1: break
except cv2.error:
    pass
finally:
    cv2.destroyAllWindows()