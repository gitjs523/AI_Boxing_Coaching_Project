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
# [신뢰도 가중 적응형 EMA 필터 & 속도 반응형 궤적]
# ==========================================
prev_kpts = None
vel_ema = np.zeros(17, dtype=np.float32)
last_valid_kpts = None
missing_frames = 0
MAX_MISSING_FRAMES = 5  # 순간적인 인식 누락 시 최대 5프레임(~0.16초) 스켈레톤 유지

ema_enabled = True  # 'f' 키로 필터 ON/OFF 전환
ALPHA_MIN = 0.12    # 정지/가드 시 강한 노이즈 억제
ALPHA_MAX = 0.75    # 연속된 빠른 펀치 시 지연 없는 추적

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
        continue

    # MoveNet 입력 변환: 192x192 RGB
    img = cv2.resize(frame, (192, 192))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if "int32" in input_type:
        input_tensor = np.expand_dims(img.astype(np.int32), axis=0)
    else:
        input_tensor = np.expand_dims(img.astype(np.float32), axis=0)

    infer_start = time.perf_counter()
    outputs = session.run(None, {input_name: input_tensor})[0]
    infer_ms = (time.perf_counter() - infer_start) * 1000.0
    raw_kpts = outputs[0, 0]  # (17, 3) -> [y, x, score]

    # ==========================================
    # [기능 1] 신뢰도 가중 1€ 스타일 적응형 EMA 필터
    # ==========================================
    kpts = raw_kpts.copy()
    if ema_enabled and prev_kpts is not None:
        raw_pos = raw_kpts[:, :2]
        conf = raw_kpts[:, 2]
        dist = np.linalg.norm(raw_pos - prev_kpts[:, :2], axis=1)

        # 1) 1프레임 만에 화면 12% 이상 순간이동하는 오인식 스파이크 감지
        is_spike = dist > 0.12

        # 2) 이동 속도 자체를 먼저 저역통과 필터링 (단발성 떨림에 alpha 상승 방지)
        safe_dist = np.where(is_spike, 0.0, dist)
        vel_ema = 0.4 * safe_dist + 0.6 * vel_ema

        # 3) 평활화된 실제 이동 속도에 따라 alpha 산출
        dynamic_alpha = np.clip(
            ALPHA_MIN + (vel_ema - 0.004) * ((ALPHA_MAX - ALPHA_MIN) / 0.035),
            ALPHA_MIN,
            ALPHA_MAX,
        )

        # 4) 관절 신뢰도(conf) 가중치 적용 및 스파이크 억제
        conf_weight = np.clip((conf - 0.20) / 0.45, 0.05, 1.0)
        effective_alpha = dynamic_alpha * conf_weight
        effective_alpha = np.where(is_spike, 0.05, effective_alpha)

        kpts[:, :2] = (
            effective_alpha[:, None] * raw_pos
            + (1.0 - effective_alpha[:, None]) * prev_kpts[:, :2]
        )
    prev_kpts = kpts.copy()

    display_frame = cv2.flip(frame, 1)
    h, w, _ = display_frame.shape

    required_indices = [0, 5, 6, 7, 8, 9, 10]
    is_detected = all(kpts[i][2] > 0.25 for i in required_indices)

    if is_detected:
        missing_frames = 0
        last_valid_kpts = kpts.copy()
        active_kpts = kpts
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

        shoulder_width = calculate_distance(l_shoulder, r_shoulder)
        norm_base = max(shoulder_width, 1.0)

        # 실제 새 프레임이 감지되었을 때만 궤적 및 카운트 갱신
        if is_detected:
            l_norm_speed = 0.0
            if left_wrist_trail:
                dist_l = calculate_distance(
                    (left_wrist_trail[-1][0], left_wrist_trail[-1][1]), l_wrist
                )
                if dist_l > w * 0.25:
                    left_wrist_trail.clear()
                else:
                    l_norm_speed = dist_l / norm_base

            r_norm_speed = 0.0
            if right_wrist_trail:
                dist_r = calculate_distance(
                    (right_wrist_trail[-1][0], right_wrist_trail[-1][1]), r_wrist
                )
                if dist_r > w * 0.25:
                    right_wrist_trail.clear()
                else:
                    r_norm_speed = dist_r / norm_base

            left_wrist_trail.append((l_wrist[0], l_wrist[1], l_norm_speed))
            right_wrist_trail.append((r_wrist[0], r_wrist[1], r_norm_speed))

        if shoulder_width > 20:
            left_arm_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
            r_guard_dist = calculate_distance(r_wrist, nose) / shoulder_width
            is_r_hand_low = r_wrist[1] > r_shoulder[1]

            if is_detected:
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
            # [기능 2] 속도 반응형 주먹 궤적 잔상
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
            vel_ema.fill(0.0)

    # ==========================================
    # [반응형 상단 UI 헤더]
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
    # [하단 좌측: 적응형 EMA 배지 / 우측: FPS & 추론 시간 HUD]
    # ==========================================
    f_scale = 0.46 * scale_base
    f_thick = max(1, int(1.5 * scale_base))
    pad_x, pad_y = int(10 * scale_base), int(6 * scale_base)

    if time.perf_counter() < saved_notice_until:
        filter_label = f"Saved: {saved_filename}"
        filter_color = (0, 255, 120)
    else:
        filter_label = (
            f"Adaptive EMA: {'ON' if ema_enabled else 'OFF'} ('f') | Shot ('s')"
        )
        filter_color = (0, 255, 200) if ema_enabled else (180, 180, 180)

    (fw, fh), _ = cv2.getTextSize(
        filter_label, cv2.FONT_HERSHEY_SIMPLEX, f_scale, f_thick
    )

    l_box_x1 = int(12 * scale_base)
    l_box_y2 = h - int(10 * scale_base)
    l_box_y1 = l_box_y2 - fh - pad_y * 2
    l_box_x2 = l_box_x1 + fw + pad_x * 2

    cv2.rectangle(
        display_frame, (l_box_x1, l_box_y1), (l_box_x2, l_box_y2), (20, 20, 20), -1
    )
    cv2.putText(
        display_frame,
        filter_label,
        (l_box_x1 + pad_x, l_box_y2 - pad_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        f_scale,
        filter_color,
        f_thick,
        cv2.LINE_AA,
    )

    perf_label = f"FPS: {fps_ema:.1f} | Infer: {infer_ms:.1f}ms"
    (pw, ph), _ = cv2.getTextSize(
        perf_label, cv2.FONT_HERSHEY_SIMPLEX, f_scale, f_thick
    )

    r_box_x2 = w - int(12 * scale_base)
    r_box_x1 = r_box_x2 - pw - pad_x * 2
    r_box_y2 = l_box_y2
    r_box_y1 = r_box_y2 - ph - pad_y * 2

    if r_box_x1 > l_box_x2 + int(8 * scale_base):
        cv2.rectangle(
            display_frame,
            (r_box_x1, r_box_y1),
            (r_box_x2, r_box_y2),
            (20, 20, 20),
            -1,
        )
        cv2.putText(
            display_frame,
            perf_label,
            (r_box_x1 + pad_x, r_box_y2 - pad_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            f_scale,
            (220, 220, 220),
            f_thick,
            cv2.LINE_AA,
        )

    cv2.imshow(
        "AI Boxing Coach - Adaptive EMA & Motion Trail (Press 'q' to quit)",
        display_frame,
    )

    elapsed_ms = (time.perf_counter() - loop_start) * 1000
    wait_ms = max(1, int(33 - elapsed_ms))
    key = cv2.waitKey(wait_ms) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("f"):
        ema_enabled = not ema_enabled
        prev_kpts = None
        vel_ema.fill(0.0)
    elif key == ord("s"):
        saved_filename = f"capture_{datetime.now().strftime('%H%M%S')}.png"
        save_path = os.path.join(SCRIPT_DIR, saved_filename)
        is_ok, buf = cv2.imencode(".png", display_frame)
        if is_ok:
            buf.tofile(save_path)
            saved_notice_until = time.perf_counter() + 1.5
            print(f"스크린샷 저장 완료: {save_path}")

cap.release()
cv2.destroyAllWindows()