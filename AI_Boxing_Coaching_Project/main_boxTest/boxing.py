"""AI Boxing Coach - MoveNet Original + Team3.
Preserves original MoveNet elbow-angle jab rule. Adds team's CLAHE,
virtual mitt game, and session statistics. S~C grade is not defined yet.
"""
import math
import os
import urllib.request

import cv2
import numpy as np
import onnxruntime as ort

from preprocessing import FramePreprocessor
from mitt_game import MittGame
from report import SessionStats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'movenet_singlepose_lightning_4.onnx')
MODEL_URL = 'https://huggingface.co/Xenova/movenet-singlepose-lightning/resolve/main/onnx/model.onnx'


def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    print('MoveNet ONNX model downloading...')
    request = urllib.request.Request(MODEL_URL, headers={'User-Agent': 'Mozilla/5.0'})
    tmp_path = MODEL_PATH + '.part'
    try:
        with urllib.request.urlopen(request, timeout=60) as response, open(tmp_path, 'wb') as file:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                file.write(chunk)
        os.replace(tmp_path, MODEL_PATH)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def calculate_angle(a, b, c):
    a, b, c = np.asarray(a), np.asarray(b), np.asarray(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = abs(rad * 180.0 / np.pi)
    return 360.0-angle if angle > 180.0 else angle


def distance(p1, p2):
    return math.hypot(p2[0]-p1[0], p2[1]-p1[1])


class BoxingCoach:
    def __init__(self):
        ensure_model()
        self.session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
        info = self.session.get_inputs()[0]
        self.input_name = info.name
        self.input_type = info.type
        self.preprocessor = FramePreprocessor()
        self.game = MittGame()
        self.stats = SessionStats(version='movenet_original')
        self.jab_state = 'GUARD'
        self.feedback = 'Ready! Keep your guard up.'
        self.feedback_color = (0, 255, 0)

    def pose(self, frame):
        image = cv2.cvtColor(cv2.resize(frame, (192,192)), cv2.COLOR_BGR2RGB)
        dtype = np.int32 if 'int32' in self.input_type else np.float32
        tensor = np.expand_dims(image.astype(dtype), axis=0)
        output = self.session.run(None, {self.input_name: tensor})[0]
        if output.shape[-2:] != (17, 3):
            raise RuntimeError(f'Unexpected MoveNet output shape: {output.shape}')
        return output.reshape(-1, 17, 3)[0]

    @staticmethod
    def point(kpts, idx, w, h, threshold=0.25):
        y, x, conf = kpts[idx]
        if conf < threshold:
            return None
        return (int((1-x)*w), int(y*h))

    def process(self, raw_frame):
        # Preprocess original orientation BEFORE pose inference; flip once for display.
        processed = self.preprocessor.apply(raw_frame)
        kpts = self.pose(processed)
        frame = cv2.flip(processed, 1)
        h, w = frame.shape[:2]
        pts = {name: self.point(kpts, idx, w, h) for name, idx in [
            ('nose',0), ('left_shoulder',5), ('right_shoulder',6),
            ('left_elbow',7), ('right_elbow',8),
            ('left_wrist',9), ('right_wrist',10)]}
        self.stats.frames += 1
        ready = all(v is not None for v in pts.values())
        shoulders = (distance(pts['left_shoulder'], pts['right_shoulder'])
                     if pts['left_shoulder'] and pts['right_shoulder'] else None)
        # Original MoveNet only: no HSV detection.
        game_hands = {'left': pts['left_wrist'], 'right': pts['right_wrist']}
        # Uses wrist coordinates only for game.
        if ready and shoulders and shoulders > 20:
            self.stats.detected_frames += 1
            angle = calculate_angle(pts['left_shoulder'], pts['left_elbow'], pts['left_wrist'])
            right_hand = pts['right_wrist']
            right_guard_dist = distance(right_hand, pts['nose'])/shoulders
            right_low = right_hand[1] > pts['right_shoulder'][1]
            if angle > 155:
                if self.jab_state == 'GUARD':
                    self.stats.jab_count += 1
                    self.jab_state = 'EXTENDED'
                if right_guard_dist > 1.4 or right_low:
                    self.feedback, self.feedback_color = 'WARNING: RIGHT hand down!', (0,0,255)
                else:
                    self.feedback, self.feedback_color = 'GOOD JAB!', (0,255,0)
            elif angle < 95:
                self.jab_state = 'GUARD'
                if pts['left_wrist'][1] > pts['left_shoulder'][1] or right_low:
                    self.feedback, self.feedback_color = 'Hands up!', (0,165,255)
                else:
                    self.feedback, self.feedback_color = 'Good Stance.', (0,255,0)
            if right_guard_dist > 1.4 or right_low:
                self.stats.warnings += 1  # counts frames, NOT individual errors
            for a,b,color in [
                ('left_shoulder','left_elbow',(255,255,0)),
                ('left_elbow','left_wrist',(255,255,0)),
                ('right_shoulder','right_elbow',self.feedback_color),
                ('right_elbow','right_wrist',self.feedback_color)]:
                cv2.line(frame, pts[a], pts[b], color, 3)
            for point in pts.values():
                cv2.circle(frame, point, 6, (255,255,255), -1)
            cv2.putText(frame, f'{int(angle)} deg', pts['left_elbow'],
                        cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 2)
        else:
            self.feedback, self.feedback_color = 'Pose not fully detected', (0,165,255)
            self.jab_state = 'GUARD'  # avoid stale extended state after tracking loss

        self.game.update(frame.shape, game_hands)
        self.game.draw(frame)
        self.stats.mitt_hits = self.game.hits
        cv2.rectangle(frame, (0,0), (w,110), (20,20,20), -1)
        cv2.putText(frame, f'JABS: {self.stats.jab_count}  MITT: {self.game.hits}',
                    (18,35), cv2.FONT_HERSHEY_SIMPLEX, .85, (255,255,255), 2)
        cv2.putText(frame, self.feedback, (18,72),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, self.feedback_color, 2)
        cv2.putText(frame, f'CLAHE: {"ON" if self.preprocessor.enabled else "OFF"}  GAME: {"ON" if self.game.enabled else "OFF"}  [C] [G] [R] [Q]',
                    (18,100), cv2.FONT_HERSHEY_SIMPLEX, .56, (230,230,230), 1)
        
        return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            raise RuntimeError('Webcam unavailable; check device permissions.')
        print('C: CLAHE, G: game, R: save report, Q: quit + save report')
        try:
            while True:
                ok, raw = cap.read()
                if not ok:
                    break
                rendered = self.process(raw)
                cv2.imshow('AI Boxing - MoveNet Original + Team3', rendered)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    self.preprocessor.enabled = not self.preprocessor.enabled
                elif key == ord('g'):
                    self.game.toggle()
                elif key == ord('r'):
                    self.save_report()
                elif key == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.save_report()

    def save_report(self):
        json_path, csv_path = self.stats.save(os.path.join(SCRIPT_DIR, 'reports'))
        print(f'Saved report: {json_path} and {csv_path}')
        print(f'Stats: {self.stats.to_dict()}')


if __name__ == '__main__':
    BoxingCoach().run()
