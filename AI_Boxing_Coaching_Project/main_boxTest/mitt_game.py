"""Virtual mitt: simple screen-space target, movement-gated, with cooldown.
This is a game hit, not an assessment of boxing technique or depth contact.
"""
import math
import time
from dataclasses import dataclass

import cv2


@dataclass
class MittConfig:
    radius: int = 44
    cooldown_s: float = 0.55
    min_speed_px_s: float = 110.0
    max_sample_gap_s: float = 0.25


class MittGame:
    def __init__(self, config=None):
        self.config = config or MittConfig()
        self.enabled = False
        self.hits = 0
        self.attempts = 0
        self.target_side = 'left'
        self.last_hit_at = float('-inf')
        self.previous = {'left': None, 'right': None}
        self.target = None

    def toggle(self):
        self.enabled = not self.enabled
        self.previous = {'left': None, 'right': None}

    def _position(self, shape):
        h, w = shape[:2]
        # Alternate target near center but on both sides, away from HUD.
        return (int(w * (0.40 if self.target_side == 'left' else 0.60)), int(h * 0.43))

    def update(self, shape, hands, timestamp=None):
        now = time.monotonic() if timestamp is None else timestamp
        self.target = self._position(shape)
        if not self.enabled:
            self.previous = {'left': None, 'right': None}
            return None
        hit = None
        for side in ('left', 'right'):
            point = hands.get(side)
            prev = self.previous[side]
            if point is not None:
                point = tuple(int(v) for v in point)
            if point is not None and prev is not None:
                previous_pt, previous_time = prev
                dt = now - previous_time
                if 0 < dt <= self.config.max_sample_gap_s:
                    speed = math.dist(point, previous_pt) / dt
                    was_outside = math.dist(previous_pt, self.target) > self.config.radius
                    inside = math.dist(point, self.target) <= self.config.radius
                    # Simple crossing check. Avoid counting an unmoving hand at the target.
                    if (side == self.target_side and was_outside and inside
                            and speed >= self.config.min_speed_px_s
                            and now - self.last_hit_at >= self.config.cooldown_s):
                        self.hits += 1
                        self.attempts += 1
                        hit = side
                        self.last_hit_at = now
                        self.target_side = 'right' if side == 'left' else 'left'
                        self.target = self._position(shape)
                        self.previous = {'left': None, 'right': None}
                        break
            self.previous[side] = (point, now) if point is not None else None
        return hit

    def draw(self, frame):
        if not self.enabled:
            return
        if self.target is None:
            self.target = self._position(frame.shape)
        color = (255, 170, 35) if self.target_side == 'left' else (40, 90, 255)
        cv2.circle(frame, self.target, self.config.radius, color, 4)
        cv2.circle(frame, self.target, 8, color, -1)
        cv2.putText(frame, f'MITT: {self.target_side.upper()}  HITS: {self.hits}',
                    (20, frame.shape[0]-23), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
