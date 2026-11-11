"""Session statistics, CSV/JSON export. Grades deliberately unspecified."""
import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import time


@dataclass
class SessionStats:
    version: str
    started_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec='seconds'))
    start_monotonic: float = field(default_factory=time.monotonic)
    frames: int = 0
    detected_frames: int = 0
    jab_count: int = 0
    mitt_hits: int = 0
    warnings: int = 0
    hsv_left_frames: int = 0
    hsv_right_frames: int = 0
    grade: str = 'UNDEFINED'

    def to_dict(self):
        elapsed = round(time.monotonic() - self.start_monotonic, 2)
        return {
            'version': self.version,
            'started_at': self.started_at,
            'duration_seconds': elapsed,
            'frames': self.frames,
            'pose_detection_rate_percent': round(100 * self.detected_frames / max(1,self.frames),1),
            'jab_count': self.jab_count,
            'mitt_hits': self.mitt_hits,
            'guard_warning_frames': self.warnings,
            'hsv_left_detected_frames': self.hsv_left_frames,
            'hsv_right_detected_frames': self.hsv_right_frames,
            'accuracy_grade': self.grade,
            'grade_note': 'S~C grade criteria will be defined after implementation/testing.',
        }

    def save(self, output_dir):
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        data = self.to_dict()
        json_path = folder / f'session_{stamp}.json'
        csv_path = folder / f'session_{stamp}.csv'
        with json_path.open('w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        with csv_path.open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(data))
            writer.writeheader()
            writer.writerow(data)
        return json_path, csv_path
