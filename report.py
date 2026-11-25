"""Session report: rule-based posture score, separate from game accuracy."""
import csv,json,time
from dataclasses import dataclass,field
from datetime import datetime
from pathlib import Path

@dataclass
class SessionStats:
    version:str='sim-v2'
    started_at:str=field(default_factory=lambda:datetime.now().isoformat(timespec='seconds'))
    start_monotonic:float=field(default_factory=time.monotonic)
    frames:int=0
    detected_frames:int=0
    jab_count:int=0
    good_jabs:int=0
    cross_count:int=0
    duck_count:int=0
    evaluated_punches:int=0
    good_punches:int=0
    guard_samples:int=0
    good_guard:int=0
    warnings:int=0
    hsv_left_frames:int=0
    hsv_right_frames:int=0
    mitt_hits:int=0
    mitt_misses:int=0
    mitt_score:int=0
    max_combo:int=0
    infer_total:float=0.0
    clahe_frames:int=0
    def sync_game(self,game):
        self.mitt_hits=game.hits; self.mitt_misses=game.misses
        self.mitt_score=game.score; self.max_combo=game.max_combo
    def evaluate(self):
        if self.evaluated_punches<5 or self.guard_samples<30: return None,'미평가'
        score=round(70*self.good_punches/self.evaluated_punches+30*self.good_guard/self.guard_samples,1)
        return score, 'S' if score>=90 else 'A' if score>=80 else 'B' if score>=70 else 'C'
    def to_dict(self):
        score,grade=self.evaluate()
        elapsed=max(.001,time.monotonic()-self.start_monotonic)
        attempts=self.mitt_hits+self.mitt_misses
        return dict(version=self.version,started_at=self.started_at,duration_seconds=round(elapsed,2),frames=self.frames,pose_detection_rate_percent=round(100*self.detected_frames/max(1,self.frames),1),jab_count=self.jab_count,cross_count=self.cross_count,duck_count=self.duck_count,evaluated_punches=self.evaluated_punches,good_punches=self.good_punches,good_jabs=self.good_jabs,jab_form_rate_percent=round(100*self.good_jabs/self.jab_count,1) if self.jab_count else None,guard_samples=self.guard_samples,guard_rate_percent=round(100*self.good_guard/self.guard_samples,1) if self.guard_samples else None,guard_warning_frames=self.warnings,posture_score=score,accuracy_grade=grade,mitt_hits=self.mitt_hits,mitt_misses=self.mitt_misses,mitt_score=self.mitt_score,mitt_accuracy_percent=round(100*self.mitt_hits/attempts,1) if attempts else None,max_combo=self.max_combo,average_fps=round(self.frames/elapsed,1),average_infer_ms=round(self.infer_total/max(1,self.frames),2),clahe_frames=self.clahe_frames,hsv_left_frames=self.hsv_left_frames,hsv_right_frames=self.hsv_right_frames,grade_note='2D 규칙 기반 참고 평가: 잽·스트레이트 반대손 가드 70%, 가드 유지 30%. 최소 평가 가능 펀치 5회 및 가드 30프레임. 최신 미트 라운드 통계.')
    def save(self,output_dir):
        folder=Path(output_dir); folder.mkdir(parents=True,exist_ok=True)
        stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        j=folder/f'session_{stamp}.json'; c=folder/f'session_{stamp}.csv'; data=self.to_dict()
        j.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
        with c.open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data));writer.writeheader();writer.writerow(data)
        return j,c
