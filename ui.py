"""Korean broadcast-style HUD. Pillow glyph patches are cached to limit CPU cost."""
import os
from functools import lru_cache
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W,H=1440,810
BG=(17,16,19); PANEL=(29,27,32); LINE=(55,51,59); WHITE=(245,244,248); MUTED=(165,160,174); RED=(61,49,241); GREEN=(117,221,82); GOLD=(78,207,255)

def font_path(bold=False):
    win=os.environ.get('WINDIR',r'C:\Windows')
    paths=[os.path.join(win,'Fonts','malgunbd.ttf' if bold else 'malgun.ttf'),os.path.join(win,'Fonts','malgun.ttf'),'/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf' if bold else '/usr/share/fonts/truetype/nanum/NanumGothic.ttf','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    for path in paths:
        if os.path.isfile(path): return path
    raise RuntimeError('한국어 표시용 글꼴을 찾을 수 없습니다. 맑은 고딕을 설치하세요.')

@lru_cache(maxsize=32)
def font(size,bold): return ImageFont.truetype(font_path(bold),size)

@lru_cache(maxsize=384)
def glyph(text,size,bold):
    f=font(size,bold); left,top,right,bottom=f.getbbox(text)
    im=Image.new('L',(max(1,right-left+3),max(1,bottom-top+3)))
    ImageDraw.Draw(im).text((1-left,1-top),text,font=f,fill=255)
    return np.asarray(im)

def txt(img,text,x,y,size=24,color=WHITE,bold=False):
    a=glyph(str(text),size,bold); hh,ww=a.shape
    if x<0 or y<0 or x>=img.shape[1] or y>=img.shape[0]: return
    hh=min(hh,img.shape[0]-y); ww=min(ww,img.shape[1]-x); a=a[:hh,:ww,None].astype(np.uint16)
    region=img[y:y+hh,x:x+ww]; region[:]=((region.astype(np.uint16)*(255-a)+np.asarray(color,dtype=np.uint16)*a)//255).astype(np.uint8)

def rect(img,x,y,w,h,color=PANEL): cv2.rectangle(img,(x,y),(x+w-1,y+h-1),color,-1)

def fit_window():
    try:
        import tkinter as tk
        root=tk.Tk();root.withdraw(); sw,sh=root.winfo_screenwidth(),root.winfo_screenheight();root.destroy()
        scale=min(1,(sw-50)/W,(sh-95)/H)
        return max(1,int(W*scale)),max(1,int(H*scale))
    except Exception: return 1280,720

class CoachUI:
    def __init__(self):
        self.window='AI BOXING COACH';self.size=fit_window();self.help=False;self.action=None
        cv2.namedWindow(self.window,cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window,self.mouse)
    def mouse(self,event,x,y,flags,param):
        if event!=cv2.EVENT_LBUTTONUP:return
        x=x*W/self.size[0];y=y*H/self.size[1]
        if 1210<=x<=1410 and 20<=y<=65:self.help=not self.help
        elif self.help and 1020<=x<=1130 and 140<=y<=190:self.help=False
    def present(self,img):
        cv2.imshow(self.window,cv2.resize(img,self.size,interpolation=cv2.INTER_AREA))
    def render(self,frame,stats,game,feedback,detected,clahe_mode,clahe_active,hsv,filter_mode,fps,infer):
        img=np.full((H,W,3),BG,np.uint8)
        rect(img,0,0,W,5,RED);rect(img,24,20,7,44,RED)
        txt(img,'AI BOXING',45,17,35,WHITE,True);txt(img,'COACH  /  복싱 트레이닝',265,28,21,MUTED,True)
        txt(img,'실시간 관절 추적' if detected else '카메라 정면에 서 주세요',800,29,21,GREEN if detected else GOLD,True)
        rect(img,1210,20,200,45,RED);txt(img,'?  조작법',1248,30,23,WHITE,True)
        # Native camera frame stays undistorted; game coordinates remain in camera pixels.
        x,y,vw,vh=24,85,994,600
        rect(img,x,y,vw,vh,(8,8,10))
        scale=min(vw/frame.shape[1],vh/frame.shape[0]); fw,fh=max(1,int(frame.shape[1]*scale)),max(1,int(frame.shape[0]*scale)); fx=x+(vw-fw)//2;fy=y+(vh-fh)//2
        img[fy:fy+fh,fx:fx+fw]=cv2.resize(frame,(fw,fh))
        cv2.rectangle(img,(x,y),(x+vw,y+vh),LINE,1)
        rect(img,x+12,y+12,225,36,(20,18,22));txt(img,'미트 훈련' if game.enabled else '자세 분석 모드',x+23,y+20,22,WHITE,True)
        if game.enabled:
            side='왼손' if game.target_side=='left' else '오른손'
            txt(img,side+'으로 타격',x+18,y+vh-39,26,GOLD,True)
        if game.message and __import__('time').monotonic()<game.message_until: txt(img,game.message,x+390,y+20,34,GREEN if '명중' in game.message else GOLD,True)
        # Coach panel
        px=1034;pw=382
        rect(img,px,85,pw,125);rect(img,px,85,5,125,RED)
        txt(img,'훈련 피드백',px+18,98,19,MUTED,True)
        txt(img,feedback if detected else '관절 인식 대기 중',px+18,135,26,WHITE,True)
        txt(img,'잽 횟수   '+str(stats.jab_count),px+18,176,21,GOLD,True)
        rect(img,px,222,pw,212);txt(img,'가상 미트  /  60초 라운드',px+18,235,21,MUTED,True)
        txt(img,f'{game.score:,}',px+18,273,53,WHITE,True);txt(img,'점',px+220,298,23,MUTED)
        txt(img,f'콤보 {game.combo}   최고 {game.max_combo}',px+18,336,24,GOLD,True)
        txt(img,f'명중 {game.hits}  실패 {game.misses}  {game.accuracy:.0f}%',px+18,372,22,WHITE)
        txt(img,('남은 시간 '+str(int(np.ceil(game.remaining)))+'초') if game.enabled else ('라운드 종료' if game.finished else 'G 키로 훈련 시작'),px+18,403,18,MUTED)
        rect(img,px,446,pw,126);txt(img,'자세 평가  /  규칙 기반',px+18,459,19,MUTED,True)
        score,grade=stats.evaluate();txt(img,grade,px+18,492,43,GREEN if score is not None else MUTED,True)
        txt(img,f'{score:.1f} / 100' if score is not None else '잽 5회 이상 필요',px+160,509,21,WHITE)
        guard=100*stats.good_guard/stats.guard_samples if stats.guard_samples else 0
        txt(img,f'가드 유지율 {guard:.0f}%' if stats.guard_samples else '가드 평가 대기',px+18,547,18,MUTED)
        rect(img,px,584,pw,101);txt(img,'시스템 성능',px+18,597,18,MUTED,True)
        txt(img,f'{fps:.1f} FPS',px+18,630,30,GREEN,True);txt(img,f'추론 {infer:.1f} ms',px+185,640,22,WHITE)
        rect(img,24,700,1392,86)
        modes=['꺼짐','켜짐','자동']; filt={'ADAPTIVE':'적응형','FIXED':'고정형','OFF':'꺼짐'}[filter_mode]
        txt(img,f'명암 보정  {modes[clahe_mode]}'+(' / 적용 중' if clahe_active else '')+f'     |     글러브 추적  {"켜짐" if hsv else "꺼짐"}     |     EMA  {filt}',42,713,22,WHITE,True)
        txt(img,'G 미트 게임    C 명암 보정    P 리포트    R 초기화    Q 종료',42,752,20,MUTED)
        if self.help:
            img=(img.astype(np.float32)*.30).astype(np.uint8);rect(img,280,128,880,554,PANEL);rect(img,280,128,880,6,RED)
            txt(img,'조작법  /  훈련 가이드',320,155,34,WHITE,True);rect(img,1020,140,110,50,RED);txt(img,'닫기',1050,154,23,WHITE,True)
            entries=[('G','60초 미트 게임 시작 / 중지'),('C','CLAHE 명암 보정: 꺼짐 → 켜짐 → 자동'),('H','HSV 색상 글러브 추적 켜기 / 끄기'),('F','EMA 필터 모드 변경'),('P','운동 통계 리포트 저장'),('R','현재 세션과 게임 기록 초기화'),('S','스크린샷 저장'),('Q','운동 종료 및 최종 리포트 저장'),('? / /','조작법 열기 / 닫기')]
            for i,(key,label) in enumerate(entries):
                yy=218+i*45;rect(img,320,yy,110,35,(55,51,59));txt(img,key,344,yy+6,21,WHITE,True);txt(img,label,455,yy+6,23,WHITE)
            txt(img,'미트: 팔을 굽혔다가 뻗어 지정된 손으로 과녁을 맞추세요.',320,645,20,GOLD)
        return img
    def summary(self,stats,paths):
        img=np.full((H,W,3),BG,np.uint8);rect(img,0,0,W,7,RED);txt(img,'훈련 종료  /  최종 리포트',48,45,44,WHITE,True)
        d=stats.to_dict(); score,grade=stats.evaluate();rect(img,48,125,400,400,PANEL);txt(img,'자세 평가 등급',80,150,27,MUTED,True);txt(img,grade,80,225,85,GREEN if score is not None else GOLD,True);txt(img,f'{score:.1f} / 100' if score is not None else '평가 표본 부족',80,350,30,WHITE,True);txt(img,'잽 자세 70% + 가드 유지 30%',80,435,21,MUTED)
        for i,(label,value) in enumerate([('잽 횟수',d['jab_count']),('가드 유지율',d['guard_rate_percent']),('미트 점수',d['mitt_score']),('미트 명중률',d['mitt_accuracy_percent']),('최고 콤보',d['max_combo']),('관절 인식률',d['pose_detection_rate_percent'])]):
            x=470+(i%2)*465;y=125+(i//2)*135;rect(img,x,y,445,120,PANEL);txt(img,label,x+20,y+15,23,MUTED,True);txt(img,('—' if value is None else str(value)+('%' if '율' in label else '')),x+20,y+54,38,WHITE,True)
        txt(img,'CSV / JSON 저장 완료' if paths else '리포트 저장 실패 — 콘솔 확인',48,565,28,GREEN if paths else RED,True)
        txt(img,'참고: 카메라 2D 관절 규칙에 기반한 평가이며 실제 복싱 기술 검증 점수는 아닙니다.',48,625,23,MUTED)
        txt(img,'아무 키나 누르면 종료합니다.',48,690,25,WHITE,True);self.present(img)
