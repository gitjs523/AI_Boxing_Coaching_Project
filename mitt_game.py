"""Timed 2-D mitt game. Temporary test: fresh moving hand entering a target; posture scored separately."""
import math
import random
import time
import cv2

class MittGame:
    def __init__(self):
        self.enabled = False
        self.hits = self.misses = self.score = self.combo = self.max_combo = 0
        self.target_side = 'left'
        self.target = None
        self.deadline = 0.0
        self.previous = {'left': None, 'right': None}
        self.armed = {'left': False, 'right': False}
        self.message = ''
        self.message_until = 0.0
        self.radius = 40
        self.last_update = None
        self.round_until = 0.0
        self.remaining = 60.0
        self.finished = False
        self.recent_punch = dict.fromkeys(("left", "right"), -1e9)
        self.marker = None
        self.contact_radius = None
        self.status = "WAIT"

    @property
    def attempts(self):
        return self.hits + self.misses

    @property
    def accuracy(self):
        return 100*self.hits/self.attempts if self.attempts else 0.0

    def toggle(self):
        if self.enabled:
            self.enabled = False
            self.target = None
        else:
            self.__init__()
            self.enabled = True
            self.round_until = time.monotonic() + 60
        self.previous = dict.fromkeys(('left','right'))

    def _spawn(self, shape, now):
        h,w = shape[:2]
        self.radius = max(18, int(min(h,w)*.115))
        lo,hi = (.25,.46) if self.target_side=='left' else (.54,.75)
        self.target = (int(w*random.uniform(lo,hi)), int(h*random.uniform(.30,.52)))
        self.deadline = min(now+5.0,self.round_until)
        self.previous = dict.fromkeys(('left','right'))
        self.armed = dict.fromkeys(('left','right'),False)
        self.recent_punch = dict.fromkeys(("left", "right"), -1e9)

    def update(self, shape, hands, angles=None, shoulder_width=0, timestamp=None, extended=None):
        now = time.monotonic() if timestamp is None else timestamp
        if not self.enabled:
            return
        self.remaining = max(0,self.round_until-now)
        if now >= self.round_until:
            if self.target is not None: self.misses += 1
            self.enabled = False
            self.finished = True
            self.target = None
            return
        if self.target is None: self._spawn(shape,now)
        if now >= self.deadline:
            self.misses += 1
            self.combo = 0
            self.message, self.message_until = '시간 초과', now+.7
            self.target_side = 'right' if self.target_side=='left' else 'left'
            self._spawn(shape,now)
            return
        side = self.target_side
        p = hands.get(side)
        angle = (angles or {}).get(side)
        self.marker = None
        self.contact_radius = None
        self.status = "WAIT"
        if p is None or angle is None or shoulder_width <= 20:
            self.previous[side] = None
            return
        p = tuple(map(float,p))
        prev = self.previous[side]
        self.marker = tuple(map(int,p))
        self.status = "TRACK"
        if angle <= 120: self.armed[side] = True
        if (extended or {}).get(side, False) or angle >= 125:
            self.recent_punch[side] = now
        if prev is not None:
            old,t = prev
            dt=now-t
            if 0 < dt <= .45:
                speed=math.dist(old,p)/dt/shoulder_width
                dx, dy = p[0]-old[0], p[1]-old[1]
                length2 = dx*dx + dy*dy
                u = max(0., min(1., ((self.target[0]-old[0])*dx+(self.target[1]-old[1])*dy)/length2)) if length2 else 0.
                closest = (old[0]+u*dx, old[1]+u*dy)
                contact_radius = self.radius + max(8, min(35, shoulder_width * .16))
                contact = math.dist(closest,self.target) <= contact_radius
                self.contact_radius = int(contact_radius)
                outside = math.dist(old, self.target) > contact_radius
                punch = now-self.recent_punch[side] <= .35 or (self.armed[side] and angle >= 105)
                moving = speed >= .18 and math.dist(old,p) >= 3
                self.status = "READY" if moving else "MOVE"
                if outside and contact and moving:
                    self.hits += 1
                    self.combo += 1
                    self.max_combo=max(self.max_combo,self.combo)
                    self.score += 10
                    self.message,self.message_until='명중! +10',now+.7
                    self.target_side='right' if side=='left' else 'left'
                    self._spawn(shape,now)
                    return side
        self.previous[side]=(p,now)

    def draw(self, frame):
        if not self.enabled or self.target is None: return
        color=(245,165,55) if self.target_side=='left' else (55,65,245)
        x,y=self.target; r=self.radius
        cv2.circle(frame,(x,y),r,(15,15,20),7,cv2.LINE_AA)
        cv2.circle(frame,(x,y),r,color,3,cv2.LINE_AA)
        cv2.circle(frame,(x,y),max(5,r//2),color,2,cv2.LINE_AA)
        cv2.circle(frame,(x,y),4,color,-1,cv2.LINE_AA)
        if self.marker is not None:
            cv2.circle(frame,self.marker,12,(0,255,255),2,cv2.LINE_AA)
            if self.contact_radius is not None:
                cv2.circle(frame,(x,y),self.contact_radius,(100,160,160),1,cv2.LINE_AA)
            cv2.line(frame,self.marker,(x,y),(100,160,160),1,cv2.LINE_AA)
        cv2.putText(frame, self.status, (max(0,x-r),max(20,y-r-12)), cv2.FONT_HERSHEY_SIMPLEX,.5,color,2)
        frac=max(0,min(1,(self.deadline-time.monotonic())/5))
        cv2.ellipse(frame,(x,y),(r+7,r+7),-90,0,360*frac,color,3,cv2.LINE_AA)
