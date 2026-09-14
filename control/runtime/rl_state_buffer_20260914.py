from collections import deque
class RLStateBuffer:
 def __init__(self,maxlen=1000): self.x=deque(maxlen=int(maxlen))
 def append(self,t_ns,state): self.x.append((int(t_ns),state)); return state
 put=append
 def latest(self): return self.x[-1][1] if self.x else None
 def latest_before(self,t):
  for n,s in reversed(self.x):
   if n<=int(t): return s
  return None
 get_latest_before=latest_before; sample=latest_before
 def __len__(self): return len(self.x)
