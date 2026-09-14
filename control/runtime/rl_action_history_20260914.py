from collections import deque
import numpy as np
class ActionHistory:
 def __init__(self,length=30,kind='tail'): self.length=int(length); self.kind=kind; self.width=3 if kind=='tail' else 5; self.x=deque(maxlen=self.length)
 def append(self,a=None,completed=True,pwm_success=True,valid=None,**kw):
  if valid is None: valid=completed and pwm_success
  if not valid:return False
  a=a or kw; g=lambda k:getattr(a,k,0) if not isinstance(a,dict) else a.get(k,0); self.x.append(tuple(float(g(k)) for k in (['theta','t'] if self.kind=='tail' else ['theta','t','b1','b2']))); return True
 add=append; record=append
 def as_array(self,flatten=False):
  z=np.zeros((self.length,self.width+1),np.float32); n=len(self.x)
  for i,v in enumerate(self.x): z[self.length-n+i,:self.width]=v; z[self.length-n+i,self.width]=1
  return z.reshape(-1) if flatten else z
 vector=lambda s:s.as_array(True)
 def __len__(self): return len(self.x)
class TailActionHistory(ActionHistory):
 def __init__(self,length=30): super().__init__(length,'tail')
class FinActionHistory(ActionHistory):
 def __init__(self,length=30): super().__init__(length,'fin')
RLActionHistory=ActionHistory
