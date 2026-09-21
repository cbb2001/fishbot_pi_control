"""Independent, completion-only histories with explicit zero padding masks."""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from threading import Lock
import math
import numpy as np


class ActionHistory:
    def __init__(self, length=30, kind='tail'):
        if isinstance(length,bool) or int(length) != length or int(length) < 0:
            raise ValueError('history length must be a nonnegative integer')
        if kind not in ('tail','fin'):
            raise ValueError('history kind must be tail or fin')
        self.length,self.kind = int(length),kind
        self.fields = ('theta','t') if kind == 'tail' else ('theta','t','b1','b2')
        self.width = len(self.fields)+1
        self._items = deque(maxlen=self.length)
        self._lock = Lock()
        self._seen = set()
        self._seen_order = deque()

    @property
    def observation_dim(self):
        return self.length*self.width

    def append(self, action=None, *, completed=False, pwm_success=False, valid=None,
               action_start_t_ns=None, completion_t_ns=None, action_id=None, **values):
        """Require both completion facts; ``valid=True`` cannot bypass them.

        Prefer record_completion(event), which checks the executor completion
        timestamp against the action duration as well as endpoint_written.
        """
        if not completed or not pwm_success or valid is False:
            return False
        action = values if action is None else action
        def get(name):
            return action[name] if isinstance(action,Mapping) else getattr(action,name)
        try:
            encoded = tuple(float(get(name)) for name in self.fields)
        except (KeyError,AttributeError,TypeError,ValueError) as exc:
            raise ValueError('completed action is missing required history fields') from exc
        if not all(math.isfinite(v) for v in encoded) or encoded[1] <= 0:
            raise ValueError('completed action contains invalid values')
        if self.kind == 'fin':
            if (encoded[2] not in (0.,1.) or encoded[3] not in (-1.,0.,1.)
                    or (encoded[2] == 1. and encoded[3] == 0.)):
                raise ValueError('fin b1/b2 invalid')
            if encoded[2] == 0.:
                encoded = (*encoded[:3], 0.)
        if (action_start_t_ns is None) != (completion_t_ns is None):
            raise ValueError('completion timing requires both start and completion timestamps')
        if completion_t_ns is not None and int(completion_t_ns)-int(action_start_t_ns) < round(encoded[1]*1e9):
            return False
        with self._lock:
            if action_id is not None and action_id in self._seen:
                return False
            self._items.append(encoded)
            if action_id is not None:
                self._seen.add(action_id)
                self._seen_order.append(action_id)
                # Deduplicate delivery retries without growing with session length.
                if len(self._seen_order) > max(1,self.length):
                    self._seen.discard(self._seen_order.popleft())
        return True

    def record_completion(self, event):
        def get(key, default=None):
            return event.get(key,default) if isinstance(event,Mapping) else getattr(event,key,default)
        action = get('action')
        start = get('start_t_ns',get('action_start_t_ns'))
        end = get('completion_t_ns',get('action_completion_t_ns'))
        if action is None or start is None or end is None or not get('endpoint_written',False):
            return False
        return self.append(action,completed=True,pwm_success=True,
                           action_start_t_ns=start,completion_t_ns=end,
                           action_id=(self.kind,int(start),int(end)))

    add = append
    record = record_completion

    def as_array(self, flatten=False):
        result = np.zeros((self.length,self.width),dtype=np.float32)
        with self._lock:
            items = tuple(self._items)
        for index,item in enumerate(items,start=self.length-len(items)):
            result[index,:-1] = item
            result[index,-1] = 1.
        return result.reshape(-1) if flatten else result

    def vector(self):
        return self.as_array(flatten=True)

    def snapshot(self):
        with self._lock:
            return list(self._items)

    def clear(self):
        with self._lock:
            self._items.clear()
            self._seen.clear()
            self._seen_order.clear()

    def __len__(self):
        with self._lock:
            return len(self._items)


class TailActionHistory(ActionHistory):
    def __init__(self,length=30):
        super().__init__(length,'tail')


class FinActionHistory(ActionHistory):
    def __init__(self,length=30):
        super().__init__(length,'fin')


RLActionHistory = ActionHistory
