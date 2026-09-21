import unittest
from types import SimpleNamespace
import numpy as np
from control.runtime.rl_action_history_20260914 import TailActionHistory,FinActionHistory


class ActionHistoryTests(unittest.TestCase):
    def event(self,action,start=0,end=200_000_000,written=True):
        return SimpleNamespace(action=action,start_t_ns=start,completion_t_ns=end,endpoint_written=written)

    def test_dimensions_padding_and_oldest_first(self):
        h = TailActionHistory(3)
        self.assertEqual(h.vector().shape,(9,))
        self.assertFalse(h.as_array().any())
        self.assertTrue(h.record_completion(self.event({'theta':10,'t':.2})))
        self.assertTrue(h.record_completion(self.event({'theta':20,'t':.2},200_000_000,400_000_000)))
        np.testing.assert_allclose(h.as_array(),[[0,0,0],[10,.2,1],[20,.2,1]])
        f = FinActionHistory(30)
        self.assertEqual(f.observation_dim,150)
        self.assertEqual(f.as_array().shape,(30,5))

    def test_completion_duration_and_pwm_both_required(self):
        h = TailActionHistory(2); action = {'theta':0,'t':.2}
        self.assertFalse(h.append(action,valid=True))
        self.assertFalse(h.append(action,completed=True,pwm_success=False,valid=True))
        self.assertFalse(h.record_completion(self.event(action,end=199_999_999)))
        self.assertFalse(h.record_completion(self.event(action,written=False)))
        self.assertFalse(h.record_completion(self.event(action,end=None)))
        self.assertEqual(len(h),0)
        self.assertTrue(h.record_completion(self.event(action)))
        self.assertFalse(h.record_completion(self.event(action)))
        self.assertEqual(len(h),1)

    def test_fixed_window_and_snapshot_isolation(self):
        h = TailActionHistory(2)
        for i in range(3): h.record_completion(self.event({'theta':i*10,'t':.2},i*200_000_000,(i+1)*200_000_000))
        x = h.as_array(); x[:]=0
        np.testing.assert_allclose(h.as_array()[:,0],[10,20])
        h.clear(); self.assertEqual(len(h),0)

    def test_fin_unflipped_direction_is_canonical_without_changing_width(self):
        h = FinActionHistory(3)
        for direction in (-1, 0, 1):
            h.append({'theta': 20, 't': .3, 'b1': 0, 'b2': direction},
                     completed=True, pwm_success=True)
        np.testing.assert_allclose(h.as_array(), [[20, .3, 0, 0, 1]] * 3)
        self.assertEqual(h.vector().shape, (15,))
        with self.assertRaises(ValueError):
            h.append({'theta': 20, 't': .3, 'b1': 1, 'b2': 0},
                     completed=True, pwm_success=True)


if __name__ == '__main__': unittest.main()
