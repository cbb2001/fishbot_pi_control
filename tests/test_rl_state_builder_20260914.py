import copy
import json
import math
import queue
import threading
import unittest
from types import SimpleNamespace
import numpy as np

from control.runtime.ring_buffer import RingBuffer
from control.runtime.sample import SensorSample
from control.runtime.rl_sensor_calibration_20260914 import RLSensorCalibration
from control.runtime.rl_state_builder_20260914 import RLStateBuilder,RLStateError
from control.runtime.rl_state_buffer_20260914 import RLStateBuffer,RLStateSyncWorker
from control.runtime.rl_reward_20260914 import RLReward


class Tracker:
    def __init__(self): self.last_query = None
    def get_servo_pose_at(self,t):
        self.last_query = t
        return {'query_t_ns':t,'estimated_angles_deg':{i:80.+i for i in range(1,8)},
                'reference_angles_deg':{i:80.+i for i in range(1,8)},
                'commanded_angles_deg':{i:80.+i for i in range(1,8)}}


class StateBuilderTests(unittest.TestCase):
    def setUp(self):
        self.imu = {'pitch_deg':0.,'yaw_deg':0.,'roll_deg':0.,'acc_mps2':[0.,0.,9.80665],'quat':[1.,0.,0.,0.]}
        self.calibration = RLSensorCalibration({'minimum_imu_samples':1,'minimum_depth_samples':1})
        self.calibration.fit([self.imu],[101325.])
        self.t = 1_000_000_000
        self.snapshot = {name:{'sample_t_ns':self.t,'valid':True,'data':copy.deepcopy(data)} for name,data in (
            ('imu',self.imu),('depth',{'pressure_pa':101325.+9806.65*.5,'depth_m':999}),('power',{'power_w':12.}))}
        self.tracker = Tracker()

    def builder(self,src=None,**kw):
        return RLStateBuilder(self.snapshot if src is None else src,self.calibration,self.tracker,
                              reward=RLReward(target_depth_m=.5),**kw)

    def test_all_fifteen_values_diagnostics_and_reward(self):
        state = self.builder().build(self.t)
        np.testing.assert_allclose(state.observation,[0,0,0,0,0,0,.5,12,81,82,83,84,85,86,87],atol=1e-7)
        self.assertEqual(self.tracker.last_query,self.t)
        self.assertEqual(state.raw_acc_mps2,(0.,0.,9.80665))
        self.assertEqual(state.reward.reward,4.)
        self.assertEqual(state.sensor_sample_t_ns,dict.fromkeys(('imu','depth','power'),self.t))
        self.assertIn('servo_pose',state.to_dict())
        json.dumps(state.to_dict(),allow_nan=False)
        with self.assertRaises(ValueError): state.observation[0]=99

    def test_each_critical_sensor_missing_stale_future_invalid(self):
        for name in ('imu','depth','power'):
            for mode in ('missing','stale','future','invalid','timestamp_missing'):
                src=copy.deepcopy(self.snapshot)
                if mode=='missing':del src[name]
                elif mode=='stale':src[name]['sample_t_ns']=0
                elif mode=='future':src[name]['sample_t_ns']=self.t+1
                elif mode=='invalid':src[name]['valid']=False
                else:del src[name]['sample_t_ns']
                with self.subTest(name=name,mode=mode), self.assertRaises(RLStateError) as caught:
                    self.builder(src,max_sensor_age_ms=dict.fromkeys(('imu','depth','power'),100)).build(self.t)
                self.assertEqual(caught.exception.sensor,name)

    def test_no_zero_or_old_depth_fallback_and_nonfinite(self):
        for name,key in (('imu','pitch_deg'),('imu','acc_mps2'),('imu','quat'),('depth','pressure_pa'),('power','power_w')):
            for invalid in ('missing','nan'):
                src=copy.deepcopy(self.snapshot)
                if invalid=='missing': del src[name]['data'][key]
                else: src[name]['data'][key]=[float('nan')]*len(src[name]['data'][key]) if isinstance(src[name]['data'][key],list) else float('nan')
                with self.subTest(name=name,key=key,invalid=invalid),self.assertRaises(RLStateError):self.builder(src).build(self.t)
        self.tracker.get_servo_pose_at=lambda t:{'estimated_angles_deg':{i:90 for i in range(1,7)}}
        with self.assertRaises(RLStateError):self.builder().build(self.t)

    def test_warning_does_not_exceed_fault_threshold(self):
        self.snapshot['imu']['sample_t_ns']=self.t-80_000_000
        state=self.builder(config={'state':{'sensor_age':{'imu':{'warn_ms':50,'fault_ms':100}}}}).build(self.t)
        self.assertEqual(state.sensor_age_ms['imu'],80.)
        self.assertEqual(len(state.warnings),1)

    def test_ringbuffer_reuse_never_takes_future_frame(self):
        buffers={name:RingBuffer(5) for name in ('imu','depth','power')}
        for name,field in self.snapshot.items():
            buffers[name].append(SensorSample(name,self.t,1,field['data']))
            future=copy.deepcopy(field['data'])
            if name=='imu':future['yaw_deg']=90
            buffers[name].append(SensorSample(name,self.t+1,2,future))
        state=self.builder(buffers).build(self.t)
        self.assertEqual(state.yaw_deg,0.)
        self.assertEqual(state.sensor_sample_t_ns['imu'],self.t)

    def test_uncalibrated_state_rejected(self):
        self.calibration.gravity_validated=False
        with self.assertRaises(RLStateError):self.builder().build(self.t)

    def test_disabled_freshness_reuses_old_valid_buffer_samples(self):
        buffers = {name: RingBuffer(5) for name in ('imu', 'depth', 'power')}
        for name, field in self.snapshot.items():
            buffers[name].append(SensorSample(name, self.t, 1, field['data']))
        cfg = {'state': {'enforce_data_freshness': False}}
        builder = self.builder(buffers, config=cfg)
        state = builder.build(self.t + 10_000_000_000)
        self.assertEqual(state.sensor_age_ms, dict.fromkeys(buffers, 10000.))
        self.assertEqual(len(state.warnings), 3)
        self.assertAlmostEqual(state.depth_m, .5)
        self.assertTrue(all(builder.last_synchronized_sample[n]['valid'] for n in buffers))
        buffers['imu'].append(SensorSample('imu', self.t, 2, self.imu, ok=False, error='decode failed'))
        with self.assertRaisesRegex(RLStateError, 'invalid sample'):
            builder.build(self.t + 10_000_000_000)

    def test_buffer_interval_mean_and_worker_exception_propagation(self):
        buf=RLStateBuffer(5); builder=self.builder()
        worker=RLStateSyncWorker(builder,buf,clock_ns=lambda:self.t)
        worker.sample_once(self.t)
        state=builder.build(self.t+10)
        buf.append(state)
        self.assertIs(buf.latest_before(self.t+5),buf.snapshot()[0])
        self.assertIsNone(buf.latest_before(self.t-1))
        self.assertEqual(len(buf.between(self.t,self.t+5)),1)
        self.assertEqual(buf.reward_stats(self.t,self.t+10)['reward_mean'],4.)
        self.assertEqual(buf.reward_stats(self.t,self.t+10)['reward_sample_count'],2)
        with self.assertRaises(ValueError):buf.append(state)
        broken=SimpleNamespace(build=lambda t:(_ for _ in ()).throw(RLStateError('depth lost',sensor='depth')))
        stop=threading.Event(); failures=queue.Queue()
        worker=RLStateSyncWorker(broken,stop_event=stop,failure_queue=failures)
        worker.start(); worker.join(1.)
        self.assertTrue(stop.is_set())
        self.assertFalse(worker.is_alive())
        self.assertIn('depth lost',failures.get_nowait()['message'])
        with self.assertRaisesRegex(RuntimeError,'depth lost'):worker.raise_if_failed()

    def test_worker_absolute_schedule_skips_overdue_ticks(self):
        now=[0]; stop=threading.Event(); observed=[]
        def build(t):
            observed.append(t)
            if len(observed)==1: now[0]=100_000_000
            if len(observed)==3: stop.set()
            return SimpleNamespace(t_ns=t)
        def wait(event,seconds):
            now[0]+=round(seconds*1e9)
        worker=RLStateSyncWorker(SimpleNamespace(build=build),rate_hz=30,
            stop_event=stop,clock_ns=lambda:now[0],wait_fn=wait)
        worker._run()
        self.assertEqual(observed,[0,99_999_999,133_333_332])
        self.assertEqual(worker.skipped_tick_count,2)
        self.assertEqual(worker.sample_count,3)

    def test_early_wakeup_never_publishes_future_state(self):
        now = [0]
        stop = threading.Event()
        observed = []
        waits = []
        def build(t):
            self.assertLessEqual(t, now[0])
            observed.append(t)
            if len(observed) == 2:
                stop.set()
            return SimpleNamespace(t_ns=t)
        def early_wait(event, seconds):
            waits.append(seconds)
            now[0] += 10_000_000 if len(waits) == 1 else round(seconds * 1e9)
        worker = RLStateSyncWorker(SimpleNamespace(build=build), rate_hz=30,
            stop_event=stop, clock_ns=lambda: now[0], wait_fn=early_wait)
        worker._run()
        self.assertEqual(observed, [0, 33_333_333])


if __name__ == '__main__':unittest.main()
