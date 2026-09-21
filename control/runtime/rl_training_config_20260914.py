from __future__ import annotations
import copy, math
from pathlib import Path
def prepare_config(config, *, overrides=None, dry_run=False):
    cfg=copy.deepcopy(config); overrides=overrides or {}
    for section in ("targets","startup","calibration","state","reward","fin_ppo","training","runtime"):
        if not isinstance(cfg.get(section),dict): raise ValueError("缺少配置段 "+section)
    if "tail_ppo" in cfg: raise ValueError("当前仅支持侧鳍 Fin")
    for arg,(sec,key) in {"h0":("targets","depth_m"),"theta0":("targets","heading_deg"),"max_episodes":("training","max_episodes"),"max_training_s":("training","max_training_s"),"episode_duration_s":("training","episode_duration_s"),"pretrain_wait_s":("startup","pretrain_wait_s")}.items():
        if overrides.get(arg) is not None: cfg[sec][key]=overrides[arg]
    if dry_run:
        sim=cfg.get("simulation",{})
        for k in ("pretrain_wait_s","sensor_warmup_s","calibration_window_s","calibration_timeout_s"):
            if k in sim: cfg["startup"][k]=sim[k]
        for k in ("minimum_imu_samples","minimum_depth_samples"):
            if k in sim: cfg["calibration"][k]=sim[k]
    t=cfg["training"]
    # 等待和监测参数必须在打开传感器或执行器前校验，NaN 不得绕过检查。
    cfg['runtime'].setdefault('safety_poll_interval_s', .01)
    cfg['state'].setdefault('stop_on_stale_decision', True)
    if not isinstance(cfg['state']['stop_on_stale_decision'], bool):
        raise ValueError('state.stop_on_stale_decision 必须为 true/false 布尔值')
    cfg['state'].setdefault('enforce_data_freshness', True)
    if not isinstance(cfg['state']['enforce_data_freshness'], bool):
        raise ValueError('state.enforce_data_freshness 必须为 true/false 布尔值')
    for section, keys, allow_zero in (
        ('startup', ('pretrain_wait_s', 'sensor_warmup_s'), True),
        ('startup', ('calibration_window_s', 'calibration_timeout_s'), False),
        ('runtime', ('poll_interval_s', 'safety_poll_interval_s', 'update_timeout_s',
                     'action_timeout_margin_s', 'sensor_ready_timeout_s', 'shutdown_timeout_s',
                     'live_status_interval_s'), False),
        ('state', ('max_state_age_ms', 'sample_hz'), False),
    ):
        for key in keys:
            value = float(cfg[section][key])
            if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
                raise ValueError(f'{section}.{key} 非法')
    for section, key in (('runtime', 'torch_num_threads'), ('state', 'fin_history_length')):
        value = cfg[section][key]
        if isinstance(value, bool) or not math.isfinite(float(value)) or value <= 0 or int(value) != value:
            raise ValueError(f'{section}.{key} 必须为正整数')
    if cfg['startup']['calibration_timeout_s'] < cfg['startup']['calibration_window_s']:
        raise ValueError('标定超时不得小于标定窗口')
    if not isinstance(cfg.get('safety', {}), dict):
        raise ValueError('safety 必须为配置字典')
    for key in ('depth_m', 'heading_deg'):
        if not math.isfinite(float(cfg['targets'][key])):
            raise ValueError('targets.' + key + ' 必须为有限数')
    for k in ("update_every_actions","max_episodes","min_actions"):
        v=t.get(k, cfg["state"].get("fin_history_length") if k=="min_actions" else None)
        if v is None or isinstance(v,bool) or int(v)!=v or int(v)<=0: raise ValueError(k+"必须为正整数")
    for k in ("episode_duration_s","depth_tolerance_m","yaw_tolerance_deg","success_hold_s","min_episode_s","max_sample_gap_s","discount_reference_dt_s"):
        if not math.isfinite(float(t[k])) or float(t[k])<0: raise ValueError(k+"非法")
    if t["episode_duration_s"]<=0 or t["yaw_tolerance_deg"]>180: raise ValueError("episode参数非法")
    if t.get("max_training_s") is not None and (not math.isfinite(float(t["max_training_s"])) or float(t["max_training_s"])<=0): raise ValueError("max_training_s非法")
    if t.get("update_hold_mode","endpoint")!="endpoint": raise ValueError("仅支持endpoint hold")
    if cfg["fin_ppo"]["min_transitions_per_update"]<2 or cfg["fin_ppo"]["min_transitions_per_update"]>t["update_every_actions"]: raise ValueError("min_transitions_per_update应在2和update_every_actions之间")
    from .rl_ppo_agent_20260914 import PPOConfig
    PPOConfig.from_dict(cfg["fin_ppo"]); return cfg
def resolve_resume(value, log_root):
    if value=="latest":
        p=sorted(Path(log_root).glob("*/checkpoint_20260914.json"),key=lambda x:x.stat().st_mtime,reverse=True)
        if not p: raise FileNotFoundError("找不到latest checkpoint")
        return p[0]
    p=Path(value).expanduser().resolve()
    if p.is_dir(): p=p/"checkpoint_20260914.json"
    if not p.is_file(): raise FileNotFoundError(str(p))
    return p
