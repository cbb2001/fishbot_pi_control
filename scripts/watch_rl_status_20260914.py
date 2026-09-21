#!/usr/bin/env python3
"""Read-only RL status watcher; closing it never affects training."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, help="session directory or status JSON path")
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()
    p = Path(args.session)
    if p.is_dir(): p = p / "rl_live_status_20260914.json"
    try:
        while True:
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                print(f"Episode {data.get('episode_index', '?')} | Policy {data.get('published_policy_version', '?')} | Role={data.get('episode_role', '?')}")
                print(f"h={data.get('latest_depth_m', data.get('h', '?'))} / {data.get('target_depth_m', data.get('h0', '?'))}  yaw={data.get('latest_yaw_deg', data.get('yaw', '?'))} / {data.get('target_heading_deg', data.get('theta0', '?'))}")
                print(f"reward={data.get('latest_reward', data.get('reward', '?'))}  PPO={data.get('candidate_training_active', '?')}  fault={data.get('fault', data.get('last_fault', 'none'))}\n", flush=True)
            except FileNotFoundError:
                print(f"waiting for {p}", flush=True)
            except Exception as exc:
                print(f"status read warning: {exc}", flush=True)
            time.sleep(max(0.1, args.interval))
    except KeyboardInterrupt:
        return 0

if __name__ == "__main__": raise SystemExit(main())
