#!/usr/bin/env python3
"""Read a detached IMU capture's status; never open hardware or signal a process."""
import argparse
import json
import time
from pathlib import Path


def tail_json(path,limit):
    if not path.exists():return []
    with path.open('rb') as f:
        f.seek(0,2);size=f.tell();f.seek(max(0,size-limit));data=f.read()
    rows=[]
    for line in data.splitlines():
        try:rows.append(json.loads(line))
        except (ValueError,UnicodeDecodeError):pass
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('job',type=Path)
    args=p.parse_args();job=json.loads((args.job/'job.json').read_text())
    session=args.job/'session';result=session/'result.json'
    raw=session/'raw_imu.jsonl';rows=tail_json(raw,16384)
    status=dict(pid=job['pid'],process_present=Path(f'/proc/{job["pid"]}').exists(),
                raw_bytes=raw.stat().st_size if raw.exists() else 0,
                expected_end_shanghai=job['expected_end_shanghai'])
    if rows:
        r=rows[-1]
        status.update(samples=r['seq'],frame_duration_s=(r['seq']-1)/200.,
            latest_data_age_s=(time.monotonic_ns()-r['receive_t_ns'])/1e9,
            crc_errors=r['data']['crc_error_count'],resync_count=r['data']['decode_resync_count'],
            discarded_bytes=r['data']['discarded_bytes'],last_tid=r['data']['tid'])
    periodic=tail_json(session/'periodic_live.jsonl',524288)
    events=[r for r in periodic if r.get('event')=='periodic_boundary']
    if events:
        e=events[-1];status.update(latest_boundary=e['cycle'],boundary_time_s=e['time_s'],
                                   latest_boundary_accepted=e['accepted'])
    if result.exists():status['result']=json.loads(result.read_text())
    if not status['process_present'] and not result.exists():
        status['console_tail']=(args.job/'console.log').read_text()[-2000:]
    print(json.dumps(status,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
