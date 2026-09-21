import json, tempfile, unittest
from pathlib import Path
from control.runtime.rl_live_status_20260914 import LiveStatusPublisher

class LiveStatusTests(unittest.TestCase):
    def test_atomic_status_is_valid_json(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'rl_live_status_20260914.json'; LiveStatusPublisher(p).publish({'episode':1},force=True)
            self.assertEqual(json.loads(p.read_text())['episode'],1)

if __name__ == '__main__': unittest.main()
