from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from control.runtime.absolute_discrete_actions import MissionValidationError
from control.runtime.absolute_manual_action_provider import ManualSequenceProvider


class AbsoluteManualProviderTests(unittest.TestCase):
    def _load(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mission.yaml"
            path.write_text(text, encoding="utf-8")
            return ManualSequenceProvider(path).load()

    def test_exact_absolute_schema(self):
        mission = self._load(
            "name: ok\n"
            "tail_actions: [{theta1: 80, theta2: 90, theta3: 90, v1: 1, v2: 2, v3: 3}]\n"
            "left_fin_actions: [{theta: 121, v: 10, b: 0}]\n"
            "right_fin_actions: []\n")
        self.assertEqual(mission.tail_actions[0].theta1, 80)

    def test_rejects_legacy_and_boolean_fields(self):
        with self.assertRaises(MissionValidationError):
            self._load("name: bad\ntail_actions: [{t: 1, A: 2}]\nleft_fin_actions: []\nright_fin_actions: []\n")
        with self.assertRaises(MissionValidationError):
            self._load("name: bad\ntail_actions: []\nleft_fin_actions: [{theta: 121, v: 10, b: true}]\nright_fin_actions: []\n")


if __name__ == "__main__":
    unittest.main()
