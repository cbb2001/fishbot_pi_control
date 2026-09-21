"""回归：smbus2 要求明确指定寄存器读取字节数。"""
import importlib.util
from pathlib import Path
import types
import unittest
from unittest.mock import patch

from drivers.ina219 import INA219Sensor


class RegisterBus:
    def __init__(self, bus):
        self.registers = {0: 0, 2: 24000, 3: 480, 4: 800}

    def read_byte(self, address):
        return 0

    def read_i2c_block_data(self, address, register, length):
        if length != 2:
            raise ValueError('INA219 registers contain two bytes')
        value = self.registers[register]
        return [value >> 8, value & 255]

    def write_i2c_block_data(self, address, register, values):
        self.registers[register] = (values[0] << 8) | values[1]

    def close(self):
        pass


class INA219SMBusTests(unittest.TestCase):
    def test_begin_and_read_with_required_smbus2_length(self):
        vendor_path = Path(__file__).resolve().parents[1] / 'drivers/DFRobot_INA219/Python/RespberryPi/DFRobot_INA219.py'
        spec = importlib.util.spec_from_file_location('ina219_vendor_test', vendor_path)
        vendor = importlib.util.module_from_spec(spec)
        with patch.dict('sys.modules', {'smbus': types.SimpleNamespace(SMBus=RegisterBus)}):
            spec.loader.exec_module(vendor)
        sensor = INA219Sensor()
        self.addCleanup(sensor.close)
        with patch.object(sensor, '_load_vendor_class', return_value=vendor.INA219):
            self.assertTrue(sensor.begin())
        reading = sensor.read()
        self.assertAlmostEqual(reading.voltage_v, 12.0)
        self.assertAlmostEqual(reading.current_a, .8)
        self.assertAlmostEqual(reading.power_w, 9.6)


if __name__ == '__main__':
    unittest.main()
