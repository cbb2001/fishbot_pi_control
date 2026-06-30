# -*- coding: utf-8 -*-
import time

try:
    import smbus
except ImportError:
    import smbus2 as smbus


class MS5837:
    MS5837_I2C_ADDRESS = 0x76

    _CMD_RESET = 0x1E
    _CMD_ADC_READ = 0x00
    _CMD_CONVERT_D1_8192 = 0x4A
    _CMD_CONVERT_D2_8192 = 0x5A

    _PROM_READ_C1 = 0xA2
    _MBAR_TO_CM_H2O = 1.019716

    def __init__(self, bus=1, addr=MS5837_I2C_ADDRESS):
        self.i2cbus = smbus.SMBus(bus)
        self.i2c_addr = addr
        self.c = [0] * 7
        self.surface_pressure_mbar = 1144.0
        self.temperature_C = 0.0
        self.pressure_mbar = 0.0
        self.depth_cm = 0.0

    def begin(self):
        try:
            self.reset()
            time.sleep(0.01)
            for i in range(1, 7):
                self.c[i] = self._read_prom(self._PROM_READ_C1 + (i - 1) * 2)
            return True
        except OSError:
            print("I2C init fail")
            return False

    def reset(self):
        self.i2cbus.write_byte(self.i2c_addr, self._CMD_RESET)

    def set_zero(self):
        self.update()
        self.surface_pressure_mbar = self.pressure_mbar

    def update(self):
        d1 = self._read_adc(self._CMD_CONVERT_D1_8192)
        d2 = self._read_adc(self._CMD_CONVERT_D2_8192)

        dT = d2 - self.c[5] * 256
        temp = 2000 + dT * self.c[6] / 8388608
        off = self.c[2] * 65536 + self.c[4] * dT / 128
        sens = self.c[1] * 32768 + self.c[3] * dT / 256

        if temp < 2000:
            ti = 3 * dT * dT / 8589934592
            offi = 3 * (temp - 2000) * (temp - 2000) / 2
            sensi = 5 * (temp - 2000) * (temp - 2000) / 8
            if temp < -1500:
                offi += 7 * (temp + 1500) * (temp + 1500)
                sensi += 4 * (temp + 1500) * (temp + 1500)
        else:
            ti = 2 * dT * dT / 137438953472
            offi = (temp - 2000) * (temp - 2000) / 16
            sensi = 0

        temp -= ti
        off -= offi
        sens -= sensi

        self.temperature_C = temp / 100
        self.pressure_mbar = ((d1 * sens / 2097152 - off) / 8192) / 10
        self.depth_cm = (self.pressure_mbar - self.surface_pressure_mbar) * self._MBAR_TO_CM_H2O

    def get_temperature_C(self):
        self.update()
        return self.temperature_C

    def get_pressure_mbar(self):
        self.update()
        return self.pressure_mbar

    def get_depth_cm(self):
        self.update()
        return self.depth_cm

    def get_depth_m(self):
        return self.get_depth_cm() / 100

    def get_data(self):
        self.update()
        return {
            "temperature_C": self.temperature_C,
            "pressure_mbar": self.pressure_mbar,
            "depth_cm": self.depth_cm,
            "depth_m": self.depth_cm / 100,
        }

    def close(self):
        self.i2cbus.close()

    def _read_prom(self, cmd):
        data = self.i2cbus.read_i2c_block_data(self.i2c_addr, cmd, 2)
        return data[0] * 256 + data[1]

    def _read_adc(self, cmd):
        self.i2cbus.write_byte(self.i2c_addr, cmd)
        time.sleep(0.02)
        data = self.i2cbus.read_i2c_block_data(self.i2c_addr, self._CMD_ADC_READ, 3)
        return data[0] * 65536 + data[1] * 256 + data[2]


if __name__ == "__main__":
    sensor = MS5837(1)

    while not sensor.begin():
        time.sleep(1)

    while True:
        data = sensor.get_data()
        print("Temp: %.2f C" % data["temperature_C"])
        print("Pressure: %.2f mbar" % data["pressure_mbar"])
        print("Depth: %.2f cm" % data["depth_cm"])
        print("")
        time.sleep(1)
