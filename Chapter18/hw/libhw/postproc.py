"""
Performs sensors reading postprocessing
"""
from .sensor_buffer import SensorsBuffer
from . import hw_sensors

import math
import collections


class Smoother:
    def __init__(self, window, components=3):
        self._window = window
        self._components = components
        self._buf = collections.deque(tuple(), window+1)         # deque class in micropython is a bit limiting...
        self._sums = [0.0 for _ in range(components)]

    def push(self, vals):
        for idx, v in enumerate(vals):
            self._sums[idx] += v
        self._buf.append(vals)
        while len(self._buf) > self._window:
            for idx, v in enumerate(self._buf.popleft()):
                self._sums[idx] -= v

    def values(self):
        return [s / self._window for s in self._sums]

    def sums(self):
        return list(self._sums)


class PostPitchRoll:
    """
    Takes raw accelerometer values, smooths it and calculates pitch and roll using simple method
    """
    SMOOTH_WINDOW = 50

    def __init__(self, buffer, pad_yaw):
        assert isinstance(buffer, SensorsBuffer)
        assert len(buffer.sensors) == 1
        assert isinstance(buffer.sensors[0],
                          hw_sensors.lis331dlh.Lis331DLH)
        self.buffer = buffer
        self.smoother = Smoother(self.SMOOTH_WINDOW, components=3)
        self.pad_yaw = pad_yaw

    def __iter__(self):
        for b_list in self.buffer:
            for b in b_list:
                data = hw_sensors.lis331dlh.Lis331DLH.decode(b)
                self.smoother.push(data)
                pitch, roll = \
                    pitch_roll_simple(*self.smoother.values())
                res = [pitch, roll]
                if self.pad_yaw:
                    res.append(0.0)
                yield res


def pitch_roll_simple(gx, gy, gz):
    """
    Simple calculation of pitch and roll from accelerometer
    https://theccontinuum.com/2012/09/24/arduino-imu-pitch-roll-from-accelerometer/
    :param gx, gy, gz: acceleration in g
    :return: tuple with (pitch, roll)
    """
    g_xz = math.sqrt(gx*gx + gz*gz)
    pitch = math.atan2(gy, g_xz)
    roll = math.atan2(-gx, gz)
    return pitch, roll
