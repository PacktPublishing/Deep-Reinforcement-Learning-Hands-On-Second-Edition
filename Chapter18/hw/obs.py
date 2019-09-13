"""
Simple orientation calculation from Accelerometer
"""
import pyb
from machine import I2C
from libhw.hw_sensors import lis331dlh as lis
import math

SDA = 'X12'
SCL = 'Y11'


def simple(gx, gy, gz):
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


def run():
    i2c = I2C(freq=400000, scl=SCL, sda=SDA)
    acc = lis.Lis331DLH(i2c)
    while True:
        pyb.delay(500)
        acc.refresh()
        res = acc.decode(acc.buffer)
        pitch, roll = simple(*res)
        print("pitch=%s, roll=%s" % (pitch, roll))
