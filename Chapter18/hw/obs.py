"""
Simple orientation calculation from Accelerometer
"""
import pyb
from machine import I2C
from libhw.hw_sensors import lis331dlh as lis
from libhw.sensor_buffer import SensorsBuffer
from libhw.postproc import PostPitchRoll

SDA = 'X12'
SCL = 'Y11'


def run():
    i2c = I2C(freq=400000, scl=SCL, sda=SDA)
    acc = lis.Lis331DLH(i2c)
    buf = SensorsBuffer([acc], timer_index=1, freq=100,
                        batch_size=10, buffer_size=100)
    post = PostPitchRoll(buf, pad_yaw=True)
    buf.start()
    try:
        while True:
            for v in post:
                print("pitch=%s, roll=%s, yaw=%s" % tuple(v))
    finally:
        buf.stop()
