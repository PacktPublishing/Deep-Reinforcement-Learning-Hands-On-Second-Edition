"""
Turns robot legs to zero positions
"""

from libhw import servo
import utime


PINS = ["B6", "B7", "B10", "B11"]
INV = [True, False, True, False]


def zero():
    ch = servo.pins_to_timer_channels(PINS)
    brain = servo.ServoBrain()
    brain.init(ch, inversions=INV)
    try:
        print(brain.positions)
        brain.positions = [0.0, 0.0, 0.0, 0.0]
        while True:
            utime.sleep_ms(1000)
    finally:
        brain.deinit()


def cycle(pin):
    ch = servo.pins_to_timer_channels(PINS)
    brain = servo.ServoBrain()
    brain.init(ch, inversions=INV)
    v = 0.0
    d = 0.1
    try:
        while True:
            brain.positions = [v if idx == pin else 0.0 for idx in range(len(PINS))]
            print(brain.positions)
            utime.sleep_ms(1000)
            v += d
            if abs(v) >= 1.0 or v <= 0:
                d = -d
    finally:
        brain.deinit()


def stand():
    ch = servo.pins_to_timer_channels(PINS)
    brain = servo.ServoBrain()
    brain.init(ch, inversions=INV)
    try:
        while True:
            brain.positions = [0.0, 0.0, 0.0, 0.0]
            utime.sleep_ms(1000)
            brain.positions = [0.5, 0.5, 0.5, 0.5]
            utime.sleep_ms(1000)
    finally:
        brain.deinit()


def set(actions):
    ch = servo.pins_to_timer_channels(PINS)
    brain = servo.ServoBrain()
    brain.init(ch, inversions=INV)
    try:
        while True:
            brain.positions = actions
            utime.sleep_ms(1000)
    finally:
        brain.deinit()


def set_old(actions):
    brain = servo.ServoBrainOld(PINS)
    brain.init()
    try:
        while True:
            brain.positions = actions
            utime.sleep_ms(1000)
    finally:
        brain.deinit()
