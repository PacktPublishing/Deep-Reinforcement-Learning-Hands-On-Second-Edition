import utime

from libhw import t1 as model

INPUT_SIZE = 28


def run():
    inputs = [[0.0]] * INPUT_SIZE
    print(inputs)

    while True:
        us = utime.ticks_us()
        _ = model.forward(inputs)
        us_n = utime.ticks_us()
        us_d = utime.ticks_diff(us_n, us)
        print(us_d)
