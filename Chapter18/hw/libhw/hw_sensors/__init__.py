from .. import sensors
from . import lis331dlh
from . import lis3mdl
from . import l3g4200d

SENSOR_CLASSES = (lis331dlh.Lis331DLH, lis3mdl.Lis3MDL, l3g4200d.L3G4200D)


def scan(i2c):
    """
    Return list of detected sensors on the bus. Default addresses are used
    :return: list of Sensors instances
    """
    res = []
    for c in SENSOR_CLASSES:
        try:
            s = c(i2c)
            res.append(s)
        except sensors.SensorInitError:
            pass
    return res


def full_scan(i2c):
    """
    Perform full scan of the bus -- try every class for every device on the bus, which is longer, but
    detects compatible devices on non-standard addresses.
    :param i2c:
    :return: list of Sensors instances
    """
    res = []
    for dev in i2c.scan():
        for c in SENSOR_CLASSES:
            try:
                s = c(i2c, dev)
                res.append(s)
            except sensors.SensorInitError:
                pass
    return res
