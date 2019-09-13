"""
Accelerometer sensor, datasheet: https://www.st.com/resource/en/datasheet/cd00213470.pdf
"""
from . import st_family
import math as m


class Lis331DLH(st_family.STSensor):
    WHOAMI_VAL = 0x32
    RANGE_2G_DIVISOR = 16380

    def __init__(self, i2c, addr=24):
        super(Lis331DLH, self).__init__(i2c, self.WHOAMI_VAL, addr)
        self._init()

    def __repr__(self):
        return "Lis331DLH(addr=%r)" % self.addr

    def __str__(self):
        return "Lis331DLH"

    def _init(self):
        # reg4: set range 2G, enable BDU (to avoid reading low and high from diff samples), disable self-test
        self.i2c.writeto_mem(self.addr, self.CTRL4_REG, bytes([0b10000000]))
        # reg1: normal power mode, data rate=400Hz, all three axises enabled
        self.i2c.writeto_mem(self.addr, self.CTRL1_REG, bytes([0b00110111]))

    def __len__(self):
        return 6

    def refresh(self):
        self.i2c.readfrom_mem_into(self.addr, self.AXIS_X_REG, self._byte[0])
        self.i2c.readfrom_mem_into(self.addr, self.AXIS_X_REG+1, self._byte[1])
        self.i2c.readfrom_mem_into(self.addr, self.AXIS_Y_REG, self._byte[2])
        self.i2c.readfrom_mem_into(self.addr, self.AXIS_Y_REG+1, self._byte[3])
        self.i2c.readfrom_mem_into(self.addr, self.AXIS_Z_REG, self._byte[4])
        self.i2c.readfrom_mem_into(self.addr, self.AXIS_Z_REG+1, self._byte[5])

    @classmethod
    def decode(cls, b):
        return [
            st_family.decode_s16(b[ofs:ofs + 2]) / cls.RANGE_2G_DIVISOR
            for ofs in range(0, 6, 2)
        ]

    @classmethod
    def preprocess(cls, vals):
        # Subtract g vector (having magnitude of 1). Not the best way of doing such normalization
        # as error could be significant.
        l = m.sqrt(vals[0] * vals[0] + vals[1] * vals[1] + vals[2] * vals[2])
        if abs(l) > 1e-5:
            c = (l - 1) / l
            return [c * v for v in vals]
        return vals
