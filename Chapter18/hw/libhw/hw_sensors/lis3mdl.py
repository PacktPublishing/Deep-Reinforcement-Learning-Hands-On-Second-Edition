"""
Magnetometer sensor, datasheet: https://www.st.com/resource/en/datasheet/lis3mdl.pdf
"""
from . import st_family


class Lis3MDL(st_family.STSensor):
    WHOAMI_VAL = 0x3D

    TEMP_REG = 0x2E

    def __init__(self, i2c, addr=28):
        super(Lis3MDL, self).__init__(i2c, whoami_val=self.WHOAMI_VAL, addr=addr)
        self._init()

    def __repr__(self):
        return "Lis3MDL(addr=%r)" % self.addr

    def __str__(self):
        return "Lis3MDL"

    def _init(self):
        # ctrl1:
        # * temp sensor enabled: 1
        # * medium performance: 01
        # * output data rate = 10HZ: 100
        # * fast data rate: 0
        # * self-test: 1
        self._write_reg(self.CTRL1_REG, 0b10110001)
        # ctrl2:
        # * scale +/- 4 gauss: 00 (bits 7 and 6)
        self._write_reg(self.CTRL2_REG, 0b00000000)
        # ctrl3:
        # * low power: 0
        # * SPI mode: 0
        # * mode: 00
        self._write_reg(self.CTRL3_REG, 0b00000000)
        # ctrl4:
        # * Z-axis mode: medium: 01
        # * BLE: 0
        self._write_reg(self.CTRL4_REG, 0b00000100)
        # ctrl5:
        # * fast read: 0
        # * BDU: 1
        self._write_reg(self.CTRL5_REG, 0b01000000)

    def __len__(self):
        return 4*2

    def refresh(self):
        self.i2c.readfrom_mem_into(self.addr, self.AXIS_X_REG, self._byte[0])
        self.i2c.readfrom_mem_into(self.addr, self.AXIS_X_REG+1, self._byte[1])

        self.i2c.readfrom_mem_into(self.addr, self.AXIS_Y_REG, self._byte[2])
        self.i2c.readfrom_mem_into(self.addr, self.AXIS_Y_REG+1, self._byte[3])

        self.i2c.readfrom_mem_into(self.addr, self.AXIS_Z_REG, self._byte[4])
        self.i2c.readfrom_mem_into(self.addr, self.AXIS_Z_REG+1, self._byte[5])

        self.i2c.readfrom_mem_into(self.addr, self.TEMP_REG, self._byte[6])
        self.i2c.readfrom_mem_into(self.addr, self.TEMP_REG+1, self._byte[7])
