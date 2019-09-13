"""
Hyroscope sensor, datasheet: http://files.amperka.ru/datasheets/L3G4200D-hyroscope.pdf
"""
from . import st_family


class L3G4200D(st_family.STSensor):
    WHOAMI_VAL = 0xD3
    L3G4200D_SCALE_250DPS_MUL = 0.00875
    L3G4200D_DEG_TO_RAD = 0.01745329252

    def __init__(self, i2c, addr=104):
        super(L3G4200D, self).__init__(i2c, whoami_val=self.WHOAMI_VAL, addr=addr)
        self._init()

    def __repr__(self):
        return "L3G4200D(addr=%r)" % self.addr

    def __str__(self):
        return "L3G4200D"

    def _init(self):
        # ctrl1:
        # * default data rate (100HZ) and cut of
        # * device enabled
        # * all axises enabled
        self._write_reg(self.CTRL1_REG, 0b00001111)
        # ctrl4:
        # * BDU enabled: 1
        # * BLE in LSB: 0
        # * scale set to 250dps: 00
        # * self-test disabled: 0
        # * interface 4 wire: 0
        self._write_reg(self.CTRL4_REG, 0b10000000)

    def __len__(self):
        return 3*2

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
            st_family.decode_s16(b[ofs:ofs + 2]) * cls.L3G4200D_SCALE_250DPS_MUL * cls.L3G4200D_DEG_TO_RAD
            for ofs in range(0, 6, 2)
        ]

    @classmethod
    def preprocess(cls, vals):
        return vals
