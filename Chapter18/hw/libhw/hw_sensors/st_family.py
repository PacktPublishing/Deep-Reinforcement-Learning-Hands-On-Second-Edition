"""
Generic family of sensors produced by ST
"""
from .. import sensors
from machine import I2C


class STSensor(sensors.Sensor):
    WHOAMI_REG = 0x0F

    CTRL1_REG = 0x20
    CTRL2_REG = 0x21
    CTRL3_REG = 0x22
    CTRL4_REG = 0x23
    CTRL5_REG = 0x24

    AXIS_X_REG = 0x28
    AXIS_Y_REG = 0x2A
    AXIS_Z_REG = 0x2C

    def __init__(self, i2c, whoami_val, addr):
        assert isinstance(i2c, I2C)
        assert isinstance(addr, int)
        super(STSensor, self).__init__()
        self.i2c = i2c
        self.addr = addr

        # check the device whoami
        val = self._read_reg_u8(self.WHOAMI_REG)
        if val != whoami_val:
            raise sensors.SensorInitError("Wrong value in WHOAMI register, expected %x, found %x",
                                          whoami_val, val)

    # warning: read functions are not ISR-safe!
    def _read_reg_u8(self, reg_addr):
        return ord(self.i2c.readfrom_mem(self.addr, reg_addr, 1))

    def _read_reg_u16(self, reg):
        l = self._read_reg_u8(reg)
        h = self._read_reg_u8(reg+1)
        return h * 256 + l

    def _read_reg_s16(self, reg):
        return twosComp(self._read_reg_u16(reg))

    def _write_reg(self, reg, val):
        self.i2c.writeto_mem(self.addr, reg, bytes([val]))


# Return a 16-bit signed number (two's compliment)
# Thanks to http://stackoverflow.com/questions/16124059/trying-to-
#   read-a-twos-complement-16bit-into-a-signed-decimal
def twosComp(x):
    if 0x8000 & x:
        x = - (0x010000 - x)
    return x


def decode_s16(buf):
    """
    Decode signed 16-bit value from bytebuffer
    :param buf: bytebuffer
    :return: int
    """
    v = (buf[1] << 8) + buf[0]
    return twosComp(v)
