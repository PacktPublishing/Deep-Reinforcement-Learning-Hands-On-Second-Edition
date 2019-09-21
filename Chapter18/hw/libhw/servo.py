import pyb


class ServoBrain:
    FREQ = 50               # 20ms -- standard pulse interval for servos
    MIN_PERCENT = 2.3
    MAX_PERCENT = 12.7
    MIN_POS = 0
    MAX_POS = 1

    def __init__(self):
        """
        Construct servo brain
        """
        self._timer_channels = None
        self._inversions = None
        self._channels = None
        self._positions = None
        self._timers = None

    def init(self, timer_channels, inversions=None):
        """
        :param timer_channels: list of tuples (pin_name, (timer, channel))
        :param inversions: list of bools specifying servos to be inverted, if None, no inversions
        """
        self._timer_channels = timer_channels
        self._inversions = inversions if inversions is not None else [False for _ in timer_channels]
        self._timers = {}
        self._channels = []
        self._positions = []
        for pin_name, (t_idx, ch_idx) in timer_channels:
            if t_idx not in self._timers:
                self._timers[t_idx] = pyb.Timer(t_idx, freq=self.FREQ)
            pin = pyb.Pin(pin_name, pyb.Pin.OUT)
            self._channels.append(self._timers[t_idx].channel(ch_idx, pyb.Timer.PWM, pin=pin))
            self._positions.append(0.0)
        self._apply_positions(self._positions)

    def deinit(self):
        if self._timers is None:
            return
        for t in self._timers.values():
            t.deinit()
        self._timer_channels = None
        self._channels = None
        self._positions = None
        self._timers = None

    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, value):
        self._positions = value
        self._apply_positions(self._positions)

    @classmethod
    def position_to_percent(cls, pos):
        return (pos-cls.MIN_POS)*(cls.MAX_PERCENT - cls.MIN_PERCENT)/(cls.MAX_POS - cls.MIN_POS) + cls.MIN_PERCENT

    def _apply_positions(self, values):
        for p, ch, inv in zip(values, self._channels, self._inversions):
            if inv:
                p = self.MAX_POS - p
            ch.pulse_width_percent(self.position_to_percent(p))


_PINS_TO_TIMER = {
    "B6": (4, 1),
    "B7": (4, 2),
    "B8": (4, 3),
    "B9": (4, 4),
    "B10": (2, 3),
    "B11": (2, 4),
}


def pins_to_timer_channels(pins):
    """
    Convert list of pins to list of tuples (timer, channel). This function is hardware-specific
    :param pins: list of pin names
    :return: list of (pin_name, (timer, channel)) tuples
    """
    res = []
    for p in pins:
        pair = _PINS_TO_TIMER.get(p)
        assert pair is not None
        res.append((p, pair))
    return res


class ServoBrainOld:
    """
    Shouldn't be used, kept only for demonstration purposes
    """
    MIN_POS = -10
    MAX_POS = 10
    NEUT_POS = 0

    _MIN_TIME_US = 500
    _MAX_TIME_US = 2500
    _NEUTRAL_TIME_US = 1500
    _INTERVAL_US = 10000
    _BASE_FREQ = 1000000

    def __init__(self, servo_pins, base_timer_index=1):
        if isinstance(servo_pins, str):
            servo_pins = (servo_pins, )
        self.servo_pins = servo_pins
        self.base_timer_index = base_timer_index
        self._base_timer = pyb.Timer(base_timer_index)
        self._pins = []
        self._pin_timers = []
        self._positions = []
        self._frequencies = []
        self._pin_callbacks = []

    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, new_pos):
        self._positions = new_pos
        self._frequencies = [self.pos_to_freq(p) for p in new_pos]

    @classmethod
    def pos_to_freq(cls, pos):
        if pos < cls.MIN_POS:
            pos = cls.MIN_POS
        elif pos > cls.MAX_POS:
            pos = cls.MAX_POS
        return cls._BASE_FREQ // (cls._MIN_TIME_US + (pos + 10) * 100)

    def init(self):
        self.positions = [self.NEUT_POS for _ in self.servo_pins]
        for idx, pin in enumerate(self.servo_pins):
            self._pin_timers.append(pyb.Timer(idx + self.base_timer_index + 1))
            p = pyb.Pin(pin, pyb.Pin.OUT)
            p.init(pyb.Pin.OUT, pyb.Pin.PULL_UP)
            p.low()
            self._pins.append(p)
            self._pin_callbacks.append(self._get_pin_callback(idx))

        self._base_timer.init(freq=self._BASE_FREQ//self._INTERVAL_US)
        self._base_timer.callback(lambda t: self._base_callback())

    def deinit(self):
        self._base_timer.deinit()
        self._pins.clear()
        for t in self._pin_timers:
            t.deinit()
        self._pin_timers.clear()
        self._positions.clear()
        self._pin_callbacks.clear()

    def _get_pin_callback(self, idx):
        def func(t):
            if not self._pins[idx].value():
                self._pins[idx].high()
            else:
                self._pins[idx].low()
                t.deinit()
        return func

    def _base_callback(self):
        for idx in range(len(self.servo_pins)):
            self._pin_timers[idx].callback(self._pin_callbacks[idx])
            self._pin_timers[idx].init(freq=self._frequencies[idx])
