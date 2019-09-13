"""
Circular buffer which gathers data using timer interrupts
"""
import pyb


class SensorsBuffer:
    """
    Implements circular data buffer for SD saving and enqueue performed by timer callback
    """
    def __init__(self, sensors, timer_index, freq, batch_size, buffer_size):
        """
        Constructs the buffer
        :param sensors: list of sensors to be queried
        :param timer_index: index of timer to be used for data gather
        :param freq: frequency of query
        :param batch_size: size of the single batch
        :param buffer_size: buffer size in batches
        """
        self.sensors = sensors
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        # total data produced by our sensors
        data_len = sum(map(len, sensors))
        # circular buffer to be filled by sensor query process
        self._buffer = [
            [bytearray(data_len) for _ in range(batch_size)]
            for _ in range(buffer_size)
        ]
        # head of the buffer - index of the first batch to be written (if equals to _tail -- buffer empty)
        self._head = 0
        # tail of the buffer - index of the batch being written
        self._tail = 0
        # offset next empty block in current batch
        self._batch_ofs = 0

        self.timer = pyb.Timer(timer_index, freq=freq)

    def start(self):
        self.timer.callback(self.callback)

    def stop(self):
        self.timer.callback(None)

    def __repr__(self):
        return "SensorsBuffer(batches_full=%d, batches_total=%d, batch_size=%d)" % (
            abs(self._tail - self._head), self.buffer_size, self.batch_size
        )

    def callback(self, t):
        ofs = 0
        buf = self._buffer[self._tail][self._batch_ofs]
        for sensor in self.sensors:
            sensor.refresh()

            # couldn't find better way to copy buffer without allocating memory :(
            for i in range(len(sensor)):
                buf[ofs] = sensor.buffer[i]
                ofs += 1
        self._batch_ofs = (self._batch_ofs + 1) % self.batch_size
        if self._batch_ofs == 0:
            self._tail = (self._tail + 1) % self.buffer_size

    def __iter__(self):
        # iterate over full batches of buffer
        while self._head != self._tail:
            yield self._buffer[self._head]
            self._head = (self._head + 1) % self.buffer_size

