import logging
import configparser


class Config:
    """
    Configuration for train/test/solve
    """
    log = logging.getLogger("Config")

    def __init__(self, file_name):
        self.data = configparser.ConfigParser()
        self.log.info("Reading config file %s", file_name)
        if not self.data.read(file_name):
            raise ValueError("Config file %s not found" % file_name)

    # sections acessors
    @property
    def sect_general(self):
        return self.data['general']

    @property
    def sect_train(self):
        return self.data['train']

    # general section
    @property
    def cube_type(self):
        return self.sect_general['cube_type']

    @property
    def run_name(self):
        return self.sect_general['run_name']

    # train section
    @property
    def train_scramble_depth(self):
        return self.sect_train.getint('scramble_depth')

    @property
    def train_cuda(self):
        return self.sect_train.getboolean('cuda', fallback=False)

    @property
    def train_learning_rate(self):
        return self.sect_train.getfloat('lr')

    @property
    def train_batch_size(self):
        return self.sect_train.getint('batch_size')

    @property
    def train_report_batches(self):
        return self.sect_train.getint('report_batches')

    @property
    def train_checkpoint_batches(self):
        return self.sect_train.getint('checkpoint_batches')

    @property
    def train_lr_decay_enabled(self):
        return self.sect_train.getboolean('lr_decay', fallback=False)

    @property
    def train_lr_decay_batches(self):
        return self.sect_train.getint('lr_decay_batches')

    @property
    def train_lr_decay_gamma(self):
        return self.sect_train.getfloat('lr_decay_gamma', fallback=1.0)

    @property
    def train_value_targets_method(self):
        return self.sect_train.get('value_targets_method', fallback='paper')

    @property
    def train_max_batches(self):
        return self.sect_train.getint('max_batches')

    @property
    def scramble_buffer_batches(self):
        return self.sect_train.getint("scramble_buffer_batches", 10)

    @property
    def push_scramble_buffer_iters(self):
        return self.sect_train.getint('push_scramble_buffer_iters', 100)

    @property
    def weight_samples(self):
        return self.sect_train.getboolean('weight_samples', True)

    # higher-level functions
    def train_name(self, suffix=None):
        res = "%s-%s-d%d" % (self.cube_type, self.run_name, self.train_scramble_depth)
        if suffix is not None:
            res += "-" + suffix
        return res
