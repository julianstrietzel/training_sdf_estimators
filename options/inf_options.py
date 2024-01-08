from argparse import Namespace

from options.base_options import BaseOptions


class InferenceOptions(BaseOptions):
    def __init__(self, opt_path):
        super(InferenceOptions, self).__init__()
        self.opt_path = opt_path
        self.opt = Namespace()
        self.initialized = False
        self.is_train = False
        self.parser = None

    def initialize(self):
        for line in open(self.opt_path, "r"):
            x = line.strip().split(":")
            if len(x) != 2:
                continue
            setattr(self.opt, x[0].strip(), convert(x[1].strip()))
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt.is_train = self.is_train
        return self.opt


def convert(txt):
    # check if txt is list
    if txt[0] == "[" and txt[-1] == "]":
        txt = txt[1:-1]
        txt = txt.split(",")
        txt = [convert(x) for x in txt]
        return txt
    try:
        k = float(txt)
        if k % 1 == 0:
            return int(k)
        return k

    except ValueError:
        return str(txt)
