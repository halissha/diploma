from deepfool import deepfool
import argparse
import mnist_v2

class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--model', type=str,
                            help='Model type')
        self.parser.add_argument('--attack', type=str,
                            help='Attack type')
        self.parser.add_argument('--version', type=str,
                            help='Tensorflow version')
        self.args = self.get_args()

    def get_args(self):
        return self.parser.parse_args()

    def get_model(self):
        match self.args.model:
            case "mnist":
                return mnist_v2.get_model()

    def get_attack(self):
        match self.args.attack:
            case "deepfool":
                return

