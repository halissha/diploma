import data
from arg_parser import ArgumentParser
import mnist_v2



if __name__ == "__main__":
    parser = ArgumentParser()
    model = parser.get_model()
    attack = parser.get_attack()


