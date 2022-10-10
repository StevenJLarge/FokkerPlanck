# This is a class-based implementation of a simulator object that can
# run simulations of particular FPE instances. Effectively this acts as
# a convenience wrapper around the raw fpe integraator obects
from abc import ABCMeta

class BaseSimulator(metaclass=ABCMeta):

    def __init__(self):
        pass


