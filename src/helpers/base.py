import torch
from abc import ABC, abstractmethod

class HFInference(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def inference(self, messages):
        return NotImplementedError
