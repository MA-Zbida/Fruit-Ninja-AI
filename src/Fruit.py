import torch

class Fruit:
    def __init__(self, xywhn: torch.Tensor):
        self.xywhn = xywhn
        self.x = float(xywhn[0])
        self.y = float(xywhn[1])
        self.width = float(xywhn[2])
        self.height = float(xywhn[3])
