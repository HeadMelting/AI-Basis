from torch.optim import Optimizer, required

#
class CustomOptim(Optimizer):
    def __init__(self,params, lr ):
        super(SGD)