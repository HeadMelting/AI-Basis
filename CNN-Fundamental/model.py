from torch import nn


class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
    
    def forward(self,x):
        x = self.fc(x)
        return x
    
