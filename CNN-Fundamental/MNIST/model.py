from torch import nn



class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels= 32, kernel_size=3, stride=2)
        self.linear = nn.Linear(in_features=1152, out_features=10,bias=True)



    def forward(self,input):

        output = self.conv1(input)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.relu(output)
        output = output.view(output.size(0),-1)
        output = self.linear(output)
        

        return output

layout = [[64, 128 , "MaxPool", 256, 512, "MaxPool"]]

def makeConv():
        layers = []
        input_channel = 3
        for output_channel in layout:
            if output_channel == "MaxPool":
                layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
                continue
            
            layers += [nn.Conv2d(input_channel,output_channel,kernel_size=3,padding=1)]
            layers += [nn.BatchNorm2d(output_channel)]

            layers += [nn.ReLU(inplace=True)]

            input_channel = output_channel

        return layers

class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.fullyLayer = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 10)
        )

       

        self.convLayer = makeConv()
   
    def forward(self,input):
        output = self.convLayer(input)
        output = output.view(output.size()[0], -1)
        output = self.fullyLayer(output)

        return output

    

    