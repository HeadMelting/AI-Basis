from torch import nn,optim,save
from torchvision.datasets import MNIST as dsets
import torchvision.transforms as transforms
from utils import get_test_dataloader, get_train_dataloader
from model import Model, CustomCNN
import os
from datetime import datetime



def train(epoch):
    print('epoch #{}'.format(epoch))
    for batch_index,(images, labels) in enumerate(mnist_train_loader):
       


        labels = labels
        images = images
        optimizer.zero_grad()

        if batch_index % 100 == 0:
             print(model.conv1.weight[0,0,0,0])
             print('iter #{}'.format(batch_index))
             
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        if batch_index % 100 == 0:
             print(model.conv1.weight[0,0,0,0])

        

        


def eval(epoch=0):
    model.eval()

    test_loss = 0.0
    correct = 0.0

    for (images, labels) in mnist_test_loader:

        images = images
        labels = labels

        outputs = model(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
    
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        epoch,
        test_loss / len(mnist_test_loader.dataset),
        correct.float() / len(mnist_test_loader.dataset)
    ))
    


if __name__ == '__main__':

    checkpoint_path = os.path.join('checkpoint', 'hsk', datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss'))

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    # model = Model()
    model = Model()


    mnist_train_loader = get_train_dataloader()
    mnist_test_loader = get_test_dataloader()

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.01, weight_decay=5e-4)
    
    
    
    for epoch in range(1, 200 + 1):
        train(epoch)
        acc = eval(epoch)
    
    weights_path = checkpoint_path.format(net='model', epoch=epoch, type='regular')
    save(model.state_dict(), weights_path)

    
