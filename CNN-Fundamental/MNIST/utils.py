from torchvision.datasets import MNIST as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

MNIST_SCALED_MEAN = (0.1307)
MNIST_SCALED_STD = (0.3081)

def get_train_dataloader():
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(MNIST_SCALED_MEAN, MNIST_SCALED_STD)
        ])

    # 28 X 28
    mnist_train = dsets(root='./data',
                          train=True,
                          transform=transform_train,
                          download=True)
    mnist_train_dataLoader = DataLoader(mnist_train, shuffle=True, batch_size=128,num_workers=2)
    
   
    if __name__ == '__main__':
         # dataset에서 불러온 data 확인
        print(mnist_train.data.float().mean()/255) #평균 확인
        print(len(mnist_train))
        print(mnist_train[1][0].size())
        a = mnist_train[1][0]
        a = to_pil_image(a)
        a.save('./image/1.png')

    return mnist_train_dataLoader



def get_test_dataloader():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_SCALED_MEAN, MNIST_SCALED_STD)
    ])

    mnist_test = dsets(root='./data',
                            train=False,
                            transform=transform_test,
                            download=True)
    mnist_test_dataLoader = DataLoader(mnist_test, shuffle=True,num_workers=2,batch_size=128)                       

    return mnist_test_dataLoader

if __name__ == '__main__':
    get_train_dataloader()