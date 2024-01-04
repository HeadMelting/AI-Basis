import numpy as np # 1.23.5
import pandas as pd # 1.5.3
from torch import nn,optim # 1.12.1
import torch
import os
from sklearn.preprocessing import MinMaxScaler

# for dirname, whatisthis, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname,filename))

# print(np.__version__)
# print(pd.__version__)
# print(torch.__version__)

data_url = 'http://lib.stat.cmu.edu/datasets/boston'
boston_raw = pd.read_csv(data_url, sep="\s+",skiprows=22, header=None)
boston = np.hstack([boston_raw.values[::2,:],boston_raw.values[1::2,:2]])
bostonColumn = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"]
bostonDF = pd.DataFrame(boston,columns=[bostonColumn])
bostonDF["PRICE"] = boston_raw.values[1::2,2]
# print(bostonDF.head())


def get_update_weights_value(bias,w1,w2,rm,lstat,target,learning_rate=0.01):
    """
    w1 : RM(방의 개수) 피처의 weight 값
    w2 : LSTAT(하위 계층 비율) 피처의 weight 값
    bias : Bias
    N : 입력 데이터 건수  
    """
    # 데이터 건수
    N = len(target)
    # 예측값
    predicted = w1 * rm + w2 *lstat + bias
    # 실제값과 예측값의 차이
    diff = target - predicted
    # bias를 array 기반으로 구하기 위해서 설정
    bias_factors = np.ones((N,))
    # Loss Function = MSE = 잔차의 제곱의 평균
    mse_loss = np.mean(np.square(diff))

    ## weight와 bias를 얼마나 Update할 것인지를 계산
    w1_update = -(2/N)*learning_rate*(np.dot(rm.T, diff)) # df.T = Transpose 전치
    w2_update = -(2/N)*learning_rate*(np.dot(lstat.T,diff))
    bias_update = -(2/N)*learning_rate*(np.dot(bias_factors.T,diff))

    return bias_update, w1_update, w2_update, mse_loss


def gradient_descent(features, target, iter_epochs = 1000, verbose=True):
    w1 = np.zeros((1,))
    w2 = np.zeros((1,))
    bias = np.zeros((1,))
    print('최초 w1,w2, bias:',w1,w2,bias)

    #learning_rate와 RM, LSTAT 피처 지정, 호출 시 numpy array형태로 RM과 LSTAT으로 된 2차원 feature가 입력됨
    learning_rate = 0.01
    rm = features[:,0]
    lstat = features[:,1]

    # iter_epochs 수만큼 반복하면서 update
    for i in range(iter_epochs):
        bias_update,w1_update, w2_update, loss = get_update_weights_value(bias,w1,w2,rm,lstat,target,learning_rate)

        w1 = w1 - w1_update
        w2 = w2 - w2_update
        bias = bias - bias_update
        if verbose and ((i+1) % 100 == 0):
            print('Epoch:',i+1,'/',iter_epochs)
            print("w1:",w1,'w2:',w2,'bias',bias,'loss:',loss)

    return w1,w2,bias



scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(bostonDF[["RM","LSTAT"]])

w1,w2,bias = gradient_descent(scaled_features,bostonDF["PRICE"].values.squeeze(), iter_epochs=1000,verbose=True)
print('#### 최종 w1, w2 ,bias #####')
print(w1, w2 ,bias)


predicted = scaled_features[:,0]*w1 + scaled_features[:,1]*w2 + bias
bostonDF["PREDICTED_PRICE"] = predicted
print(bostonDF.head(10))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(2,1)

        nn.init.zeros_(self.fc.weight)
        nn.init.ones_(self.fc.bias)

    def forward(self,x):
        x = self.fc(x)
        return x
    
model = Net()

optimizer = optim.Adam(model.parameters(),lr=0.05)
lossfunction = nn.MSELoss()

x_train = torch.tensor(scaled_features,dtype=torch.float32)
y_train = torch.tensor(bostonDF['PRICE'].values,dtype=torch.float32).view(-1,1)

for epoch in range(3000):
    y_pred = model(x_train)

    loss = lossfunction(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/3000], Loss: {loss.item():.4f}")