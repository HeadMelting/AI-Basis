import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data_url = 'http://lib.stat.cmu.edu/datasets/boston'
boston_raw = pd.read_csv(data_url, sep="\s+",skiprows=22, header=None)
boston = np.hstack([boston_raw.values[::2,:],boston_raw.values[1::2,:2]])
bostonColumn = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"]
bostonDF = pd.DataFrame(boston,columns=[bostonColumn])
bostonDF["PRICE"] = boston_raw.values[1::2,2]

# def get_update_weights_value_sgd(bias, w1, w2, rm_sgd, lstat_sgd, target_sgd, learning_rate=0.01):
#     N = target_sgd.shape[0]
#     predicted_sgd = w1 * rm_sgd + w2 * lstat_sgd + bias
#     diff_sgd = target_sgd - predicted_sgd
#     bias_factors = np.ones((N,))

#     w1_update = -(2/N)*learning_rate*(np.dot(rm_sgd.T, diff_sgd))
#     w2_update = -(2/N)*learning_rate*(np.dot(lstat_sgd.T, diff_sgd))
#     bias_update = -(2/N)*learning_rate*(np.dot(bias_factors.T, diff_sgd))

#     return bias_update, w1_update, w2_update


# def st_gradient_descent(features, target, iter_epochs=1000, verbose=True):
#     np.random.seed = 2021
#     w1 = np.zeros((1,))
#     w2 = np.zeros((1,))
#     bias = np.zeros((1,))
#     print('최초 w1, w2, bias:',w1,w2,bias)

#     learning_rate = 0.01
#     rm = features[:,0]
#     lstat = features[:,1]

#     for i in range(iter_epochs):
#         sgd_index = np.random.choice(target.shape[0], 1)
#         rm_sgd = rm[sgd_index]
#         lstat_sgd = lstat[sgd_index]
#         target_sgd = target[sgd_index]

#         bias_update, w1_update, w2_update = get_update_weights_value_sgd(bias, w1, w2, rm_sgd, lstat_sgd, target_sgd, learning_rate)
        
#         # SGD로 구한 weight/bias의 update 적용. 
#         w1 = w1 - w1_update
#         w2 = w2 - w2_update
#         bias = bias - bias_update
#         if verbose:
#             print('Epoch:', i+1,'/', iter_epochs)
#             # Loss는 전체 학습 데이터 기반으로 구해야 함.
#             predicted = w1 * rm + w2*lstat + bias
#             diff = target - predicted
#             mse_loss = np.mean(np.square(diff))
#             print('w1:', w1, 'w2:', w2, 'bias:', bias, 'loss:', mse_loss)


#     return w1, w2, bias


scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(bostonDF[['RM','LSTAT']])

# w1, w2, bias = st_gradient_descent(scaled_features, bostonDF['PRICE'].values, iter_epochs=5000, verbose=True)
# print('##### 최종 w1, w2, bias #######')
# print(w1, w2, bias)

# predicted = scaled_features[:, 0]*w1 + scaled_features[:, 1]*w2 + bias
# bostonDF['PREDICTED_PRICE_SGD'] = predicted
# bostonDF.head(10)

### MINI-BATCH

def get_update_weights_value_batch(bias, w1, w2, rm_batch, lstat_batch, target_batch, learning_rate=0.01):
    
    # 데이터 건수
    N = target_batch.shape[0]
    # 예측 값. 
    predicted_batch = w1 * rm_batch+ w2 * lstat_batch + bias
    # 실제값과 예측값의 차이 
    diff_batch = target_batch - predicted_batch
    # bias 를 array 기반으로 구하기 위해서 설정. 
    bias_factors = np.ones((N,))
    print(predicted_batch.shape,"diadsfasdfasdfadsfsdafasd")
    
    # weight와 bias를 얼마나 update할 것인지를 계산.  
    w1_update = -(2/N)*learning_rate*(np.dot(rm_batch.T, diff_batch))
    w2_update = -(2/N)*learning_rate*(np.dot(lstat_batch.T, diff_batch))
    bias_update = -(2/N)*learning_rate*(np.dot(bias_factors.T, diff_batch))
    
    # Mean Squared Error값을 계산. 
    #mse_loss = np.mean(np.square(diff))
    
    # weight와 bias가 update되어야 할 값 반환 
    return bias_update, w1_update, w2_update

# def batch_random_gradient_descent(features, target, iter_epochs=1000, batch_size=30, verbose=True):
#     # w1, w2는 numpy array 연산을 위해 1차원 array로 변환하되 초기 값은 0으로 설정
#     # bias도 1차원 array로 변환하되 초기 값은 1로 설정. 
#     np.random.seed = 2021
#     w1 = np.zeros((1,))
#     w2 = np.zeros((1,))
#     bias = np.zeros((1, ))
#     print('최초 w1, w2, bias:', w1, w2, bias)
    
#     # learning_rate와 RM, LSTAT 피처 지정. 호출 시 numpy array형태로 RM과 LSTAT으로 된 2차원 feature가 입력됨.
#     learning_rate = 0.01
#     rm = features[:, 0]
#     lstat = features[:, 1]
    
#     # iter_epochs 수만큼 반복하면서 weight와 bias update 수행. 
#     for i in range(iter_epochs):
#         # batch_size 갯수만큼 데이터를 임의로 선택. 
#         batch_indexes = np.random.choice(target.shape[0], batch_size)
#         rm_batch = rm[batch_indexes]
#         lstat_batch = lstat[batch_indexes]
#         target_batch = target[batch_indexes]
#         # Batch GD 기반으로 Weight/Bias의 Update를 구함. 
#         bias_update, w1_update, w2_update = get_update_weights_value_batch(bias, w1, w2, rm_batch, lstat_batch, target_batch, learning_rate)
        
#         # Batch GD로 구한 weight/bias의 update 적용. 
#         w1 = w1 - w1_update
#         w2 = w2 - w2_update
#         bias = bias - bias_update
#         if verbose:
#             print('Epoch:', i+1,'/', iter_epochs)
#             # Loss는 전체 학습 데이터 기반으로 구해야 함.
#             predicted = w1 * rm + w2*lstat + bias
#             diff = target - predicted
#             mse_loss = np.mean(np.square(diff))
#             print('w1:', w1, 'w2:', w2, 'bias:', bias, 'loss:', mse_loss)
        
#     return w1, w2, bias


# w1, w2, bias = batch_random_gradient_descent(scaled_features, bostonDF['PRICE'].values, iter_epochs=5000, batch_size=30, verbose=True)
# print('##### 최종 w1, w2, bias #######')
# print(w1, w2, bias)


# batch_gradient_descent()는 인자로 batch_size(배치 크기)를 입력 받음. 
def batch_gradient_descent(features, target, iter_epochs=1000, batch_size=30, verbose=True):
    # w1, w2는 numpy array 연산을 위해 1차원 array로 변환하되 초기 값은 0으로 설정
    # bias도 1차원 array로 변환하되 초기 값은 1로 설정. 
    np.random.seed = 2021
    w1 = np.zeros((1,))
    w2 = np.zeros((1,))
    bias = np.zeros((1, ))
    print('최초 w1, w2, bias:', w1, w2, bias)
    
    # learning_rate와 RM, LSTAT 피처 지정. 호출 시 numpy array형태로 RM과 LSTAT으로 된 2차원 feature가 입력됨.
    learning_rate = 0.01
    rm = features[:, 0]
    lstat = features[:, 1]

    print(rm.shape)
    
    # iter_epochs 수만큼 반복하면서 weight와 bias update 수행. 
    for i in range(iter_epochs):
        # batch_size 만큼 데이터를 가져와서 weight/bias update를 수행하는 로직을 전체 데이터 건수만큼 반복
        for batch_step in range(0, target.shape[0], batch_size):
            # batch_size만큼 순차적인 데이터를 가져옴. 
            rm_batch = rm[batch_step:batch_step + batch_size]
            lstat_batch = lstat[batch_step:batch_step + batch_size]
            target_batch = target[batch_step:batch_step + batch_size]
        
            bias_update, w1_update, w2_update = get_update_weights_value_batch(bias, w1, w2, rm_batch, lstat_batch, target_batch, learning_rate)

            # Batch GD로 구한 weight/bias의 update 적용. 
            w1 = w1 - w1_update
            w2 = w2 - w2_update
            bias = bias - bias_update
        
            print(w1_update.shape,"#####!@#!@#!@!@#!@")
            if verbose:
                print('Epoch:', i+1,'/', iter_epochs, 'batch step:', batch_step)
                # Loss는 전체 학습 데이터 기반으로 구해야 함.
                predicted = w1 * rm + w2 * lstat + bias
                diff = target - predicted
                mse_loss = np.mean(np.square(diff))
                print('w1:', w1, 'w2:', w2, 'bias:', bias, 'loss:', mse_loss)
        
    return w1, w2, bias

w1, w2, bias = batch_gradient_descent(scaled_features, bostonDF['PRICE'].values.squeeze(), iter_epochs=5000, batch_size=30, verbose=True)
print('##### 최종 w1, w2, bias #######')
print(w1, w2, bias)