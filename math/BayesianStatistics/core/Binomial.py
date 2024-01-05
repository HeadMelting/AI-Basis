import numpy as np
import math
import sympy as sy
import matplotlib.pyplot as plt
from .base import BaseDistribution
import random


class BinomialDistribution(BaseDistribution):
    '''
     k        N - k   
    θ ⋅(1 - θ)     ⋅N!
    ──────────────────
    k!⋅(N - k)!    
    '''

    def __init_formula__(self, **symbol):
        return sy.factorial(symbol['N'])/sy.factorial(symbol['k']) / sy.factorial(
            symbol['N']-symbol['k'])*symbol['theta']**symbol['k']*(1-symbol['theta'])**(symbol['N']-symbol['k'])

    def get_value(self, theta, N, k):
        return super().get_value(theta=theta, N=N, k=k)

    def inference(self, max_iter=20000, N_trial=50):
        '''
        N_trial번 코인을 던져서, 몇번 앞면이 나오는지.
        '''
        num_heads = 0
        num_heads_arr = []

        for i in range(max_iter):
            isHead = random.randint(0, 1)
            if isHead:
                num_heads += 1

            if i % N_trial == 0 and i != 0:
                num_heads_arr.append(num_heads)
                num_heads = 0

        num_heads_arr = np.array(num_heads_arr, dtype=np.float64)

        k = np.arange(0, 50, 1)

        y = [math.comb(N_trial, k_)*math.pow(0.5, N_trial) for k_ in k]

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(num_heads_arr, bins=100, weights=np.ones_like(
            num_heads_arr) / len(num_heads_arr))
        plt.title(f'Coin Flip Experiment | N={N_trial}')
        plt.subplot(1, 2, 2)
        plt.plot(k, y)
        plt.title('Binomial Distribution PDF')
        plt.show()


if __name__ == '__main__':
    b = BinomialDistribution()
    print(b.get_value(0.5, 10, 2))
    b.plot(x_symbol='theta', N=16, k=8)
    b.integrate(('theta', 0, 1), N=10, k=5)
    print(b)
    b.inference()
