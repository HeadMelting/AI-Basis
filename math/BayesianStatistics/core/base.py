import numpy as np
import matplotlib.pyplot as plt

import math
from typing import Any
from abc import abstractmethod
import sympy as sy


class BaseDistribution:
    '''
    Methods:
        - Plot Distribution
        - get Values
        - Inference (optional)
    '''

    def __init__(self):
        x, y, z, theta, p, N, k, a, b = sy.symbols(
            'x, y, z, theta, p, N, k, a, b')
        self.__formula__ = self.__init_formula__(
            x=x, y=y, z=z, theta=theta, p=p, N=N, k=k, a=a, b=b)

    def __print_formula__(self, **symbols):
        if self.__formula__ == None:
            raise NotImplementedError('__init_formula__ should be Implemented')
        new_formula = self.__formula__.subs(symbols)
        print(sy.pretty(new_formula))

    def get_value(self, **symbols):
        if self.__formula__ == None:
            raise NotImplementedError('__init_formula__ should be Implemented')

        return self.__formula__.subs(symbols).evalf()

    @abstractmethod
    def __init_formula__(self, **symbols) -> Any:
        '''
        return: sympy formula
        '''
        pass

    def inference(self):
        raise NotImplementedError

    def plot(self, start=0, end=1, x_symbol='theta', **symbols):
        self.__print_formula__(**symbols)

        x_axis = np.linspace(start, end, min(math.ceil((end-start)*100), 500))
        points = np.array(
            [[x, self.get_value(**symbols, **{x_symbol: x})] for x in x_axis])
        plt.plot(points[:, 0], points[:, 1])
        plt.show()

    def scatter(self, start=0, end=10, x_symbol='theta', **symbols):
        self.__print_formula__(**symbols)

        points = np.array(
            [[x+start, self.get_value(**symbols, **{x_symbol: x})] for x in range(end-start+1)])
        plt.scatter(points[:, 0], points[:, 1], marker='o')
        plt.show()

    def integrate(self, *tuple, **symbols):
        '''
        Args:
            - *tuple: (symbol, start, end) theta를 0부터 1까지 적분
            - **symbols: 기본 파라미터 적용 


        Example:
            ```python
            binomial = BinomialDistribution()
            binomial.integrate(('theta', 0, 1), N=10, k=4)
            ```
        '''
        if self.__formula__ == None:
            raise NotImplementedError('__init_formula__ should be Implemented')
        self.__print_formula__(**symbols)

        new_formula = self.__formula__.subs(symbols)
        I = sy.Integral(new_formula, *tuple)
        print(sy.pretty(I))
        print('\n= ', sy.integrate(new_formula, *tuple).evalf())

    def __str__(self):
        return sy.pretty(self.__formula__)
