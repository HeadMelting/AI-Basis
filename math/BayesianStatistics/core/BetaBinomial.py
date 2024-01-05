from typing import Any
from .base import BaseDistribution
import sympy as sy


'''
nCk * B(x+a, N+b-x) / B(a,b)
'''


class BetaBinomialDistribution(BaseDistribution):

    def __init__(self):
        '''
        Symbols:
            - `N`
            - `x`
            - `a`
            - `b`


        ```
        Γ(N + 1)⋅Γ(a + b)⋅Γ(a + x)⋅Γ(N + b - x)   
        ────────────────────────────────────────────
        Γ(a)⋅Γ(b)⋅Γ(x + 1)⋅Γ(N + a + b)⋅Γ(N - x + 1)
        ```


        '''
        super().__init__()

    def __init_formula__(self, **symbols) -> Any:
        return sy.gamma(symbols['N']+1)/sy.gamma(symbols['x']+1)/sy.gamma(symbols['N']-symbols['x']+1) *\
            sy.gamma(symbols['x']+symbols['a'])*sy.gamma(symbols['N']+symbols['b']-symbols['x']) / sy.gamma(symbols['N']+symbols['a']+symbols['b']) *\
            sy.gamma(symbols['a']+symbols['b']) / \
            sy.gamma(symbols['a']) / sy.gamma(symbols['b'])

    def get_value(self, x, N, a, b):
        return super().get_value(x=x, N=N, a=a, b=b)


if __name__ == '__main__':
    beta_bin = BetaBinomialDistribution()
    beta_bin.__print_formula__()
    beta_bin.scatter(0, 10, x_symbol='x', N=10, a=2, b=4)
