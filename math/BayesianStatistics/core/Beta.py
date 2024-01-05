import math
from typing import Any
import numpy as np
import sympy as sy
import matplotlib.pyplot as plt
from .base import BaseDistribution


class BetaDistribution(BaseDistribution):
    '''
    a - 1        b - 1
    θ     ⋅(1 - θ)     ⋅Γ(a + b)
    ────────────────────────────
            Γ(a)⋅Γ(b)
    '''

    def __init_formula__(self, **symbols) -> Any:
        return sy.gamma(symbols['a'] + symbols['b'])/sy.gamma(
            symbols['a'])/sy.gamma(symbols['b'])*symbols['theta']**(
                symbols['a'] - 1)*(1 - symbols['theta'])**(symbols['b'] - 1)

    def get_value(self, theta, a, b):
        return super().get_value(theta=theta, a=a, b=b)

    def inference(self):
        return super().inference()


if __name__ == '__main__':
    b = BetaDistribution()
    print(b.get_value(0.5, 10, 2))
    b.plot(x_symbol='theta', a=16, b=8)
    b.integrate(('theta', 0, 1), a=10, b=5)
    print(b)
