# Author: wujiahang
import math
class OneEuroFilter:
    def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.freq=freq; self.min_cutoff=min_cutoff; self.beta=beta; self.d_cutoff=d_cutoff
        self.x_prev=None; self.dx_prev=0.0
    def alpha(self, cutoff):
        import math
        tau = 1.0/(2*math.pi*cutoff); te = 1.0/self.freq
        return 1.0/(1.0 + tau/te)
    def __call__(self, x):
        if self.x_prev is None: self.x_prev=x; return x
        dx=(x-self.x_prev)*self.freq
        a_d=self.alpha(self.d_cutoff)
        dx_hat=a_d*dx+(1-a_d)*self.dx_prev
        cutoff=self.min_cutoff + self.beta*abs(dx_hat)
        a=self.alpha(cutoff)
        x_hat=a*x+(1-a)*self.x_prev
        self.x_prev, self.dx_prev = x_hat, dx_hat
        return x_hat
