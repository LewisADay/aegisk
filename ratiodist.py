
import numpy as np
import scipy
import math
from pynverse import inversefunc

class ratiodist():
    """ Class divising two gaussian distributions (the ratio distribution)
    The gaussians are assumed to be non-correlated
    """

    def __init__(self, mu1, sig1, mu2, sig2):
        self.mu1 = mu1
        self.sig1 = sig1
        self.var1 = sig1*sig1
        self.mu2 = mu2
        self.sig2 = sig2
        self.var2 = sig2*sig2
        self.phi = scipy.stats.norm.cdf

        a = lambda z: np.sqrt((1/self.var1)*(z**2) + (1/self.var2))
        b = lambda z: (self.mu1/self.var1)*z + (self.mu2/self.var2)
        c = (self.mu1**2)/self.var1 + (self.mu2**2)/self.var2
        d = lambda z: np.exp(((b(z)**2)-(c*(a(z)**2)))/(2*(a(z)**2)))

        lhs1 = lambda z: (b(z)*d(z))/(a(z)**3) * (1/(np.sqrt(2*math.pi)*self.sig1*sig2))
        lhs2 = lambda z: self.phi(b(z)/a(z)) - self.phi(-b(z)/a(z))
        rhs = lambda z: (1/((a(z)**2) * math.pi * self.sig1 *self.sig2))*np.exp(-c/2)

        self.p = lambda z: lhs1(z)*lhs2(z) + rhs(z)
        self.ppf = inversefunc(self.cdf)

    def pdf(self, x):
        return self.p(x)

    def cdf(self, x):
        return scipy.integrate.quad(self.p, -np.inf, x)[0]

    def boundaries(self):
        mean = self.mu1 / self.mu2
        low_bound = (self.mu1 - 2*self.sig1) / (self.mu2 + 2*self.sig2)
        high_bound = (self.mu1 + 2*self.sig1) / (self.mu2 - 2*self.sig2)
        return mean, low_bound, high_bound

def ratiodistapprox(mu1, var1, mu2, var2):
    mean = mu1 / mu2
    var = (mu1*mu1)/(mu2*mu2) * ((var1/(mu1*mu1)) + (var2/(mu2*mu2)))
    stddev = np.sqrt(var)
    return scipy.stats.norm(mean, stddev)

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import random
    from scipy import stats

    mu1 = 50
    sig1 = 10
    mu2 = 10
    sig2 = 1

    ratio = ratiodist(mu1, sig1, mu2, sig2)

    plotlow = stats.norm(mu1, sig1).ppf(0.001) / stats.norm(mu2, sig2).ppf(0.999)
    plothigh = stats.norm(mu1, sig1).ppf(0.999) / stats.norm(mu2, sig2).ppf(0.001)

    x = np.linspace(
        plotlow,
        plothigh,
        10000
        )

    approx = ratiodistapprox(mu1, sig1*sig1, mu2, sig2*sig2)

    plt.plot(x, approx.pdf(x))
    plt.plot(x, ratio.pdf(x))
    plt.show()
