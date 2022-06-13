
import numpy as np
import scipy
import math


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

    def pdf(self, x):
        return self.p(x)

    def cdf(self, x):
        return scipy.integrate.quad(self.p, -np.inf, x)[0]

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import random

    mu1 = 100
    sig1 = 1
    mu2 = 10
    sig2 = 1

    ratio = ratiodist(mu1, sig1, mu2, sig2)

    plotlow = scipy.stats.norm(mu1, sig1).ppf(0.001) / scipy.stats.norm(mu2, sig2).ppf(0.999)
    plothigh = scipy.stats.norm(mu1, sig1).ppf(0.999) / scipy.stats.norm(mu2, sig2).ppf(0.001)

    x = np.linspace(
        plotlow,
        plothigh,
        10000
        )

    ratiodist = np.zeros(shape=(1000000,))
    for i in range(len(ratiodist)):
        num1 = random.normalvariate(mu1, sig1)
        num2 = random.normalvariate(mu2, sig2)
        ratiodist[i] = num1/num2

    plt.hist(ratiodist, density=True, histtype='stepfilled', alpha=0.2, bins=100)
    plt.plot(x, ratio.pdf(x))
    plt.show()
