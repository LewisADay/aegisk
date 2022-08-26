import numpy as np
import scipy
import math

class timefunc:
    def __init__(self, name):
        self.name = name

    def __call__(self, n=1):
        raise NotImplementedError


class halfnorm(timefunc):
    def __init__(self, scale=np.sqrt(np.pi / 2)):
        timefunc.__init__(self, "halfnorm")

        self.scale = scale

    def __call__(self, n=1):
        return np.abs(np.random.normal(scale=self.scale, size=n))


class pareto(timefunc):
    def __init__(self, power=5.0):
        timefunc.__init__(self, "pareto")

        self.power = power
        self.offset = (power - 1.0) / power

    def __call__(self, n=1):
        return self.offset * (1.0 + np.random.pareto(a=self.power, size=n))


class exponential(timefunc):
    def __init__(self, rate=1.0):
        timefunc.__init__(self, "exponential")

        self.rate = rate
        self.scale = 1.0 / rate

    def __call__(self, n=1):
        return np.random.exponential(scale=self.scale, size=n)

class consttime(timefunc):
    def __init__(self, prob, scale=100):
        timefunc.__init__(self, "consttime")
        
        self.scale = scale
        
    def __call__(self, x, n=1):
        x = np.array(x)
        res = []
        res.append(self.scale + math.sin(np.sum(x.ravel().flatten())))
        return np.array(res)
        
    
class gausstime(timefunc):
    def __init__(self, prob, scale=0.5, mean=0.5):
        timefunc.__init__(self, "gausstime")
        
        self.prob = prob
        
        if scale is None:
            scale = np.sqrt((prob.lb + prob.ub) / 2)
        
        self.scale = scale
        
        if mean is None:
            mean = (self.prob.lb+self.prob.ub)/2

        self.mean = mean
        
    def __call__(self, x, n=1):
        return scipy.stats.norm.pdf(x, self.mean, self.scale).flatten() + 1
    
class corrtime(timefunc):
    def __init__(self, prob, a=10, b=10, name=None):
        if name is None:
            timefunc.__init__(self, "corrtime")
        else:
            timefunc.__init__(self, name)
        
        self.prob = prob
        self.a = a
        self.b = b
        self.yopt = self.prob.yopt
    
    def __call__(self, x, n=1):
        return ((self.prob(x)-self.yopt) + self.b) * self.a
    
class negcorrtime(corrtime):
    def __init__(self, prob, a=100, b=10):
        corrtime.__init__(self, prob, a, "negcorrtime")
    
    def __call__(self, x, n=1):
        #return self.max - (self.prob(x)-self.yopt) * self.a
        return (1/(self.prob(x) - self.yopt + self.b)) * self.a
