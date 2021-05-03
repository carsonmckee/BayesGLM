class Distribution(object):

    __slots__ = ["ARG_VALUES", "link"]

    NAME = "Distribution"

    def __init__(self):
        self.ARG_VALUES = []
    
    def __str__(self):
        args = []
        for i in self.ARG_VALUES:
            if isinstance(i, str):
                args.append(i)
            elif isinstance(i, int) or isinstance(i, float):
                args.append(str(round(i, 2)))
            else:
                args.append(str(i.variable))
        out = self.NAME + "("
        for i in args:
            out += i + ", "
        out = out[:-2] + ")"
        return out
    
    def __eq__(self, other):
        a = isinstance(other, Distribution)
        b = (self.NAME == other.NAME)
        c = (self.ARG_VALUES == other.ARG_VALUES)
        if a and b and c:
            return True
        else:
            return False
    
    def __ne__(self, other):
        return not self.__eq__(other)


class Normal(Distribution):

    NAME = 'normal'

    def __init__(self, mu=0, sigma=1, link="identity"):
        super(Normal, self).__init__()
        self.ARG_VALUES = (mu, sigma)
        self.link = link

class Binomial(Distribution):

    NAME = 'binomial'

    def __init__(self, p = 0.5, link="logit"):
        super(Binomial, self).__init__()
        self.ARG_VALUES = [p]
        self.link = link

class Poisson(Distribution):

    NAME = 'poisson'

    def __init__(self, lamb = 0.5, link="log"):
        super(Poisson, self).__init__()
        self.ARG_VALUES = [lamb]
        self.link = link

class Gamma(Distribution):

    NAME = 'gamma'

    def __init__(self, rate=1, shape=1, link="inverse"):
        super(Gamma, self).__init__()
        self.ARG_VALUES = [rate, shape]
        self.link = link
    
class Laplace(Distribution):

    NAME = 'laplace'

    def __init__(self, loc=0, scale=1, link="identity"):
        super(Laplace, self).__init__()
        self.ARG_VALUES = [loc, scale]
        self.link = link

class Uninformed(Distribution):

    NAME = 'uninformed'

    def __init__(self):
        super(Uninformed, self).__init__()
        link = "identity"


class HalfCauchy(Distribution):

    NAME = 'halfcauchy'

    def __init__(self, loc, scale, link="identity"):
        super(HalfCauchy, self).__init__()
        self.ARG_VALUES = [loc, scale]
        self.link = link
