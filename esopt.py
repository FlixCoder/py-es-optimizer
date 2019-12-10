import numpy as np
import pickle

"""
TODO:
- parallel?
- LR scheduling
"""


""" Rectified Adam optimizer """
#becomes SGD with beta2 = 0
class RAdam:
    """
    Attributes:
    lr : float - learning rate
    lam : float - weight decay coefficient
    beta1 : float - exponential moving average factor
    beta2 : float - exponential second moment average factor (squared gradient)
    eps : float - small epsilon to avoid divide by zero (fuzz factor)
    t : int - number of steps taken with this optimizer
    avggrad1 : np.array - (internal) first order moment (avg)
    avggrad2 : np.array - (internal) second order moment (squared avg)

    Methods:
    Constructor(lr, lambda, beta1, beta2, eps) -> object instance
    get_delta(params, gradient) -> delta update
    """

    #constructor
    def __init__(self, lr = 0.001, lam = 0, beta1 = 0.9, beta2 = 0.999, eps = 1e-8):
        self.lr = lr
        self.lam = lam
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.t = 0
        self.avggrad1 = None
        self.avggrad2 = None
    #}

    def checkReady(self):
        assert(self.lr > 0)
        assert(self.lam >= 0)
        assert(self.beta1 >= 0 and self.beta1 < 1)
        assert(self.beta2 >= 0 and self.beta2 < 1)
        assert(eps >= 0)
    #}

    #compute delta update from params and gradient
    def get_delta(self, params, grad):
        if self.avggrad1 is None or self.avggrad2 is None:
            self.avggrad1 = np.zeros(params.shape)
            self.avggrad2 = np.zeros(params.shape)
        #}

        self.t += 1
        beta1PT = np.power(self.beta1, self.t)
        beta2PT = np.power(self.beta2, self.t)
        smaINF = 2.0 / (1.0 - self.beta2) - 1.0
        smaT = smaINF - 2.0 * self.t * beta2PT / (1.0 - beta2PT)
        tmp = (smaT - 4) / (smaINF - 4) * (smaT - 2) / (smaINF - 2) * smaINF / smaT
        if tmp < 0: tmp = 0
        rT = np.sqrt(tmp)

        self.avggrad1 = self.beta1 * self.avggrad1 + (1 - self.beta1) * grad
        self.avggrad2 = self.beta2 * self.avggrad2 + (1 - self.beta2) * np.square(grad)
        
        delta = None
        if smaT > 5:
            lr_unbias12 = self.lr * np.sqrt(1 - beta2PT) / (1 - beta1PT)
            delta = lr_unbias12 * rT * self.avggrad1 / (np.sqrt(self.avggrad2) + self.eps)
        else:
            lr_unbias1 = self.lr / (1 - beta1PT)
            delta = lr_unbias1 * self.avggrad1
        #}
        delta -= self.lr * self.lam * params #weight decay

        return delta
    #}
#}

""" Adamax optimizer """
class Adamax:
    """
    Attributes:
    lr : float - learning rate
    lam : float - weight decay coefficient
    beta1 : float - exponential moving average factor
    beta2 : float - exponential second moment average factor (squared gradient)
    eps : float - small epsilon to avoid divide by zero (fuzz factor)
    t : int - number of steps taken with this optimizer
    avggrad1 : np.array - (internal) first order moment (avg)
    avggrad2 : np.array - (internal) second order moment (squared avg)

    Methods:
    Constructor(lr, lambda, beta1, beta2, eps) -> object instance
    get_delta(params, gradient) -> delta update
    """

    #constructor
    def __init__(self, lr = 0.002, lam = 0, beta1 = 0.9, beta2 = 0.999, eps = 1e-8):
        self.lr = lr
        self.lam = lam
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.t = 0
        self.avggrad1 = None
        self.avggrad2 = None
    #}

    def checkReady(self):
        assert(self.lr > 0)
        assert(self.lam >= 0)
        assert(self.beta1 >= 0 and self.beta1 < 1)
        assert(self.beta2 >= 0 and self.beta2 < 1)
        assert(eps >= 0)
    #}

    #compute delta update from params and gradient
    def get_delta(self, params, grad):
        if self.avggrad1 is None or self.avggrad2 is None:
            self.avggrad1 = np.zeros(params.shape)
            self.avggrad2 = np.zeros(params.shape)
        #}

        self.t += 1
        beta1PT = np.power(self.beta1, self.t)
        lr_unbias = self.lr / (1 - beta1PT)

        self.avggrad1 = self.beta1 * self.avggrad1 + (1 - self.beta1) * grad
        self.avggrad2 = np.maximum(self.beta2 * self.avggrad2, np.abs(grad))
        
        delta = lr_unbias * self.avggrad1 / (self.avggrad2 + self.eps)
        delta -= self.lr * self.lam * params #weight decay

        return delta
    #}
#}

""" Lookahead optimizer (on top of other optimizer) """
class Lookahead:
    """
    Attributes:
    subopt : optimizer-class - sub-optimizer
    k : int - number of steps between parameter synchronizations
    alpha : float - outer step size
    t : int - number of taken timesteps
    paramsave : np.array - (internal) temporary storage of model parameters for the k steps

    Methods:
    Constructor() -> object instance
    get_delta(params, gradient) -> delta update
    """

    def __init__(self, subopt = None, k = 5, alpha = 0.5):
        self.subopt = subopt
        self.k = k
        self.alpha = alpha

        self.t = 0
        self.paramsave = None
    #}

    def checkReady(self):
        assert(self.subopt != None)
        assert(self.alpha > 0)
        assert(self.k > 0)
    #}

    #compute delta update from params and gradient
    def get_delta(self, params, grad):
        if self.paramsave is None or self.t == 0:
            self.paramsave = params
        #}

        #inner update
        delta = self.subopt.get_delta(params, grad)
        self.t += 1

        #outer update
        if self.t % self.k == 0:
            diff = params + delta - self.paramsave
            new = self.paramsave + self.alpha * diff
            delta = new - params
            self.paramsave = new
        #}

        return delta
    #}
#}


#CLASS DEFINITION
""" Evolution-Strategy Optimizer class. Optimizing parameters in a black box fashion. """
class ESOpt:
    """
    Attributes:
    params : np.array - NumPy array with parameters to be optimized
    opt : optimizer-instance - ?
        must have method get_delta(params, gradient) returning parameter manipulation delta
    eval : fn - evaluator function to be called while optimization
        parameters: model parameters to test (np.array), iteration count (int)
        return: score, bigger = better (float)
    samples : int - number of double-sided samples to estimate the gradient
    std : float - standard deviation to generate random parameter perturbation
    evalTest : fn - evaluator function to be called after optimization to return current model score
        parameters: model parameters to test (np.array)
        return: score, bigger = better (float)
    steps : int - number of optimization steps already performed

    Methods:
    Constructor(self, params, opt, eval, samples, std, evalTest) -> object instance
    optimize(self, n) -> (score, gradnorm)
    optimize_ranked(self, n) -> (score, gradnorm)
    save(self, file)
    load(file) -> object
    """
    
    #SHARED ATTRIBUTES
    
    #METHODS
    #constructor, params are (most) class attributes
    def __init__(self, params = None, optimizer = None, evaluator = None, samples = 50, std = 0.02, evaluatorTest = None):
        self.params = params
        self.opt = optimizer
        self.eval = evaluator
        self.samples = samples
        self.std = std
        self.evalTest = evaluatorTest
        self.steps = 0
    #}

    #assure class attributes are set up properly for optimization
    def checkReady(self):
        assert(self.params is not None)
        assert(self.opt != None)
        assert(self.eval != None)
        assert(self.std > 0)
        assert(self.samples > 0)
        assert(callable(self.eval))
        if self.evalTest != None: assert(callable(self.evalTest))

        if type(self.params) is list: self.params = np.array(self.params, dtype = np.single)
    #}
    
    #optimize parameters for n steps
    #returns (score, gradient norm)
    def optimize(self, n):
        self.checkReady()
        grad = np.zeros(self.params.shape)
        for i in range(n):
            seed = np.random.randint(2000000000)
            grad = np.zeros(self.params.shape)
            for j in range(self.samples):
                np.random.seed(seed + j)
                eps = np.random.normal(0, self.std, size=self.params.shape)
                pPeps = self.params + eps
                pMeps = self.params - eps
                scoreP = self.eval(pPeps, self.steps)
                scoreM = self.eval(pMeps, self.steps)
                grad += eps * (scoreP - scoreM)
            #}
            grad /= (2 * self.samples * self.std)
            delta = self.opt.get_delta(self.params, grad)
            self.params += delta
            self.steps += 1
        #}
        if self.evalTest != None: score = self.evalTest(self.params)
        else: score = self.eval(self.params, 999999999)
        return (score, np.linalg.norm(grad))
    #}

    #optimize parameters for n steps with centered ranks
    #returns (score, gradient norm)
    def optimize_ranked(self, n):
        self.checkReady()
        grad = np.zeros(self.params.shape)
        for i in range(n):
            seed = np.random.randint(2000000000)
            grad = np.zeros(self.params.shape)
            scores = []
            for j in range(self.samples):
                np.random.seed(seed + j)
                eps = np.random.normal(0, self.std, size=self.params.shape)
                pPeps = self.params + eps
                pMeps = self.params - eps
                scoreP = self.eval(pPeps, self.steps)
                scoreM = self.eval(pMeps, self.steps)
                scores.append((j, True, scoreP))
                scores.append((j, False, scoreM))
            #}
            scores = sorted(scores, key = lambda x: x[2], ascending = True)
            for rank in range(len(scores)):
                centered_rank = rank / (self.samples - 0.5) - 1 #2 * (rank / (2 * self.samples - 1) - 0.5)
                j, pos, score = scores[rank]
                np.random.seed(seed + j)
                eps = np.random.normal(0, self.std, size=self.params.shape)
                if pos: grad += eps * centered_rank
                else: grad -= eps * centered_rank
            #}
            grad /= (2 * self.samples * self.std)
            delta = self.opt.get_delta(self.params, grad)
            self.params += delta
            self.steps += 1
        #}
        if self.evalTest != None: score = self.evalTest(self.params)
        else: score = self.eval(self.params, 999999999)
        return (score, np.linalg.norm(grad))
    #}
    
    #save the current optimizer to a file (without evaluator!)
    def save(self, file):
        with open(file, "wb") as f:
            data = [self.params, self.opt, self.samples, self.std, self.steps]
            pickle.dump(data, f)
        #}
    #}
    
    #load optimizer from file (without evaluator!)
    def load(file):
        obj = ESOpt()
        with open(file, "rb") as f:
            params, opt, samples, std, steps = pickle.load(f)
            obj.params = params
            obj.opt = opt
            obj.samples = samples
            obj.std = std
            obj.steps = steps
        #}
        return obj
    #}
#}


if __name__ == "__main__":
    #test
    opt = ESOpt([0], RAdam(), lambda x,y: 0)
    opt.optimize(1)
    opt.save("test.opt")
    opt = ESOpt.load("test.opt")
    opt.eval = lambda x,y: 0
    opt.optimize(1)
#}

