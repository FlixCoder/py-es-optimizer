import numpy as np
import pickle

"""
TODO:
"""


""" Evolution-Strategy Optimizer class. Optimizing parameters in a black box fashion. """
class ESOpt:
    """
    Attributes:
    params : np.array - NumPy array with parameters to be optimized
    eval : fn - evaluator function to be called while optimization
        parameters: model parameters to test (np.array), iteration count (int)
        return: score, bigger = better (float)
    lr : float - step size / learning rate for update steps
    samples : int - number of double-sided samples to estimate the gradient
    std : float - standard deviation to generate random parameter perturbation
    evalTest : fn - evaluator function to be called after optimization to return current model score
        parameters: model parameters to test (np.array)
        return: score, bigger = better (float)

    steps : int - number of optimization steps already performed

    Methods:
    Constructor(self, params, eval, lr, samples, std, evalTest) -> object instance
    optimize(self, n) -> (score, gradnorm)
    save(self, file)
    load(file) -> object
    """
    
    #constructor, params are (most) class attributes
    def __init__(self, params = None, evaluator = None, lr = 1, samples = 50, std = 0.02, evaluatorTest = None):
        self.params = params
        self.eval = evaluator
        self.lr = lr
        self.samples = samples
        self.std = std
        self.evalTest = evaluatorTest
        self.steps = 0
    #}

    #assure class attributes are set up properly for optimization
    def checkReady(self):
        assert(self.params is not None)
        assert(self.eval != None)
        assert(self.lr > 0)
        assert(self.std > 0)
        assert(self.samples > 0)
        assert(callable(self.eval))
        if self.evalTest != None: assert(callable(self.evalTest))

        if type(self.params) is list: self.params = np.array(self.params, dtype = np.single)
    #}
    
    #optimize parameters for n steps using second order method
    #returns score on test evaluator if given, else training score with always the same steps parameters
    """
    exact algorithm:
    For n steps:
        - initialize vectors for gradient and second order derivatives (with zero)
        -> named grad and second
        - evaluate current score to be used in second order computation multiple times
        -> named curScore
        - for i = 0..samples, where samples is the number of double sided samples (hyperparameter, size of population)
            1. generate parameter modification vector eps and the corresponding new parameters pPeps and pMeps
            2. evaluate these parameters and save the score in scoreP and scoreM
            3. add first order difference to grad
            4. add second order difference to grad
        - calculate average by dividing by the number of samples
        - use the results to perform a Newton-like optimization step in the following way:
            1. use second as "approximation" of the Hessian, i.e. only the diagonal values were calculated
            2. the inversion of the diagonal hessian is all its elements inverted, stored in a vector
            3. to avoid saddle-points, convert second to the absolute values
            4. make sure, second is not too close to zero (divide by min(second) and scale by tanh(thresh+max(grad)))
            5. the update step delta now computes as element-wise delta = grad / second
            6. here for ascend: new params = old params + lr * delta, so update step is dampened by step size lr
    """
    def optimize(self, n):
        self.checkReady()
        for i in range(n):
            seed = np.random.randint(2000000000) #for later purposes: regenerating the same eps vectors
            grad = np.zeros(self.params.shape)
            second = np.zeros(self.params.shape)
            curScore = self.eval(self.params, self.steps)
            for j in range(self.samples):
                np.random.seed(seed + j) #seed for possible regeneration of eps vectors
                #generate eps and the corresponding new parameters to test
                eps = np.random.normal(0, self.std, size=self.params.shape)
                pPeps = self.params + eps
                pMeps = self.params - eps
                #evaluate the parameters to receive their score
                scoreP = self.eval(pPeps, self.steps)
                scoreM = self.eval(pMeps, self.steps)
                #use the score to calculate the first and second order gradient/derivative, eps is direction (with step size?)
                grad += eps * (scoreP - scoreM) #use (f(x+h) - f(x-h)) / 2h, division after loop
                second += eps * (scoreP - 2*curScore + scoreM) #use (f(x-h) - 2f(x) + f(x+h)) / h^2, division after loop
            #}
            #take average of the first/second order vectors
            grad /= (self.samples * 2 * self.std) #divide by 2h also (on average)
            second /= (self.samples * self.std * self.std) #divide by h^2 also (on average)
            #compute delta update using a modified Newton's optimizer
            second = np.abs(second)
            second = np.tanh(0.5 + np.max(np.abs(grad))) * second / np.min(second)
            delta = np.divide(grad, second)
            #take parameter step
            self.params += self.lr * delta
            self.steps += 1
        #}
        if self.evalTest != None: score = self.evalTest(self.params)
        else: score = self.eval(self.params, 999999999)
        return score
    #}

    #save the current optimizer to a file (without evaluator!)
    def save(self, file):
        with open(file, "wb") as f:
            data = [self.params, self.samples, self.std, self.steps]
            pickle.dump(data, f)
        #}
    #}
    
    #load optimizer from file (without evaluator!)
    def load(file):
        obj = ESOpt()
        with open(file, "rb") as f:
            params, samples, std, steps = pickle.load(f)
            obj.params = params
            obj.samples = samples
            obj.std = std
            obj.steps = steps
        #}
        return obj
    #}
#}


if __name__ == "__main__":
    #test
    opt = ESOpt([0], lambda x,y: 0)
    opt.optimize(1)
    opt.save("test.opt")
    opt = ESOpt.load("test.opt")
    opt.eval = lambda x,y: 0
    opt.optimize(1)
#}

