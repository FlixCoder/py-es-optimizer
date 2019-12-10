import numpy as np
from esopt_second_order import ESOpt


def main():
    params = np.array([0.0, 0.0, 0.0])
    opt = ESOpt(params, evalTrain, lr=0.5, samples = 25, std = 0.02)

    print("Beginning:")
    print("Initial Score: {}".format(evalTrain(params, 0)))
    for i in range(10):
        score = opt.optimize(10)
        print("{:2d}: {}".format(i+1, score))
    #}
    print("Finished!")
    print(opt.params)
#}

#evaluation of three parameters w0, w1, w2 to fit the function f(x) = x^2 = w0 + w1*x + w2*x^2
def evalTrain(params, t):
    X = np.array([0.0, 1.0, 2.0, -1, 4])
    Y = np.array([0.0, 1.0, 4.0, 1, 16])
    pred = params[0] + params[1] * X + params[2] * np.square(X)
    mse = np.square(Y - pred).sum() / len(X)
    return -mse
#}

if __name__ == "__main__":
    main()
#}

