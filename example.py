import numpy as np
import esopt


def main():
    params = np.array([0.0, 0.0, 0.0])
    optimizer = esopt.RAdam(0.5, beta1=0.1)
    opt = esopt.ESOpt(params, optimizer, evalTrain, samples = 25, std = 0.02)

    print("Beginning:")
    print("Initial Score: {}".format(evalTrain(params, 0)))
    for i in range(10):
        score, gradnorm = opt.optimize(10)
        print("{:2d}: {}".format(i+1, score))
    #}
    print("Finished!")
    print(opt.params)
#}

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

