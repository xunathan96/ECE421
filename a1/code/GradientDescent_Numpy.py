from starter import *

def gradMSE(w, b, X, y, reg):
    N = np.size(X, 0)   # number of rows in X
    ones = np.ones(N)
    gradMSE_W = (1/N) * X.T @ (X @ w + b - y) + reg*w
    gradMSE_B = (1/N) * ones.T @ (X @ w + b - y)
    return gradMSE_W, gradMSE_B


def gradCE(w, b, X, y, reg):
    N = np.size(X, 0)   # number of data points in X
    z = X @ w + b
    ones = np.ones(N)
    neg_sigmoid = 1./(1. + np.exp(z))

    dLoss_dz = (1/N) * ((1-y) - neg_sigmoid)
    dLoss_dw = X.T @ dLoss_dz + reg*w
    dLoss_db = ones.T @ dLoss_dz
    return dLoss_dw, dLoss_db

def grad_descent(X, y, alpha, iterations, reg, lossType, EPS=1e-7):
    # INITIALIZE WEIGHTS & BIAS
    w = np.random.normal(0, 0.1, (X.shape[1], y.shape[1]))
    b = np.random.uniform(-1, 1)
    train_loss, valid_loss, test_loss = [], [], []
    train_accuracy, valid_accuracy, test_accuracy = [], [], []

    for t in range(iterations):
        # CALCULATE LOSS
        if lossType=="MSE":
            E_in = MSE(w, b, X, y, reg)
        elif lossType=="CE":
            E_in = crossEntropyLoss(w, b, X, y, reg)

        # Measure Training, Validation, and Test Performance
        _train_loss, _valid_loss, _test_loss, \
        _train_accuracy, _valid_accuracy, _test_accuracy \
            = measurePerformance(w, b, X, y, reg, lossType)
        train_loss.append(_train_loss); valid_loss.append(_valid_loss); test_loss.append(_test_loss)
        train_accuracy.append(_train_accuracy); valid_accuracy.append(_valid_accuracy); test_accuracy.append(_test_accuracy)

        # UPDATE STEP
        if lossType=="MSE":
            gradMSE_W, gradMSE_B = gradMSE(w, b, X, y, reg)
        elif lossType=="CE":
            gradMSE_W, gradMSE_B = gradCE(w, b, X, y, reg)
        w = w - alpha * gradMSE_W                           # Update Weights
        b = b - alpha * gradMSE_B                           # Update Biases

        # STOPPING CONDITION
        if lossType=="MSE":
            E_in_new = MSE(w, b, X, y, reg)
        elif lossType=="CE":
            E_in_new = crossEntropyLoss(w, b, X, y, reg)
        if np.abs(E_in_new - E_in) < EPS:
            break

    return w, b, train_loss, valid_loss, test_loss, train_accuracy, valid_accuracy, test_accuracy
