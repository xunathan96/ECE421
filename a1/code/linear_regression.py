from GradientDescent_Numpy import *

def q3_LearningRate():
    # INITIALIZING DATA
    X, y = parseData(trainingData, trainingLabels)
    iterations = 5000
    reg = 0
    lossType = "MSE"

    alpha_list = [0.005, 0.001, 0.0001]
    for alpha in alpha_list:
        w, b, \
        train_loss, valid_loss, test_loss, \
        train_accuracy, valid_accuracy, test_accuracy \
            = grad_descent(X, y, alpha, iterations, reg, lossType)

        print("----------------------------------------------")
        print("Learning Rate: {}".format(alpha))
        print("Training Loss:  ", train_loss[-1])
        print("Validation Loss:", valid_loss[-1])
        print("Test Loss:      ", test_loss[-1])
        print("Training Accuracy:  ", train_accuracy[-1])
        print("Validation Accuracy:", valid_accuracy[-1])
        print("Test Accuracy:      ", test_accuracy[-1])

        plt.figure(0)
        plt.plot(train_loss, label='Learning Rate = {}'.format(alpha))
        plt.figure(1)
        plt.plot(valid_loss, label='Learning Rate = {}'.format(alpha))
        plt.figure(2)
        plt.plot(test_loss, label='Learning Rate = {}'.format(alpha))

    plt.figure(0)
    plt.legend()
    plt.title('Training Loss vs Learning Rate')
    plt.ylabel('Total Loss')
    plt.xlabel('Iteration')

    plt.figure(1)
    plt.legend()
    plt.title('Validation Loss vs Learning Rate')
    plt.ylabel('Total Loss')
    plt.xlabel('Iteration')

    plt.figure(2)
    plt.legend()
    plt.title('Test Loss vs Learning Rate')
    plt.ylabel('Total Loss')
    plt.xlabel('Iteration')

    plt.show()

def q4_Regularization():
    # INITIALIZING DATA
    X, y = parseData(trainingData, trainingLabels)
    iterations = 5000
    alpha = 0.005
    lossType = "MSE"

    reg_list = [0.001, 0.1, 0.5]
    for reg in reg_list:
        w, b, \
        train_loss, valid_loss, test_loss, \
        train_accuracy, valid_accuracy, test_accuracy \
            = grad_descent(X, y, alpha, iterations, reg, lossType)

        print("----------------------------------------------")
        print("Regularization Parameter: {}".format(reg))
        print("Training Loss:  ", train_loss[-1])
        print("Validation Loss:", valid_loss[-1])
        print("Test Loss:      ", test_loss[-1])
        print("Training Accuracy:  ", train_accuracy[-1])
        print("Validation Accuracy:", valid_accuracy[-1])
        print("Test Accuracy:      ", test_accuracy[-1])

        plt.figure(0)
        plt.plot(train_loss, label='Regularization = {}'.format(reg))
        plt.figure(1)
        plt.plot(valid_loss, label='Regularization = {}'.format(reg))
        plt.figure(2)
        plt.plot(test_loss, label='Regularization = {}'.format(reg))

    plt.figure(0)
    plt.legend()
    plt.title('Training Loss vs Regularization')
    plt.ylabel('Total Loss')
    plt.xlabel('Iteration')

    plt.figure(1)
    plt.legend()
    plt.title('Validation Loss vs Regularization')
    plt.ylabel('Total Loss')
    plt.xlabel('Iteration')

    plt.figure(2)
    plt.legend()
    plt.title('Test Loss vs Regularization')
    plt.ylabel('Total Loss')
    plt.xlabel('Iteration')

    plt.show()


def q5_NormalEquation():
    # INITIALIZING DATA
    X, y = parseData(trainingData, trainingLabels)
    iterations = 5000
    reg = 0
    alpha = 0.005
    lossType = "MSE"

    # NORMAL EQUATION ----------------------------------------------------------
    start = timer()
    w, b = normalEquation(X, y, reg)
    end = timer()

    time = end - start
    loss = MSE(w, b, X, y, reg)
    accuracy = classificationAccuracy(X, y, w, b)

    print("Training Loss:", loss)
    print("Training Accuracy:", accuracy)
    print("Time:", time)

    # GRADIENT DESCENT ---------------------------------------------------------
    start = timer()
    w, b, \
    train_loss, valid_loss, test_loss, \
    train_accuracy, valid_accuracy, test_accuracy \
        = grad_descent(X, y, alpha, iterations, reg, lossType)
    end = timer()

    time = end - start
    accuracy = classificationAccuracy(X, y, w, b)

    print("Training Loss:", train_loss[-1])
    print("Training Accuracy:", accuracy)
    print("Time:", time)


if __name__ == "__main__":
    print("Hello World")
    #q3_LearningRate()
    #q4_Regularization()
    q5_NormalEquation()
