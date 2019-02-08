from StochasticGradientDescent_Tensorflow import *
from GradientDescent_Numpy import *

def performance_SGD():
    w, b, \
    train_loss, train_accuracy, \
    valid_loss, valid_accuracy, \
    test_loss, test_accuracy \
        = stochasticGradientDescent(batch_size=500,
                                    n_epochs=700,
                                    learning_rate=0.001,
                                    _lambda=0,
                                    lossType="CE",
                                    ADAM=False)

    print("Training Loss:  ", train_loss[-1])
    print("Validation Loss:", valid_loss[-1])
    print("Test Loss:      ", test_loss[-1])
    print("Training Accuracy:  ", train_accuracy[-1])
    print("Validation Accuracy:", valid_accuracy[-1])
    print("Test Accuracy:      ", test_accuracy[-1])

    plt.plot(train_loss, color='blue', label='training data')
    plt.plot(valid_loss, color='red', label='validation data')
    plt.plot(test_loss, color='green', label='test data')
    plt.legend()
    plt.title('Training, Validation, Test Losses')
    plt.ylabel('Total Loss')
    plt.xlabel('Epoch')
    plt.show()

    plt.plot(train_accuracy, color='blue', label='training data')
    plt.plot(valid_accuracy, color='red', label='validation data')
    plt.plot(test_accuracy, color='green', label='test data')
    plt.legend()
    plt.title('Training, Validation, Test Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.show()

def batch_size_ADAM():
    # Batch sizes of 100, 700, 1750

    w, b, \
    train_loss, train_accuracy, \
    valid_loss, valid_accuracy, \
    test_loss, test_accuracy \
        = stochasticGradientDescent(batch_size=1750,
                                    n_epochs=700,
                                    learning_rate=0.001,
                                    _lambda=0,
                                    lossType="CE",
                                    ADAM=True)

    print("Training Loss:  ", train_loss[-1])
    print("Validation Loss:", valid_loss[-1])
    print("Test Loss:      ", test_loss[-1])
    print("Training Accuracy:  ", train_accuracy[-1])
    print("Validation Accuracy:", valid_accuracy[-1])
    print("Test Accuracy:      ", test_accuracy[-1])

    plt.plot(train_loss, color='blue', label='training data')
    plt.plot(valid_loss, color='red', label='validation data')
    plt.plot(test_loss, color='green', label='test data')
    plt.legend()
    plt.title('Training, Validation, Test Losses')
    plt.ylabel('Total Loss')
    plt.xlabel('Epoch')
    plt.show()

    plt.plot(train_accuracy, color='blue', label='training data')
    plt.plot(valid_accuracy, color='red', label='validation data')
    plt.plot(test_accuracy, color='green', label='test data')
    plt.legend()
    plt.title('Training, Validation, Test Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.show()

def hyperparameters_ADAM():
    # beta1 = {0.95, 0.99}
    # beta2 = {0.99, 0.9999}
    # epsilon = {1e-09, 1e-4}

    w, b, \
    train_loss, train_accuracy, \
    valid_loss, valid_accuracy, \
    test_loss, test_accuracy \
        = stochasticGradientDescent(batch_size=500,
                                    n_epochs=700,
                                    learning_rate=0.001,
                                    _lambda=0,
                                    lossType="CE",
                                    ADAM=True,
                                    beta2=0.9999)

    print("Training Loss:  ", train_loss[-1])
    print("Validation Loss:", valid_loss[-1])
    print("Test Loss:      ", test_loss[-1])
    print("Training Accuracy:  ", train_accuracy[-1])
    print("Validation Accuracy:", valid_accuracy[-1])
    print("Test Accuracy:      ", test_accuracy[-1])

def SGD_vs_BatchGD():
    w1, b1, \
    train_loss1, train_accuracy1, \
    valid_loss1, valid_accuracy1, \
    test_loss1, test_accuracy1 \
        = stochasticGradientDescent(batch_size=3500,
                                    n_epochs=5000,
                                    learning_rate=0.001,
                                    _lambda=0,
                                    lossType="MSE",
                                    ADAM=True)

    print("Stochastic Gradient Descent ---------------------------------")
    print("Training Loss:  ", train_loss1[-1])
    print("Validation Loss:", valid_loss1[-1])
    print("Test Loss:      ", test_loss1[-1])
    print("Training Accuracy:  ", train_accuracy1[-1])
    print("Validation Accuracy:", valid_accuracy1[-1])
    print("Test Accuracy:      ", test_accuracy1[-1])

    # INITIALIZING DATA
    X, y = parseData(trainingData, trainingLabels)
    iterations = 5000
    reg = 0
    alpha = 0.001
    lossType = "MSE"

    w2, b2, \
    train_loss2, valid_loss2, test_loss2, \
    train_accuracy2, valid_accuracy2, test_accuracy2 \
        = grad_descent(X, y, alpha, iterations, reg, lossType)

    print("Batch Gradient Descent ---------------------------------")
    print("Training Loss:  ", train_loss2[-1])
    print("Validation Loss:", valid_loss2[-1])
    print("Test Loss:      ", test_loss2[-1])
    print("Training Accuracy:  ", train_accuracy2[-1])
    print("Validation Accuracy:", valid_accuracy2[-1])
    print("Test Accuracy:      ", test_accuracy2[-1])


    plt.plot(train_loss1, color='blue', label='Stochastic GD')
    plt.plot(train_loss2, color='green', label='Batch GD')
    plt.legend()
    plt.title('Batch GD vs Stochastic GD')
    plt.ylabel('Total Loss')
    plt.xlabel('Epoch')
    plt.show()

    plt.plot(train_accuracy1, color='blue', label='Stochastic GD')
    plt.plot(train_accuracy2, color='green', label='Batch GD')
    plt.legend()
    plt.title('Batch GD vs Stochastic GD')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.show()


if __name__ == "__main__":
    print("Hello World")
    #performance_SGD()
    #batch_size_ADAM()
    hyperparameters_ADAM()
    #GD_vs_BatchGD()
