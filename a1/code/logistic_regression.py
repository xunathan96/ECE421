from GradientDescent_Numpy import *

def q2_Loss_and_Accuracy():
    # INITIALIZING DATA
    X, y = parseData(trainingData, trainingLabels)
    alpha = 0.005
    reg = 0.1
    iterations = 5000
    lossType = "CE"

    w, b, \
    train_loss, valid_loss, test_loss, \
    train_accuracy, valid_accuracy, test_accuracy \
        = grad_descent(X, y, alpha, iterations, reg, lossType)

    print("----------------------------------------------")
    print("Training Loss:  ", train_loss[-1])
    print("Validation Loss:", valid_loss[-1])
    print("Test Loss:      ", test_loss[-1])
    print("Training Accuracy:  ", train_accuracy[-1])
    print("Validation Accuracy:", valid_accuracy[-1])
    print("Test Accuracy:      ", test_accuracy[-1])

    plt.plot(train_loss, color='blue', label='Training Loss')
    plt.plot(valid_loss, color='green', label='Validation Loss')
    plt.plot(test_loss, color='red', label='Test Loss')
    plt.legend()
    plt.title('Logistic Regression Loss')
    plt.ylabel('Total Loss')
    plt.xlabel('Iteration')
    plt.show()

    plt.plot(train_accuracy, color='blue', label='Training Accuracy')
    plt.plot(valid_accuracy, color='green', label='Validation Accuracy')
    plt.plot(test_accuracy, color='red', label='Test Accuracy')
    plt.legend()
    plt.title('Logistic Regression Accuracy')
    plt.ylabel('Total Loss')
    plt.xlabel('Iteration')
    plt.show()


def q3_Logistic_vs_Linear_Regression():
    # INITIALIZING DATA
    X, y = parseData(trainingData, trainingLabels)
    alpha = 0.005
    reg = 0
    iterations = 5000

    lossType = "CE"
    w, b, \
    train_loss_ce, valid_loss, test_loss, \
    train_accuracy_ce, valid_accuracy, test_accuracy \
        = grad_descent(X, y, alpha, iterations, reg, lossType)

    lossType = "MSE"
    w, b, \
    train_loss_mse, valid_loss, test_loss, \
    train_accuracy_mse, valid_accuracy, test_accuracy \
        = grad_descent(X, y, alpha, iterations, reg, lossType)

    print("Linear Training Loss:", train_loss_mse[-1])
    print("Logistic Training Loss:", train_loss_ce[-1])

    print("Linear Training Accuracy:  ", train_accuracy_mse[-1])
    print("Logistic Training Accuracy:  ", train_accuracy_ce[-1])

    plt.plot(train_loss_mse, color='blue', label='Linear Regression Loss')
    plt.plot(train_loss_ce, color='green', label='Logistic Regression Loss')
    plt.legend()
    plt.title('Linear vs Logistic Regression')
    plt.ylabel('Total Loss')
    plt.xlabel('Iteration')
    plt.show()

    plt.plot(train_accuracy_mse, color='blue', label='Linear Regression Loss')
    plt.plot(train_accuracy_ce, color='green', label='Logistic Regression Loss')
    plt.legend()
    plt.title('Linear vs Logistic Regression')
    plt.ylabel('Total Loss')
    plt.xlabel('Iteration')
    plt.show()



if __name__ == "__main__":
    print("Hello World")
    #q2_Loss_and_Accuracy()
    q3_Logistic_vs_Linear_Regression()
