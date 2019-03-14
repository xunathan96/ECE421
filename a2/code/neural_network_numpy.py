from starter import *
import time

def xavier_init(neurons_in, n_units, neurons_out):
    shape = (neurons_in, n_units)
    var = 2./(neurons_in + neurons_out)
    W = np.random.normal(0, np.sqrt(var), shape)
    return W

def init_weights(n_input, n_hidden, n_output):
    W = []
    W.append(None)
    W.append(xavier_init(n_input, n_hidden, n_output))
    W.append(xavier_init(n_hidden, n_output, 1))
    return W

def init_biases(n_input, n_hidden, n_output):
    b = []
    b.append(None)
    b.append(np.zeros((1, n_hidden)))
    b.append(np.zeros((1, n_output)))
    return b

def forward_propagation(X_input, W, b):
    X, S = [None]*3, [None]*3
    X[0] = X_input

    # UPDATE HIDDEN LAYER
    S[1] = X[0] @ W[1] + b[1]
    X[1] = relu(S[1])

    # UPDATE OUTPUT LAYER
    S[2] = X[1] @ W[2] + b[2]
    X[2] = softmax(S[2])

    return X, S

def backpropagation(X, S, W, Y):
    SENS = [None]*3

    # SEED SENSITIVITY
    N = Y.shape[0]
    SENS[2] = (1/N) * (X[2] - Y)

    # BACKPROPAGATION
    SENS[1] = (SENS[2] @ (W[2]).T) * derivative_relu(S[1])

    return SENS

def compute_gradients(X_input, Y, W, b):
    gradW = [0]*3
    gradb = [0]*3
    N = X_input.shape[0]

    # RUN PROPAGATIONS
    X, S = forward_propagation(X_input, W, b)
    SENS = backpropagation(X, S, W, Y)

    # GRADIENT of OUTPUT LAYER WEIGHTS + BIASES
    gradW[2] = (X[1]).T @ SENS[2]
    gradb[2] = np.sum(SENS[2], axis=0)

    # GRADIENT of HIDDEN LAYER WEIGHTS + BIASES
    gradW[1] = (X[0]).T @ SENS[1]
    gradb[1] = np.sum(SENS[1], axis=0)

    return gradW, gradb

def measure_performance(W, b):
    Y_pred, S = forward_propagation(X_train, W, b)
    train_loss = avgCE(Y_train, Y_pred[2])
    train_acc = accuracy(Y_train, Y_pred[2])

    Y_pred, S = forward_propagation(X_valid, W, b)
    valid_loss = avgCE(Y_valid, Y_pred[2])
    valid_acc = accuracy(Y_valid, Y_pred[2])

    Y_pred, S = forward_propagation(X_test, W, b)
    test_loss = avgCE(Y_test, Y_pred[2])
    test_acc = accuracy(Y_test, Y_pred[2])

    return train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc

def gradient_descent(X, Y, n_epochs, alpha, gamma, n_hidden_units):
    n_input_neurons = X.shape[1]
    n_output_neurons = Y.shape[1]

    # INITIALIZE WEIGHTS + BIASES
    W = init_weights(n_input_neurons, n_hidden_units, n_output_neurons)
    b = init_biases(n_input_neurons, n_hidden_units, n_output_neurons)
    VW_o = np.ones_like(W[2]) * 1e-5
    VW_h = np.ones_like(W[1]) * 1e-5
    Vb_o = np.ones_like(b[2]) * 1e-5
    Vb_h = np.ones_like(b[1]) * 1e-5

    # Create Loss/Accuracy dictionaries
    loss = {'train': [], 'valid': [], 'test': []}
    accuracy = {'train': [], 'valid': [], 'test': []}

    for t in range(n_epochs):
        gradW, gradb = compute_gradients(X, Y, W, b)
        print("EPOCH {}".format(t))
        # UPDATE OUTPUT LAYER
        VW_o = gamma * VW_o + alpha * gradW[2]
        W[2] = W[2] - VW_o
        Vb_o = gamma * Vb_o + alpha * gradb[2]
        b[2] = b[2] - Vb_o

        # UPDATE HIDDEN LAYER
        VW_h = gamma * VW_h + alpha * gradW[1]
        W[1] = W[1] - VW_h
        Vb_h = gamma * Vb_h + alpha * gradb[1]
        b[1] = b[1] - Vb_h

        # MEASURE PERFORMANCE
        train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc = measure_performance(W, b)
        loss['train'].append(train_loss)
        accuracy['train'].append(train_acc)
        loss['valid'].append(valid_loss)
        accuracy['valid'].append(valid_acc)
        loss['test'].append(test_loss)
        accuracy['test'].append(test_acc)

    return W, b, loss, accuracy

def main():
    start_time = time.time()
    W, b, loss, accuracy = gradient_descent(X_train, Y_train,
                                    n_epochs=200,
                                    alpha=0.005,
                                    gamma=0.9,
                                    n_hidden_units=100)
    end_time = time.time()
    print("--- %s seconds ---" % (time.time() - start_time))

    print("TRAINING ----------------------------")
    print("Loss:    ", loss['train'][-1])
    print("Accuracy:", accuracy['train'][-1])

    print("VALIDATION ----------------------------")
    print("Loss:    ", loss['valid'][-1])
    print("Accuracy:", accuracy['valid'][-1])

    print("TESTING ----------------------------")
    print("Loss:    ", loss['test'][-1])
    print("Accuracy:", accuracy['test'][-1])


    plt.plot(loss['train'], color='blue', label='training data')
    plt.plot(loss['valid'], color='red', label='validation data')
    plt.plot(loss['test'], color='green', label='test data')
    plt.legend()
    plt.title('Loss Curves')
    plt.ylabel('Average CE Loss')
    plt.xlabel('Epoch')
    plt.show()

    plt.plot(accuracy['train'], color='blue', label='training data')
    plt.plot(accuracy['valid'], color='red', label='validation data')
    plt.plot(accuracy['test'], color='green', label='test data')
    plt.legend()
    plt.title('Accuracy Curves')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.show()

if __name__ == '__main__':
    main()
