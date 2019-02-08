from starter import *

def buildGraph(X_data, y_data, learning_rate, lossType, ADAM, beta1=0.9, beta2=0.999, epsilon=1e-08):

    # INITIALIZE ---------------------------------------------------------------
    tf.set_random_seed(421)
    n_features = X_data.shape[1]

    # Define placeholders for input
    X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
    y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
    reg = tf.placeholder(tf.float32, name="lambda")

    # Define weights and biases
    w = tf.get_variable("w", (n_features, 1),
                        initializer=tf.truncated_normal_initializer(mean=0, stddev=0.5))
    b = tf.get_variable("b", (1, ),
                        initializer=tf.constant_initializer(0.0))

    # DEFINE LOSS FUNCTIONS ----------------------------------------------------
    if lossType=="MSE":
        y_pred = tf.add(tf.matmul(X, w), b, name="y_pred")
        mse = 0.5 * tf.reduce_mean(tf.square(y_pred - y))   # Mean Squared Error
        wd = 0.5 * reg * tf.reduce_sum(tf.square(w))        # Weight Decay
        loss = tf.add(mse, wd, name="loss")                 # Total Loss

    elif lossType=="CE":
        y_pred_linear = tf.add(tf.matmul(X, w), b)
        y_pred = tf.nn.sigmoid(y_pred_linear, name="y_pred")
        ce = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_pred_linear))
        wd = 0.5 * reg * tf.reduce_sum(tf.square(w))
        loss = tf.add(ce, wd, name="loss")

    # INITIALIZE GRADIENT DESCENT OPTIMIZER ------------------------------------
    if ADAM==False:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    else:
        optimizer = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon).minimize(loss)

    return X, w, b, y_pred, y, loss, optimizer, reg

def stochasticGradientDescent(batch_size, n_epochs, learning_rate, _lambda, lossType, ADAM, beta1=0.9, beta2=0.999, epsilon=1e-08):
    X_data, y_data = parseData(trainingData, trainingLabels)
    n_datapoints = X_data.shape[0]
    iter_per_epoch = int(np.ceil(n_datapoints/batch_size))
    iterations = iter_per_epoch * n_epochs

    X, w, b, y_pred, y, loss, optimizer, reg = buildGraph(X_data, y_data, learning_rate, lossType, ADAM, beta1, beta2, epsilon)
    global_init = tf.global_variables_initializer()
    train_loss, valid_loss, test_loss = [], [], []
    train_accuracy, valid_accuracy, test_accuracy = [], [], []

    with tf.Session() as sess:
        sess.run(global_init)
        for iter in range(iterations):
            # New Epoch Shuffle Data
            if (iter+1) % iter_per_epoch == 0:
                shuffle_index = np.random.choice(n_datapoints, n_datapoints, replace=False)
                X_data, y_data = X_data[shuffle_index], y_data[shuffle_index]

            # Select Mini-Batch
            X_batch, y_batch = X_data[:batch_size], y_data[:batch_size]

            # Gradient Descent Step on Mini-Batch
            _, _w, _b, _loss = sess.run([optimizer, w, b, loss],
                                feed_dict={X: X_batch, y: y_batch, reg:_lambda})

            # Shift data by batch size so next sample is new
            X_data = np.roll(X_data, batch_size, axis=0)
            y_data = np.roll(y_data, batch_size, axis=0)

            # After New Epoch Calculate the Loss and Accuracy
            if (iter+1) % iter_per_epoch == 0:
                print("EPOCH {}".format(int((iter+1)/iter_per_epoch)))

                # Measure Training, Validation, and Test Performance on ALL DATA (not just mini-batch)
                _train_loss, _valid_loss, _test_loss, \
                _train_accuracy, _valid_accuracy, _test_accuracy \
                    = measurePerformance(_w, _b, X_data, y_data, _lambda, lossType)
                train_loss.append(_train_loss); train_accuracy.append(_train_accuracy)
                valid_loss.append(_valid_loss); valid_accuracy.append(_valid_accuracy)
                test_loss.append(_test_loss); test_accuracy.append(_test_accuracy)

        w_optimal, b_optimal = sess.run([w, b])

    return w_optimal, b_optimal, \
            train_loss, train_accuracy, \
            valid_loss, valid_accuracy, \
            test_loss, test_accuracy
