from starter import *
import time

img_rows = 28
img_cols = 28
img_depth = 1
n_features = 784
n_classes = 10

def convolution_layer(input, n_channels, filter_size, n_filters, name):
    weights = tf.get_variable(
        "weight_{}".format(name),
        shape=[filter_size, filter_size, n_channels, n_filters],
        initializer=tf.contrib.layers.xavier_initializer()
    )
    biases = tf.get_variable(
        "bias_{}".format(name),
        shape=[n_filters],
        initializer=tf.constant_initializer(0.0)
    )
    layer = tf.nn.conv2d(
        input=input,
        filter=weights,
        strides=[1, 1, 1, 1],
        padding="SAME"
    )

    layer = layer + biases
    return layer, weights, biases

def relu_layer(input):
    layer = tf.nn.relu(input)
    return layer

def batch_normalization_layer(input):
    batch_mean, batch_var = tf.nn.moments(input, axes=[0, 1, 2])
    scale = tf.Variable(tf.ones([32]))
    beta = tf.Variable(tf.zeros([32]))

    layer = tf.nn.batch_normalization(input, batch_mean, batch_var,
        offset=beta,
        scale=scale,
        variance_epsilon=1e-3
    )
    return layer

def pooling_layer(input, kernel_size, stride_size):
    layer = tf.nn.max_pool(
        value=input,
        ksize=[1, kernel_size, kernel_size, 1],
        strides=[1, stride_size, stride_size, 1],
        padding="SAME"
    )
    return layer

def flatten_layer(input):
    layer = tf.contrib.layers.flatten(input)
    return layer

def fully_connected_layer(input, n_inputs, n_outputs, name):
    weights = tf.get_variable(
        "weight_{}".format(name),
        shape=[n_inputs, n_outputs],
        initializer=tf.contrib.layers.xavier_initializer()
    )
    biases = tf.get_variable(
        "bias_{}".format(name),
        shape=[n_outputs],
        initializer=tf.constant_initializer(0.0)
    )

    layer = tf.matmul(input, weights) + biases
    return layer, weights, biases


def build_CNN(learning_rate, weight_decay):
    # Define placeholders for input
    X = tf.placeholder(tf.float32, shape=[None, n_features], name="X")
    Y = tf.placeholder(tf.float32, shape=[None, n_classes], name="Y")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    # INPUT LAYER
    input_layer = tf.reshape(X, shape=[-1, img_rows, img_cols, img_depth])

    # CONVOLUTION LAYER
    conv_layer_1, conv_weights_1, conv_biases_1 = convolution_layer(
        input=input_layer,
        n_channels=1,
        filter_size=3,
        n_filters=32,
        name="conv_1"
    )

    # RELU ACTIVATION
    relu_layer_2 = relu_layer(conv_layer_1)

    # BATCH NORMALIZATION LAYER
    batch_norm_layer_3 = batch_normalization_layer(relu_layer_2)

    # MAX POOLING LAYER (2X2)
    max_pool_layer_4 = pooling_layer(batch_norm_layer_3, kernel_size=2, stride_size=2)

    # FLATTEN LAYER
    flatten_layer_5 = flatten_layer(max_pool_layer_4)

    # FULLY CONNECTED LAYER
    full_connect_layer_6, fc_weights_6, fc_biases_6 = fully_connected_layer(
        input=flatten_layer_5,
        n_inputs=6272, # calculated dimension of flattened_layer_5 = 14x14x32
        n_outputs=784,
        name="fc_layer_1"
    )

    # APPLY DROPOUT
    drop_out = tf.nn.dropout(full_connect_layer_6, keep_prob=keep_prob)

    # RELU ACTIVATION
    relu_layer_7 = relu_layer(drop_out)

    # FULLY CONNECTED LAYER
    full_connect_layer_8, fc_weights_8, fc_biases_8 = fully_connected_layer(
        input=relu_layer_7,
        n_inputs=784,
        n_outputs=10,
        name="fc_layer_2"
    )

    # SOFTMAX OUTPUT
    Y_pred = tf.nn.softmax(full_connect_layer_8)

    # CROSS ENTROPY LOSS
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=Y,
        logits=full_connect_layer_8
    )
    # REGULARIZATION
    regularizer = tf.nn.l2_loss(conv_weights_1) \
                + tf.nn.l2_loss(fc_weights_6) \
                + tf.nn.l2_loss(fc_weights_8)
    # TOTAL LOSS
    loss = tf.reduce_mean(cross_entropy) + weight_decay * regularizer

    # CLASSIFICATION ACCURACY
    Y_class = tf.argmax(Y, axis=1)
    Y_pred_class = tf.argmax(Y_pred, axis=1)
    correct_predictions = tf.equal(Y_class, Y_pred_class)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    # ADAM OPTIMIZER
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    return optimizer, X, Y, keep_prob, loss, accuracy



def train_CNN(X_data, Y_data, batch_size, n_epochs, learning_rate, weight_decay, p):
    n_datapoints = X_data.shape[0]
    iter_per_epoch = int(np.ceil(n_datapoints/batch_size))
    iterations = iter_per_epoch * n_epochs

    # Create Loss/Accuracy dictionaries to store performance data
    loss_curves = {'train': [], 'valid': [], 'test': []}
    accuracy_curves = {'train': [], 'valid': [], 'test': []}

    # SET UP COMPUTATIONAL GRAPH
    optimizer, X, Y, keep_prob, loss, accuracy = build_CNN(learning_rate, weight_decay)
    global_init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(global_init)
        for iter in range(iterations):
            # NEW EPOCH
            if iter % iter_per_epoch == 0:
                print("EPOCH {}".format(int((iter+1)/iter_per_epoch)))

                # CALCULATE TRAINING LOSS & ACCURACY
                feed_dict_train = {X: X_data, Y: Y_data, keep_prob: 1.0}
                _loss, _acc = sess.run([loss, accuracy], feed_dict=feed_dict_train)
                loss_curves['train'].append(_loss)
                accuracy_curves['train'].append(_acc)

                # CALCULATE VALIDATION LOSS & ACCURACY
                feed_dict_valid = {X: X_valid, Y: Y_valid, keep_prob: 1.0}
                _loss, _acc = sess.run([loss, accuracy], feed_dict=feed_dict_valid)
                loss_curves['valid'].append(_loss)
                accuracy_curves['valid'].append(_acc)

                # CALCULATE TEST LOSS & ACCURACY
                feed_dict_test = {X: X_test, Y: Y_test, keep_prob: 1.0}
                _loss, _acc = sess.run([loss, accuracy], feed_dict=feed_dict_test)
                loss_curves['test'].append(_loss)
                accuracy_curves['test'].append(_acc)

                # SHUFFLE DATA ON NEW EPOCH
                X_data, Y_data = shuffle(X_data, Y_data)

            # SELECT MINI-BATCH
            X_batch, Y_batch = X_data[:batch_size], Y_data[:batch_size]

            # GRADIENT DESCENT STEP on mini-batch
            feed_dict_batch = {X: X_batch, Y: Y_batch, keep_prob: p}
            sess.run([optimizer], feed_dict=feed_dict_batch)

            # SHIFT DATA BY BATCH SIZE so next sample is new
            X_data = np.roll(X_data, batch_size, axis=0)
            Y_data = np.roll(Y_data, batch_size, axis=0)

    return loss_curves, accuracy_curves


def main():
    start_time = time.time()
    loss, accuracy = train_CNN(
        X_train,
        Y_train,
        batch_size=32,
        n_epochs=50,
        learning_rate=1e-4,
        weight_decay=0,
        p=1.0
    )
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
    plt.ylabel('CE Loss')
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
