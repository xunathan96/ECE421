from starter_kmeans import *
import time

# K: number of clusters
def build_graph(K, learning_rate):
    tf.set_random_seed(421)

    # DEFINE INPUT PLACEHOLDERS
    X = tf.placeholder(tf.float32, shape=[None, dim], name="X")

    # DEFINE VARIABLES TO LEARN
    MU = tf.get_variable('MU',
        shape=[K, dim],
        initializer=tf.initializers.random_normal(mean=0, stddev=1))

    # DEFINE LOSS FUNCTIONS
    loss = calculate_loss(X, MU)

    # DETERMINE CLUSTER ASSIGNMENTS
    s = cluster_assignments(X, MU)

    # INITIALIZE GRADIENT DESCENT OPTIMIZER 
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=0.9,
        beta2=0.99,
        epsilon=1e-5
    ).minimize(loss)

    return optimizer, X, MU, s, loss


def train_clusters(K, learning_rate, n_epochs):
    optimizer, X, MU, s, loss = build_graph(K, learning_rate)
    global_init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(global_init)

        loss_curves = {'train': [], 'valid': []}
        cluster_assignments = {}

        for iter in range(n_epochs):
            # GRADIENT DESCENT STEP on data set
            feed_dict_batch = {X: data}
            [_opt, _loss] = sess.run([optimizer, loss], feed_dict=feed_dict_batch)
            loss_curves['train'].append(_loss)

            # GET VALIDATION LOSS
            if is_valid:
                feed_dict_batch = {X: val_data}
                [_loss] = sess.run([loss], feed_dict=feed_dict_batch)
                loss_curves['valid'].append(_loss)

        # GET CLUSTER ASSIGNMENTS
        feed_dict_batch = {X: data}
        [cluster_assignments['train']] = sess.run([s], feed_dict=feed_dict_batch)
        if is_valid:
            feed_dict_batch = {X: val_data}
            [cluster_assignments['valid']] = sess.run([s], feed_dict=feed_dict_batch)

        # GET LEARNED K-MEANS CLUSTERS
        [MU] = sess.run([MU], feed_dict={})

    return MU, loss_curves, cluster_assignments


def main():
    start_time = time.time()
    K = 5
    MU, loss, cluster_assignments = train_clusters(
        K=K,
        learning_rate=0.01,
        n_epochs=1000
    )
    end_time = time.time()
    print("--- %s seconds ---" % (time.time() - start_time))

    # REPORT FINAL TRAINING AND VALIDATION LOSS
    print("Training Loss:", loss['train'][-1])
    type = 'train'
    if is_valid:
        print("Validation Loss:", loss['valid'][-1])
        type = 'valid'

    # CALCULATE CLUSTER DISTRIBUTIONS
    for cluster in range(K):
        print("Cluster {}:\n\t# Points: {}\n\tPercent of Points: {}".format(
            cluster,
            np.sum(cluster_assignments[type]==cluster),
            np.mean(cluster_assignments[type]==cluster))
        )
    print("MU:\n", MU)

    # PLOT LOSS CURVE
    plt.plot(loss[type], color='blue', label='training data' if not is_valid else 'validation data')
    plt.legend()
    plt.title('Loss Curve')
    plt.ylabel('Squared Distance Loss')
    plt.xlabel('Iteration')
    plt.show()

    # PLOT CLUSTER ASSIGNMENTS
    colors = ['red','green','blue','purple', 'orange']
    plt.scatter(
        data[:,0] if not is_valid else val_data[:,0],
        data[:,1] if not is_valid else val_data[:,1],
        s=0.5,
        c=cluster_assignments[type],
        cmap=matplotlib.colors.ListedColormap(colors)
    )
    plt.title('Cluster Assignments')
    plt.xlabel('x[0]')
    plt.ylabel('x[1]')
    plt.show()



if __name__ == '__main__':
    main()
