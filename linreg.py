import numpy as np
import tensorflow as tf

# Generate synthetic data
N = 100
w_true = 5
b_true = 2
noise_scale = .1
x_np = np.random.rand(N,1)
noise = np.random.normal(scale = 0.1, size = (N,1))
y_np = np.reshape(x_np * w_true + b_true + noise, (-1))

# Generate tensorflow graph
with tf.name_scope("placeholders"):
    x = tf.placeholder(tf.float32, (N,1))
    y = tf.placeholder(tf.float32, (N,))

with tf.name_scope("weights"):
    # x is a scalar, so W is a single learnable weight
    # (there is only one feature)
    W = tf.Variable(tf.random_normal((1,1)))
    b = tf.Variable(tf.random_normal((1,)))

with tf.name_scope("prediction"):
    y_pred = tf.matmul(x, W) + b
    # h(x) = x * W + b
    # x: (N,1) column vector; W: (1,1) vector (there is only one feature for x)

with tf.name_scope("loss"):
    l = tf.reduce_sum((y - y_pred) ** 2) # element-wise exponent
    # J(W) = (y - h(x))^2

# Training op
with tf.name_scope("optim"):
    train_op = tf.train.AdamOptimizer(.001).minimize(l)

with tf.name_scope("summaries"):
    tf.summary.scalar("loss", l)
    merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter('/tmp/lr-train', tf.get_default_graph())


n_steps = 1000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(n_steps):
        feed_dict = {x: x_np, y: y_np}
        _, summary, loss = sess.run([train_op, merged, l], feed_dict=feed_dict)
        print("step %d, loss: %f" % (i, loss))
        train_writer.add_summary(summary, i)
