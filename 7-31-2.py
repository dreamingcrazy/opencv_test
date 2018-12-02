from tensorflow.examples.tutorials.mnist import input_data
# 下载手写数字

import tensorflow as tf
tf.nn.conv2d()
MODLE_EXIST = True

mnist = input_data.read_data_sets('./data1/mnist/', one_hot=True)
# print(mnist)

# 获取特征数据
fuature = mnist.train.images
# 获取目标数据
labels = mnist.train.labels
#
# print(fuature)
# print(labels)

# 通过人工神经网络算法进行预测
with tf.variable_scope('data'):
    # 1、数据
    # 1.1特征值
    x = tf.placeholder(dtype=tf.float32, shape=[None, 28 * 28 * 1])
    # 1.2目标值
    y = tf.placeholder(dtype=tf.uint8, shape=[None, 10])

with tf.variable_scope('model'):
    # 2、模型
    # 2.1权重
    # [None,784][784,10]=[None,10]
    weights = tf.Variable(initial_value=tf.random_normal(shape=[784, 10], dtype=tf.float32), name='weights')
    # 2.2偏置值
    bais = tf.Variable(initial_value=0.0, name='bais')
    # 2.3进行预测
    y_predict = tf.matmul(x, weights) + bais

with tf.variable_scope('softmax-loss'):
    # 3、损失计算
    # 3.1计算损失
    # tf.nn.softmax()
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_predict)
    # labels = None, logits = None,
    # 3.2计算平均损失
    mean_loss = tf.reduce_mean(loss)

with tf.variable_scope('youhua'):
    # 4、通过梯度下降算法进行损失优化
    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(mean_loss)


init_op = tf.global_variables_initializer()
save = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init_op)
    if MODLE_EXIST:
        save.restore(sess=sess,save_path='./save_1/simple_nn')
        x_test, y_test = mnist.train.next_batch(100)
        y_predict1=sess.run(tf.matmul(x_test, weights) + bais)
    #     sess, save_path

    else:
        for i in range(10):
            x_train, y_train = mnist.train.next_batch(100)
            sess.run(train_op, feed_dict={x: x_train, y: y_train})
        save.save(sess=sess,save_path='./save_1/simple_nn')
        # sess,
        # save_path,
        # # global_step = None,
