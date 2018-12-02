# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
# def weights(shape):
#     return tf.Variable(initial_value=tf.random_normal(shape=shape))
# def baiss(shape):
#     return tf.Variable(initial_value=tf.constant(0.0,shape=shape))
# def model():
#     with tf.variable_scope('data'):
#         x = tf.placeholder(tf.float32,shape=[None,28*28*1])
#         y = tf.placeholder(tf.float32,shape=[None,10])
#     with tf.variable_scope('layer1'):
#         weight = weights(shape=(5,5,1,32))
#         bais = baiss(shape=[32])
#         x_reshape = tf.reshape(x,shape=[-1,28,28,1])
#         conv1 = tf.nn.conv2d(input=x_reshape,filter=weight,strides=[1,1,1,1],padding='SAME')+bais
#         x_relu = tf.nn.relu(conv1)
#         x_pool1 = tf.nn.max_pool(value=x_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#         with tf.variable_scope('quan'):
#             x_t = tf.reshape(x_pool1,shape=(-1,14*14*32))
#             w = weights(shape=(14*14*32,10))
#             b = baiss(shape=[10])
#             y_predict = tf.matmul(x_t,w)+b
#         return x,y,y_predict
# def loss_cop(y,y_predict):
#         loss = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_predict)
#         mean_loss = tf.reduce_mean(loss)
#         return mean_loss
# def sgd(meanloss,y,y_predict):
#     train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(meanloss)
#     equal_list = tf.equal(tf.argmax(y_predict,1),tf.argmax(y,1))
#     ac = tf.reduce_mean(tf.cast(equal_list,tf.float32))
#     return train_op,ac
# def main():
#     minist = input_data.read_data_sets('./data',one_hot=True)#导入数据，进行one_hot编码
#     x,y,y_predict = model()
#     loss = loss_cop(y,y_predict)
#     train_op,ac = sgd(loss,y,y_predict)
#     init_op = tf.global_variables_initializer()
#     with tf.Session() as sess:
#         sess.run(init_op)
#         for i in range(10000):
#             x_train,y_train = minist.train.next_batch(100)
#             sess.run(train_op,feed_dict={x:x_train,y:y_train})
#             ret = sess.run(ac, feed_dict={x: x_train, y: y_train})
#             print(ret)
# if __name__ == '__main__':
#     main()
import cv2


img = cv2.imread("pic.png")  # 读取图片
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  # 转换灰色

# OpenCV人脸识别分类器
classifier = cv2.CascadeClassifier(
    ".\haarcascades\haarcascade_frontalface_default.xml"
)
color = (0, 255, 0)  # 定义绘制颜色
# 调用识别人脸
faceRects = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
if len(faceRects):  # 大于0则检测到人脸
    for faceRect in faceRects:  # 单独框出每一张人脸
        x, y, w, h = faceRect
        # 框出人脸
        cv2.rectangle(img, (x, y), (x + h, y + w), color, 2)
        # 左眼
        cv2.circle(img, (x + w // 4, y + h // 4 + 30), min(w // 8, h // 8),
                   color)
        # 右眼
        cv2.circle(img, (x + 3 * w // 4, y + h // 4 + 30), min(w // 8, h // 8),
                   color)
        # 嘴巴
        cv2.rectangle(img, (x + 3 * w // 8, y + 3 * h // 4),
                      (x + 5 * w // 8, y + 7 * h // 8), color)

cv2.imshow("image", img)  # 显示图像
c = cv2.waitKey(10)

cv2.waitKey(0)