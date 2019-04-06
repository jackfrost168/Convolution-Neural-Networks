from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

inputs_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='inputs_')
y = tf.placeholder(tf.float32, [None, 10])
sess = tf.InteractiveSession()

def weight_variable(shape):
#这里是构建初始变量
  initial = tf.truncated_normal(shape, mean=0,stddev=0.1) #s生成正态分布的随机数
#创建变量
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#定义残差网络的identity_block块(输入和输出维度相同)
def identity_block(X_input, kernel_size, in_filter, out_filters, stage, block):
        """
        Implementation of the identity block as defined in Figure 3

        Arguments:
        X_input -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
        filter -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        training -- train or test

        Returns:
        X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
        """

        # defining name basis
        block_name = 'res' + str(stage) + block
        f1, f2, f3 = out_filters
        with tf.variable_scope(block_name):
            X_shortcut = X_input

            #first
            #W_conv1 = weight_variable([1, 1, in_filter, f1])
            #X = tf.nn.conv2d(X_input, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
            #b_conv1 = bias_variable([f1])
            #X = tf.nn.relu(X+ b_conv1)
            X = tf.layers.conv2d(X_input, f1, (1,1), padding='SAME', activation=tf.nn.relu)

            #second
            #W_conv2 = weight_variable([kernel_size, kernel_size, f1, f2])
            #X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
            #b_conv2 = bias_variable([f2])
            #X = tf.nn.relu(X+ b_conv2)
            X = tf.layers.conv2d(X, f2, (kernel_size, kernel_size), padding='SAME', activation=tf.nn.relu)

            #third

            #W_conv3 = weight_variable([1, 1, f2, f3])
            #X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
            #b_conv3 = bias_variable([f3])
            #X = tf.nn.relu(X+ b_conv3)
            X = tf.layers.conv2d(X, f3, (1,1), padding='SAME', activation=tf.nn.relu)
            #final step
            add = tf.add(X, X_shortcut)
            b_conv_fin = bias_variable([f3])
            add_result = tf.nn.relu(add+b_conv_fin)

        return add_result

#定义conv_block模块，由于该模块定义时输入和输出尺度不同，需要进行卷积操作来改变尺度，从而相加
def convolutional_block( X_input, kernel_size, in_filter,
                            out_filters, stage, block, stride=1):
        """
        Implementation of the convolutional block as defined in Figure 4

        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        training -- train or test
        stride -- Integer, specifying the stride to be used

        Returns:
        X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
        """

        # defining name basis
        block_name = 'res' + str(stage) + block
        with tf.variable_scope(block_name):
            f1, f2, f3 = out_filters

            x_shortcut = X_input
            #filter 1*1*f1
            #W_conv1 = weight_variable([1, 1, in_filter, f1])
            #X = tf.nn.conv2d(X_input, W_conv1,strides=[1, stride, stride, 1],padding='SAME')
            #b_conv1 = bias_variable([f1])
            #X = tf.nn.relu(X + b_conv1)
            X = tf.layers.conv2d(X_input, f1, (1,1), padding='SAME', activation=tf.nn.relu)

            #second 3*3*f2
            #W_conv2 =weight_variable([kernel_size, kernel_size, f1, f2])
            #X = tf.nn.conv2d(X, W_conv2, strides=[1,1,1,1], padding='SAME')
            #b_conv2 = bias_variable([f2])
            #X = tf.nn.relu(X+b_conv2)
            X = tf.layers.conv2d(X, f2, (kernel_size, kernel_size), padding='SAME', activation=tf.nn.relu)

            #third 1*1*f3
            #W_conv3 = weight_variable([1,1, f2,f3])
            #X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1,1], padding='SAME')
            #b_conv3 = bias_variable([f3])
            #X = tf.nn.relu(X+b_conv3)
            X = tf.layers.conv2d(X, f3, (1, 1), padding='SAME', activation=tf.nn.relu)
            #shortcut path
            W_shortcut =weight_variable([1, 1, in_filter, f3])
            x_shortcut = tf.nn.conv2d(x_shortcut, W_shortcut, strides=[1, stride, stride, 1], padding='VALID')

            #final
            add = tf.add(x_shortcut, X)
            #建立最后融合的权重
            b_conv_fin = bias_variable([f3])
            add_result = tf.nn.relu(add + b_conv_fin)

        return add_result

x = tf.reshape(inputs_, (-1,28,28,1))

x1 = convolutional_block(x, 3, 1, [32, 32, 64], stage=1, block='a' )
pooling1 = tf.layers.max_pooling2d(x1, pool_size=(2, 2), strides=(2, 2), padding='SAME')
#这里操作后变成14x14x64

x2 = convolutional_block(X_input=pooling1, kernel_size=3, in_filter=64,  out_filters=[64, 64, 128], stage=2, block='b', stride=1)
pooling2 = tf.layers.max_pooling2d(x2, pool_size=(2, 2), strides=(2, 2), padding='SAME')
#上述conv_block操作后，尺寸变为7x7x128

x3 = identity_block(pooling2, 3, 128, [64, 64, 128], stage=2, block='c' )
pooling3 = tf.layers.max_pooling2d(x3, pool_size=(2, 2), strides=(2, 2), padding='SAME')
#上述操作后张量尺寸变成4*4*128

flat = tf.reshape(pooling3, [-1,4*4*128])   #将向量拉直

w_fc1 = weight_variable([4 * 4 * 128, 1024])
b_fc1 = bias_variable([1024])

h_fc1 = tf.nn.relu(tf.matmul(flat, w_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

#建立损失函数，在这里采用交叉熵函数
cross_entropy = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #cast将bool型转为float，求均值得到正确率
#初始化变量

sess.run(tf.global_variables_initializer())

print("start")
for i in range(2000):
    batch = mnist.train.next_batch(100)
    input = batch[0].reshape((-1,28,28,1))
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
        inputs_:input, y: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: input, y: batch[1], keep_prob: 0.5})








