from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)   #导入自带MNIST数据集
print("start")

inputs_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='inputs_')
y = tf.placeholder(tf.float32, [None, 10])
sess = tf.InteractiveSession()

def weight_variable(shape):
#这里是构建初始变量
  initial = tf.truncated_normal(shape, mean=0,stddev=0.1)
#创建变量
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

x = tf.reshape(inputs_, (-1,28,28,1))   #x为-1*28*28*1

conv1 = tf.layers.conv2d(x, 32, (3,3), padding='same', activation=tf.nn.relu)
pooling1 = tf.layers.max_pooling2d(conv1, (2,2), (2,2), padding='same')
#conv1变为14*14*32
conv2 = tf.layers.conv2d(pooling1, 64, (3,3), padding='same', activation=tf.nn.relu)
pooling2 = tf.layers.max_pooling2d(conv2, (2,2), (2,2), padding='same')
#7*7*64
conv3 = tf.layers.conv2d(pooling2, 64, (3,3), padding='same', activation=tf.nn.relu)
pooling3 = tf.layers.max_pooling2d(conv3, (2,2), (2,2), padding='same')
#4*4*64
#全连接层
flat = tf.reshape(pooling3, [-1,4*4*64])  #变为一维向量

w_fc1 = weight_variable([4 * 4 *64, 1024])
b_fc1 = bias_variable([1024])

h_fc1 = tf.nn.relu(tf.matmul(flat, w_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

prediction = tf.nn.softmax(y_conv,name='prediction')    #softmax返回一组概率向量
#建立损失函数，在这里采用交叉熵函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
#最小化损失
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))  #返回bool型(结果，标签)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  #求平均，计算正确率（tf.cast:bool-float转换）
#初始化变量

#sess = tf.Session()
sess.run(tf.global_variables_initializer())
pos = 0
trains = []
print("houyu")
for i in range(6001):
    batch = mnist.train.next_batch(100)
    images = batch[0].reshape((-1,28,28,1)) #像素归一化为[0,1]之间的值
    #print("images shape:",np.shape(images))
    labels = batch[1]
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
        inputs_:images, y: labels, keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: images, y: labels, keep_prob: 0.5})

'''
#images = mnist.test.images[1:10].reshape(-1,28,28,1)
csvFile = open("all/test.csv", "r")   #打开测试数据
reader = csv.reader(csvFile)
# 建立空字典
i = 0
images = {}
labels = {}
for item in reader:
    # 忽略第一行
    if reader.line_num == 1:
        continue
    images[i] = item[0:784] #将图像数据写如images（字典）
    #labels[i] = item[0]
    i = i + 1
csvFile.close()
lab = []
for m in range(280):    #一共28000个数据，一次输入100个
    if m % 20 == 0:
        print(m)
    A = np.array([[images[value]] for value in range(m*100,m*100+100)],dtype='float32')
    A = A / 255
    imgs = A.reshape(-1,28,28,1)

#labels = mnist.test.labels[1:10]
#test = mnist.test.next_batch(10)
#print("test.images:",mnist.test.images.shape,type(mnist.test.images))
#print("images:",images.shape,type(images))
#print("next_batch:",test[0].shape,type(test))
#testdata = test[0].reshape((-1,28,28,1))
#train_accuracy = accuracy.eval(feed_dict={inputs_:images, y: labels, keep_prob: 1.0})
#print("hello, test accuracy %g"%(train_accuracy))
    results = sess.run(prediction,feed_dict={inputs_: imgs, keep_prob: 1.0})    #对测试数据分类
    Results0_1 = sess.run(tf.argmax(results,1)) #返回最大概率的分类值的类别
    Results0_1 = Results0_1.tolist()    #结果转换为list
    lab.extend(Results0_1)  #加到之前的结果末尾
#results = prediction.eval(feed_dict={inputs_: images})
print("Results:", results)
print("Results0_1:", Results0_1, "type:", type(Results0_1))
#print("trueResults:", labels)
print("lab:", lab)

imageshow = np.array([images[27999]],dtype = 'float32')
imageshow = imageshow / 255.0
imageshow = imageshow.reshape((28,28))
plt.imshow(imageshow)
plt.show()

with open("all/empty.csv","w",newline='') as csvfile:
    writer = csv.writer(csvfile, dialect='excel')
    #先写入columns_name
    #writer.writerow(["index","a_name","b_name"])
    writer.writerow(["Label"])
    #写入多行用writerows
    list = [0,1,3,1,2,3,2,3,4]
    a = np.array(lab)
    a = np.transpose([a])
    #print(a)
    writer.writerows(a)
'''






