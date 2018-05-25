import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# load the dataset into memory
data_file = open('train_formula.txt', 'rb')
train_data = np.asanyarray(pickle.load(data_file), np.float32)
data_file.close()
data_file = open('train_label.txt', 'rb')
train_labels = np.asanyarray(pickle.load(data_file), np.int64)
data_file.close()
data_file = open('test_formula.txt', 'rb')
test_data = np.asanyarray(pickle.load(data_file), np.float32)
data_file.close()
data_file = open('test_label.txt', 'rb')
test_labels = np.asanyarray(pickle.load(data_file), np.int64)
data_file.close()

samples = 75000
testsamples = 25000
x_train = np.concatenate((train_data,test_data[0:testsamples,:,:]))
x_train = np.reshape(x_train, (samples, 4, 4, -1))
x_test = np.reshape(test_data[testsamples:,:,:], (testsamples, 4, 4, -1))
y_train = np.concatenate((train_labels,test_labels[0:testsamples]))
y_test = test_labels[testsamples:]

steps = 7500
batch = 10
recordstep = 751
iter = np.ones((recordstep))
accur = np.ones((recordstep))
lossfunc = np.ones((recordstep))
accur_test = np.ones((recordstep))
loss_test = np.ones((recordstep))

# Weight and Bias
def weight_variable(shape, name):
  W = tf.get_variable(name, shape=shape,initializer=tf.contrib.layers.xavier_initializer())
  return W


def bias_variable(shape):
  initial = tf.constant(0.0, shape=shape)
  return tf.Variable(initial)


x = tf.placeholder(shape=[None, 4, 4, 1], dtype=tf.float32,name='images')
y = tf.placeholder(shape=[None], dtype=tf.int64)

# Convolution and Pooling
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 1, 1, 1], padding='VALID')

# 1st Convolution layer - Activation - Pooling Layer
W_conv1 = weight_variable([3, 3, 1, 4],'W1')
b_conv1 = bias_variable([4])
x_image = tf.reshape(x, [-1, 4, 4, 1])
conv1 = tf.add(conv2d(x_image, W_conv1), b_conv1, name='conv_l1')
h_conv1 = tf.nn.relu(conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Fully Connected layer
W_fc1 = weight_variable([1 * 1 * 4, 20],'W4')
b_fc1 = bias_variable([20])
h_conv3_flat = tf.reshape(h_pool1, [-1, 1*1*4])
h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Output Layer
W_fc2 = weight_variable([20, 2],'W5')
b_fc2 = bias_variable([2])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_hat = tf.nn.softmax(y_conv, axis=1, name='Y_hat')

y_4sess = tf.matmul(h_fc1, W_fc2) + b_fc2
y_4sess1 = tf.nn.softmax(y_4sess, axis=1, name='Y_hat')
pred_op = tf.cast(tf.argmax((y_4sess1),1),tf.int64)

# Training
cross_entropy = tf.reduce_mean(
tf.nn.softmax_cross_entropy_with_logits(labels=tf.cast(tf.one_hot(y, 2), tf.float32), logits=y_hat))
train_step = tf.train.AdamOptimizer(.00005,.9,.999).minimize(cross_entropy)
y_pred = tf.argmax(y_conv, 1)
correct_prediction = tf.equal(y_pred, y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# For Test set
cross_entropy_test = tf.reduce_mean(
tf.nn.softmax_cross_entropy_with_logits(labels=tf.cast(tf.one_hot(y, 2), tf.float32), logits=y_4sess1))
correct_prediction_test = tf.equal(pred_op, y)
accuracy_test = tf.reduce_mean(tf.cast(correct_prediction_test, tf.float32))

q = 0
final_step = 0
limit = np.float64(0.7)

# Save the model
tf.get_collection('validation_nodes')

# Add opts to the collection
tf.add_to_collection('validation_nodes', x)
tf.add_to_collection('validation_nodes', y)
tf.add_to_collection('validation_nodes', pred_op)

# start training
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(steps):
        index = batch * (i % 100)

        # ACCURACY and LOSS
        if i % 100 == 0:
            iter[q] = i

            # TRAINING ACCURACY and LOSS
            lossfunc[q],accur[q] = sess.run([cross_entropy, accuracy],feed_dict={
              x: x_train[index:index+batch,:,:,:], y: y_train[index:index+batch], keep_prob: 1.0})
            print('step %d, training accuracy %g, training loss %f' % (iter[q], accur[q], lossfunc[q]))

            # TESTING ACCURACY and LOSS
            loss_test[q], accur_test[q] = sess.run([cross_entropy_test, accuracy_test], feed_dict={
              x: x_test, y: y_test})
            print('step %d, test accuracy %g, test loss %f' % (iter[q], accur_test[q], loss_test[q]))
            # if accur_test[q] >= limit:
            #     print(accur_test[q] >= limit)
            #     final_step = i
            #     q += 1
            #     break
            q += 1

        # TRAIN
        train_step.run(
            feed_dict={x: x_train[index:index + batch, :, :, :], y: y_train[index:index + batch], keep_prob: 0.75})


    #save_path = saver.save(sess, "./my_model")

    # FINAL ACCURACY and LOSS
    iter[q] = steps
    lossfunc[q], accur[q] = sess.run([cross_entropy, accuracy], feed_dict={
        x: x_train[samples - batch:samples, :, :, :], y: y_train[samples - batch:samples], keep_prob: 1.0})
    print('step %d, training accuracy %g, training loss %f' % (iter[q], accur[q], lossfunc[q]))
    loss_test[q], accur_test[q], prediction = sess.run([cross_entropy_test, accuracy_test, pred_op], feed_dict={
        x: x_test, y: y_test})
    print('step %d, test accuracy %g, test loss %f' % (iter[q], accur_test[q], loss_test[q]))


fig1 = plt.figure()
plt.scatter(iter[0:q], accur[0:q],color='blue')
plt.xlabel('epoch')
plt.ylabel('train accuracy')
plt.show()

fig2 = plt.figure()
plt.scatter(iter[0:q], lossfunc[0:q],color='blue')
plt.xlabel('epoch')
plt.ylabel('train loss')
plt.show()

fig3 = plt.figure()
plt.scatter(iter[0:q], accur_test[0:q],color='green')
plt.xlabel('epoch')
plt.ylabel('test accuracy')
plt.show()

fig4 = plt.figure()
plt.scatter(iter[0:q], loss_test[0:q],color='green')
plt.xlabel('epoch')
plt.ylabel('test loss')
plt.show()