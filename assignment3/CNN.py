# -*- coding: UTF-8 -*-
import csv
import numpy as np
import tensorflow as tf

# make data
def one_hot(sequence):
    '''
    This function will transform "AGCT" into binary code
    :param sequence:
    :return: Transformed data
    '''
    data = []
    for ele in sequence:
        if ele == "G":
            data.append([0, 0, 0, 1])
        elif ele == "T":
            data.append([0, 0, 1, 0])
        elif ele == "A":
            data.append([0, 1, 0, 0])
        else:
            data.append([1, 0, 0, 0])
    return data

def transform_train_data(path):
    label = []
    raw_data = []
    with open(path, "r") as csvfile:

        reader = csv.reader(csvfile)
        for row in reader:
            if row[0]=='id':continue
            label.append(row[2])
            raw_data.append(row[1])

    train_data = []
    for seq in raw_data:
        train_data.append(one_hot(seq))


    train_label = np.zeros([len(raw_data), 2],dtype=float)
    i = 0
    for ele in label:
        if ele=="0":train_label[i,:] = [1,0]
        else:train_label[i,:] = [0,1]
        i = i + 1
    return np.array(train_data,dtype=float),np.array(train_label)

def transform_test_data(path):

    raw_data = []
    with open(path, "r") as csvfile:

        reader = csv.reader(csvfile)
        for row in reader:
            if row[0]=='id':continue

            raw_data.append(row[1])

    test_data = []
    for seq in raw_data:
        test_data.append(one_hot(seq))
    return np.array(test_data,dtype=float)

def get_val_data(x,y):
    r = np.random.permutation(x.shape[0])
    v = r[0:20]
    t = r[20:]
    val_data = x[v,:,:]
    val_label = y[v,:]

    train_data = x[t,:,:]
    train_label = y[t,:]
    return val_data,val_label,train_data,train_label

def weight_variable(shape):
    initial = tf.truncated_normal(shape, mean=0,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv1d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv1d(x, W, stride=1, padding='SAME')

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def compute_result(v_xs):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs,keep_prob:1})
    pred = np.argmax(y_pre, 1)
    return pred

if __name__=='__main__':

    # get data
    train_data,train_label = transform_train_data('train.csv') #(2000, 14, 4),(2000,2)
    test_data = transform_test_data('test.csv') # (400,14,4)
    val_data, val_label,train_data,train_label = get_val_data(train_data, train_label)
    # hyper parameter
    total_size = train_data.shape[0]
    batch_size = 64

    lr = 0.01       # learning rate

    xs = tf.placeholder(tf.float32, [None, 14, 4])
    ys = tf.placeholder(tf.float32, [None, 2])
    keep_prob = tf.placeholder(tf.float32)

    ## conv1 layer ##
    W_conv1 = weight_variable([5,4,16]) # patch 1*3, in size 4, out size 16
    b_conv1 = bias_variable([16])
    h_conv1 = tf.nn.relu(conv1d(xs, W_conv1) + b_conv1) # output size (2000,14,16)
    print h_conv1.shape
    ## conv2 layer ##
    W_conv2 = weight_variable([3,16,32]) # patch 1*5, in size 16, out size 32
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(conv1d(h_conv1, W_conv2) + b_conv2) # output size (2000,14,32)

    print h_conv2.shape
    ## fc1 layer ##
    W_fc1 = weight_variable([14*32, 100])
    b_fc1 = bias_variable([100])
    # [n_samples, 14,32] ->> [n_samples,14*32]
    h_pool2_flat = tf.reshape(h_conv2, [-1, 14*32])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    ## fc2 layer ##
    W_fc2 = weight_variable([100, 2])
    b_fc2 = bias_variable([2])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # the error between prediction and real data
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                                  reduction_indices=[1]))       # loss
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    sess = tf.Session()
    # important step
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)

    store_loss=[]


    for i in range(10000):
        indices = np.random.permutation(1980)[:batch_size]
        batch_xs = train_data[indices, :,:]
        batch_ys = train_label[indices, :]

        _,loss_ = sess.run([train_step,cross_entropy], feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
        store_loss.append(loss_)
        if i % 50 == 0:
            indices = np.random.permutation(total_size)[:batch_size]
            batch_xs = val_data
            batch_ys = val_label
            accuracy_ = compute_accuracy(batch_xs,batch_ys)
            print('train loss: %.8f' % loss_, '| test accuracy: %.4f' % accuracy_, 'step:%d' % i)

    with open("cnn_loss.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        for ele in store_loss:
            writer.writerow([ele])

    # test_result = compute_result(test_data)
    # with open("test_resutl7.csv", "w") as csvfile:
    #     writer = csv.writer(csvfile)
    #
    #     # columns_name
    #     writer.writerow(["id", "prediction"])
    #     # writerows
    #     i = 0
    #     for ele in test_result:
    #         writer.writerows([[i, ele]])
    #         i = i + 1