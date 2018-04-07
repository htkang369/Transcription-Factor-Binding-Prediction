import csv
import numpy as np
import tensorflow as tf

def one_hot(sequence):
    '''
    This function will transform "AGCT" into binary code
    :param sequence:
    :return: Transformed data
    '''
    data = []
    for ele in sequence:
        if len(data)==0:
            if ele == "G":
                data.append([0, 0, 0, 1])
            elif ele == "T":
                data.append([0, 0, 1, 0])
            elif ele == "A":
                data.append([0, 1, 0, 0])
            else:
                data.append([1, 0, 0, 0])
        else:
            if ele == "G":
                data[0] = data[0]+[0, 0, 0, 1]
            elif ele == "T":
                data[0] = data[0]+[0, 0, 1, 0]
            elif ele == "A":
                data[0] = data[0]+[0, 1, 0, 0]
            else:
                data[0] = data[0]+[1, 0, 0, 0]
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

    train_data = np.zeros([len(raw_data), 56], dtype=float)
    i = 0
    for seq in raw_data:
        train_data[i, :] = one_hot(seq)[0]
        i = i + 1

    train_label = np.zeros([len(raw_data), 2],dtype=float)
    i = 0
    for ele in label:
        if ele=="0":train_label[i,:] = [1,0]
        else:train_label[i,:] = [0,1]
        i = i + 1
    return train_data,np.array(train_label)

def transform_test_data(path):

    raw_data = []
    with open(path, "r") as csvfile:

        reader = csv.reader(csvfile)
        for row in reader:
            if row[0]=='id':continue

            raw_data.append(row[1])

    train_data = np.zeros([len(raw_data), 56], dtype=float)
    i = 0
    for seq in raw_data:
        train_data[i, :] = one_hot(seq)[0]
        i = i + 1
    return train_data

def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    weights = tf.Variable(tf.random_normal([in_size, out_size],mean=0.0, stddev=0.02, dtype=tf.float32))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1 )
    wx_plus_b = tf.matmul(inputs, weights) + biases
    if activation_function is None:
        outputs = wx_plus_b
    else:
        outputs = activation_function(wx_plus_b)
    return outputs



if __name__=='__main__':

    def compute_accuracy(v_xs, v_ys):
        global predict
        y_pre = sess.run(predict, feed_dict={xs: v_xs,keep_prob:1})
        correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
        return result

    def compute_result(v_xs):
        global predict
        y_pre = sess.run(predict, feed_dict={xs: v_xs,keep_prob:1})
        prediction = np.argmax(y_pre, 1)
        # res = sess.run(prediction,feed_dict={xs: v_xs})
        return prediction


    train_data,train_label = transform_train_data('train.csv')
    test_data = transform_test_data('test.csv')

    total_size = train_data.shape[0]
    batch_size = 64
    # define placeholder for inputs to network
    xs = tf.placeholder(tf.float32, [None, 56])  # 2000*56
    ys = tf.placeholder(tf.float32, [None, 2])   # 2000*2
    keep_prob = tf.placeholder(tf.float32) # dropout

    # add output layer
    layer1 = add_layer(xs, 56, 100, activation_function=tf.nn.relu)
    layer1 = tf.nn.dropout(layer1, keep_prob=keep_prob)
    layer2 = add_layer(layer1, 100, 50, activation_function=tf.nn.relu)
    predict = add_layer(layer2,50,2,activation_function=tf.nn.softmax)

    # the error between prediction and real data
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(predict),
                                                  reduction_indices=[1]))  # loss
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    sess = tf.Session()

    init = tf.global_variables_initializer()
    sess.run(init)
    store_loss = []
    for i in range(20000):
        indices = np.random.permutation(total_size)[:batch_size]
        batch_xs = train_data[indices, :]
        batch_ys = train_label[indices, :]
        loss,_ = sess.run([cross_entropy,train_step], feed_dict={xs: batch_xs, ys: batch_ys,keep_prob : 0.5})
        store_loss.append(loss)
        if i%100==0:
            val_indices = np.random.permutation(total_size)[:100]
            v_xs = train_data[val_indices, :]
            v_ys = train_label[val_indices, :]
            print compute_accuracy(v_xs, v_ys)

    with open("fc_loss.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        for ele in store_loss:
            writer.writerow([ele])


    #
    # test_result = compute_result(test_data)
    # with open("test_resutl2.csv", "w") as csvfile:
    #     writer = csv.writer(csvfile)
    #
    #     # columns_name
    #     writer.writerow(["id", "prediction"])
    #     # writerows
    #     i = 0
    #     for ele in test_result:
    #         writer.writerows([[i,ele]])
    #         i = i+1











