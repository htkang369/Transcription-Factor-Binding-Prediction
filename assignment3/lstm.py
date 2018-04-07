import tensorflow as tf
import csv
import numpy as np

tf.set_random_seed(1)
np.random.seed(1)

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

# data
train_data,train_label = transform_train_data('train.csv')
test_data = transform_test_data('test.csv')

total_size = train_data.shape[0]

# Hyper Parameters
BATCH_SIZE = 64
TIME_STEP = 14          # rnn time step / image height
INPUT_SIZE = 4         # rnn input size / image width
LR = 0.01               # learning rate



# tensorflow placeholders
tf_x = tf.placeholder(tf.float32, [None, TIME_STEP * INPUT_SIZE])       # shape(batch, 56)
image = tf.reshape(tf_x, [-1, TIME_STEP, INPUT_SIZE])                   # (batch, 14,4)
tf_y = tf.placeholder(tf.int32, [None, 2])                             # input y

# RNN
rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=64)
outputs, (h_c, h_n) = tf.nn.dynamic_rnn(
    rnn_cell,                   # cell you have chosen
    image,                      # input
    initial_state=None,         # the initial hidden state
    dtype=tf.float32,           # must given if set initial_state = None
    time_major=False,           # False: (batch, time step, input); True: (time step, batch, input)
)
output = tf.layers.dense(outputs[:, -1, :], 2)              # output based on the last output step


loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)           # compute cost
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
sess.run(init_op)     # initialize var in graph
store_loss=[]
for step in range(3000):    # training
    indices = np.random.permutation(total_size)[:BATCH_SIZE]
    b_x = train_data[indices, :]
    b_y = train_label[indices, :]

    _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})
    store_loss.append(loss_)

    if step % 50 == 0:      # testing
        indices = np.random.permutation(total_size)[:BATCH_SIZE]
        b_x = train_data[indices, :]
        b_y = train_label[indices, :]
        accuracy_ = sess.run(accuracy, {tf_x: b_x, tf_y: b_y})
        print('train loss: %.8f' % loss_, '| test accuracy: %.4f' % accuracy_,'step:%d' % step)

with open("lstm_loss.csv", "w") as csvfile:
    writer = csv.writer(csvfile)
    for ele in store_loss:
        writer.writerow([ele])
#
# # print 10 predictions from test data
# test_output = sess.run(output, {tf_x: test_data})
# pred_y = np.argmax(test_output, 1)
# print(pred_y, 'prediction number')
#
# with open("test_resutl4.csv", "w") as csvfile:
#     writer = csv.writer(csvfile)
#
#     # columns_name
#     writer.writerow(["id", "prediction"])
#     # writerows
#     i = 0
#     for ele in pred_y:
#         writer.writerows([[i, ele]])
#         i = i + 1