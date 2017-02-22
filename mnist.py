import tensorflow as tf
import pandas as pd
import numpy as np

BATCH_SIZE = 100
VALIDATION_SIZE = 5000
def dense_to_one_hot(labels_dense, num_classes):   #Convert to One Hot Vectors
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


train_data = pd.read_csv('train.csv')
images = train_data.iloc[:,1:].values
images = images.astype(np.float)
# Normalize from [0:255] => [0.0:1.0]
images = np.multiply(images, 1.0 / 255.0)
image_size = images.shape[1]
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)
#Labels
labels_flat = train_data[[0]].values.ravel()
labels_count = np.unique(labels_flat).shape[0]
labels = dense_to_one_hot(labels_flat, labels_count)
labels = labels.astype(np.uint8)

train_images = images[VALIDATION_SIZE:]
train_labels = labels[VALIDATION_SIZE:]

# read test data from CSV file 
test_images = pd.read_csv('test.csv').values

test_images = test_images.astype(np.float)

# convert from [0:255] => [0.0:1.0]
test_images = np.multiply(test_images, 1.0 / 255.0)

sess = tf.InteractiveSession()


# Next Batch
def next_batch(batch_size):    
    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed
    
    start = index_in_epoch
    index_in_epoch += batch_size
    
    # when all trainig data have been already used, it is reorder randomly    
    if index_in_epoch > num_examples:
        epochs_completed += 1
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]


def weight_variable(shape):#Weight Var
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):#Bias vae
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):#2d Convolution
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):#Pooling
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
						

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])



W_conv1 = weight_variable([5, 5, 1, 32])    #CNN 1
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) #CNN1
h_pool1 = max_pool_2x2(h_conv1)#Pooling



W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)#CNN2
h_pool2 = max_pool_2x2(h_conv2)#POOL2


W_fc1 = weight_variable([7 * 7 * 64, 1024])#Fully Connected Layer1
b_fc1 = bias_variable([1024]) 

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


keep_prob = tf.placeholder(tf.float32)  #Drop out
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)   #Fully Connected Layer2

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())


epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]

for i in range(15000):
  batch_images,batch_labels = next_batch(BATCH_SIZE)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch_images, y_: batch_labels, keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch_images, y_: batch_labels, keep_prob: 0.5})


predict = tf.argmax(y_conv, 1)
EVAL_BATCH_SIZE = 50
# generating predicted labels
pred = np.zeros(test_images.shape[0])
for i in range(0,test_images.shape[0]//EVAL_BATCH_SIZE):
    pred[i*EVAL_BATCH_SIZE : (i+1)*EVAL_BATCH_SIZE] = predict.eval(feed_dict={x: test_images[i*EVAL_BATCH_SIZE : (i+1)*EVAL_BATCH_SIZE], keep_prob: 1.0})

# write output
np.savetxt('Submission.csv', 
           np.c_[range(1,len(test_images)+1),pred], 
           delimiter=',', 
           header = 'ImageId,Label', 
           comments = '', 
           fmt='%d')