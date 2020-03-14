import tensorflow as tf
from tensorflow.python.platform import flags
FLAGS = flags.FLAGS


class ZZNet(object):
    def __init__(self):
        self.channels = 3
        self.dim_hidden = FLAGS.base_num_filters
        self.img_size = 256
        self.train_flag = True
        self.bn = tf.layers.batch_normalization
        #self.bn = batch_norm

    def construct_weights(self):
        '''
        create weights
        '''
        weights_1 = {}
        weights_2 = {}
        weights_3 = {}
        weights_4 = {}

        k = 3
        dtype = tf.float32
        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer = tf.contrib.layers.xavier_initializer(dtype=dtype)
        weights_1['conv_init/kernel'] = tf.get_variable('conv_init/kernel', shape=(3, 3, 3, self.dim_hidden),
                                                      initializer=conv_initializer, dtype=dtype)
        weights_1['conv_init/biases'] = tf.get_variable('conv_init/biases', shape=(self.dim_hidden,),
                                                      initializer=fc_initializer,
                                                      dtype=dtype)
        weights_1['conv1_1/kernel'] = tf.get_variable('conv1_1/kernel',
                                                    shape=(3, 3, self.dim_hidden, self.dim_hidden * 2),
                                                    initializer=conv_initializer, dtype=dtype)
        weights_1['conv1_1/biases'] = tf.get_variable('conv1_1/biases', shape=(self.dim_hidden * 2,),
                                                    initializer=fc_initializer,
                                                    dtype=dtype)
        weights_1['conv1_2/kernel'] = tf.get_variable('conv1_2/kernel',
                                                    shape=(3, 3, self.dim_hidden * 2, self.dim_hidden * 3),
                                                    initializer=conv_initializer, dtype=dtype)
        weights_1['conv1_2/biases'] = tf.get_variable('conv1_2/biases', shape=(self.dim_hidden * 3,),
                                                    initializer=fc_initializer,
                                                    dtype=dtype)
        weights_1['conv1_3/kernel'] = tf.get_variable('conv1_3/kernel',
                                                    shape=(3, 3, self.dim_hidden * 3, self.dim_hidden * 2),
                                                    initializer=conv_initializer, dtype=dtype)
        weights_1['conv1_3/biases'] = tf.get_variable('conv1_3/biases', shape=(self.dim_hidden * 2,),
                                                    initializer=fc_initializer,
                                                    dtype=dtype)
        weights_2['conv2_1/kernel'] = tf.get_variable('conv2_1/kernel',
                                                    shape=(3, 3, self.dim_hidden * 2, self.dim_hidden * 2),
                                                    initializer=conv_initializer, dtype=dtype)
        weights_2['conv2_1/biases'] = tf.get_variable('conv2_1/biases', shape=(self.dim_hidden * 2,),
                                                    initializer=fc_initializer,
                                                    dtype=dtype)
        weights_2['conv2_2/kernel'] = tf.get_variable('conv2_2/kernel',
                                                    shape=(3, 3, self.dim_hidden * 2, self.dim_hidden * 3),
                                                    initializer=conv_initializer, dtype=dtype)
        weights_2['conv2_2/biases'] = tf.get_variable('conv2_2/biases', shape=(self.dim_hidden * 3,),
                                                    initializer=fc_initializer,
                                                    dtype=dtype)
        weights_2['conv2_3/kernel'] = tf.get_variable('conv2_3/kernel',
                                                    shape=(3, 3, self.dim_hidden * 3, self.dim_hidden * 2),
                                                    initializer=conv_initializer, dtype=dtype)
        weights_2['conv2_3/biases'] = tf.get_variable('conv2_3/biases', shape=(self.dim_hidden * 2,),
                                                    initializer=fc_initializer,
                                                    dtype=dtype)
        weights_3['conv3_1/kernel'] = tf.get_variable('conv3_1/kernel',
                                                    shape=(3, 3, self.dim_hidden * 2, self.dim_hidden * 2),
                                                    initializer=conv_initializer, dtype=dtype)
        weights_3['conv3_1/biases'] = tf.get_variable('conv3_1/biases', shape=(self.dim_hidden * 2,),
                                                    initializer=fc_initializer,
                                                    dtype=dtype)
        weights_3['conv3_2/kernel'] = tf.get_variable('conv3_2/kernel',
                                                    shape=(3, 3, self.dim_hidden * 2, self.dim_hidden * 3),
                                                    initializer=conv_initializer, dtype=dtype)
        weights_3['conv3_2/biases'] = tf.get_variable('conv3_2/biases', shape=(self.dim_hidden * 3,),
                                                    initializer=fc_initializer,
                                                    dtype=dtype)
        weights_3['conv3_3/kernel'] = tf.get_variable('conv3_3/kernel',
                                                    shape=(3, 3, self.dim_hidden * 3, self.dim_hidden * 2),
                                                    initializer=conv_initializer, dtype=dtype)
        weights_3['conv3_3/biases'] = tf.get_variable('conv3_3/biases', shape=(self.dim_hidden * 2,),
                                                    initializer=fc_initializer,
                                                    dtype=dtype)
        weights_4['conv4_1/kernel'] = tf.get_variable('conv4_1/kernel',
                                                    shape=(3, 3, self.dim_hidden * 6, self.dim_hidden * 2),
                                                    initializer=conv_initializer, dtype=dtype)
        weights_4['conv4_1/biases'] = tf.get_variable('conv4_1/biases', shape=(self.dim_hidden * 2,),
                                                    initializer=fc_initializer,
                                                    dtype=dtype)
        weights_4['conv4_2/kernel'] = tf.get_variable('conv4_2/kernel',
                                                    shape=(3, 3, self.dim_hidden * 2, self.dim_hidden),
                                                    initializer=conv_initializer, dtype=dtype)
        weights_4['conv4_2/biases'] = tf.get_variable('conv4_2/biases', shape=(self.dim_hidden,),
                                                    initializer=fc_initializer,
                                                    dtype=dtype)
        weights_4['conv4_3/kernel'] = tf.get_variable('conv4_3/kernel', shape=(3, 3, self.dim_hidden, 1),
                                                    initializer=conv_initializer,
                                                    dtype=dtype)
        weights_4['conv4_3/biases'] = tf.get_variable('conv4_3/biases', shape=(1,), initializer=fc_initializer,
                                                    dtype=dtype)

        return weights_1, weights_2, weights_3, weights_4



    def forward(self, inp, weights, reuse=False):
        # forward of the representation module
        inp = tf.reshape(inp, shape=[-1, 256, 256, 3])
        net = tf.nn.conv2d(inp, weights['conv_init/kernel'], [1, 1, 1, 1], padding='SAME') + weights['conv_init/biases']
        net = tf.nn.relu(net)
        net = self.bn(net, name='bn_init', training=True)

        net = tf.nn.conv2d(net, weights['conv1_1/kernel'], [1, 1, 1, 1], padding='SAME') + weights['conv1_1/biases']
        net = tf.nn.relu(net)
        net = self.bn(net, name='bn1_1', training=True)

        net = tf.nn.conv2d(net, weights['conv1_2/kernel'], [1, 1, 1, 1], padding='SAME') + weights['conv1_2/biases']
        net = tf.nn.relu(net)
        net = self.bn(net, name='bn1_2', training=True)

        net = tf.nn.conv2d(net, weights['conv1_3/kernel'], [1, 1, 1, 1], padding='SAME') + weights['conv1_3/biases']
        net = tf.nn.relu(net)
        net = self.bn(net, name='bn1_3', training=True)

        pool1 = tf.nn.max_pool(net, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        net = tf.nn.conv2d(pool1, weights['conv2_1/kernel'], [1, 1, 1, 1], padding='SAME') + weights['conv2_1/biases']
        net = tf.nn.relu(net)
        net = self.bn(net, name='bn2_1', training=True)

        net = tf.nn.conv2d(net, weights['conv2_2/kernel'], [1, 1, 1, 1], padding='SAME') + weights['conv2_2/biases']
        net = tf.nn.relu(net)
        net = self.bn(net, name='bn2_2', training=True)

        net = tf.nn.conv2d(net, weights['conv2_3/kernel'], [1, 1, 1, 1], padding='SAME') + weights['conv2_3/biases']
        net = tf.nn.relu(net)
        net = self.bn(net, name='bn2_3', training=True)

        pool2 = tf.nn.max_pool(net, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        net = tf.nn.conv2d(pool2, weights['conv3_1/kernel'], [1, 1, 1, 1], padding='SAME') + weights['conv3_1/biases']
        net = tf.nn.relu(net)
        net = self.bn(net, name='bn3_1', training=True)

        net = tf.nn.conv2d(net, weights['conv3_2/kernel'], [1, 1, 1, 1], padding='SAME') + weights['conv3_2/biases']
        net = tf.nn.relu(net)
        net = self.bn(net, name='bn3_2', training=True)

        net = tf.nn.conv2d(net, weights['conv3_3/kernel'], [1, 1, 1, 1], padding='SAME') + weights['conv3_3/biases']
        net = tf.nn.relu(net)
        net = self.bn(net, name='bn3_3', training=True)

        pool3 = tf.nn.max_pool(net, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        feature1 = tf.layers.average_pooling2d(pool1, pool_size=[4, 4], strides=[4, 4], padding='same')
        feature2 = tf.layers.average_pooling2d(pool2, pool_size=[2, 2], strides=[2, 2], padding='same')

        feature3 = pool3
        # print('pool1 pool2', pool1.get_shape(), pool2.get_shape())
        pool_concat = tf.concat([feature1, feature2, feature3], axis=-1)

        net = tf.nn.conv2d(pool_concat, weights['conv4_1/kernel'], [1, 1, 1, 1], padding='SAME') + weights[
            'conv4_1/biases']
        net = tf.nn.relu(net)
        net = self.bn(net, name='bn4_1', training=True)

        net = tf.nn.conv2d(net, weights['conv4_2/kernel'], [1, 1, 1, 1], padding='SAME') + weights['conv4_2/biases']
        net = tf.nn.relu(net)
        net = self.bn(net, name='bn4_2', training=True)

        net = tf.nn.conv2d(net, weights['conv4_3/kernel'], [1, 1, 1, 1], padding='SAME') + weights['conv4_3/biases']
        net = tf.nn.relu(net)

        return net






