import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
import numpy as np

VDCNN_VARIABLES = 'vdcnn_variables'


class Discriminator(object):
  def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, pretrained_path=False, l2_reg_lambda=0.0,depth=9):
    self.conv_depth = {9: [2, 2, 2, 2], 17: [4, 4, 4, 4], 29: [10, 10, 4, 4], 49: [16, 16, 10, 6]}                         
    self.s = sequence_length
    self.f0 = 1  # means "language channel" similar to image channel
    self.embedding_size = embedding_size
    self.temp_kernel = (3, embedding_size)
    self.kernel = (3, 1)
    self.stride = (2, 1)
    self.kmax = 8  # not useful until kmax pooling implemented
    self.num_filters = [64, 128, 256, 512]
    self.activation = tf.nn.relu
    self.fc1_hidden_size = 1024
    self.fc2_hidden_size = 512
    self.num_output = num_classes
    self.l2_reg_lambda = l2_reg_lambda
        #self._extra_train_ops = []  # used to store all the extra train operations other than gradient descent
    self.depth = depth
    self.vocab_size = vocab_size
        #self.is_training = is_training
    self.num_classes = num_classes	
        # Placeholders for input, output and dropout
    self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
    self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
    #self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
    self.pretrained_path = pretrained_path

        # Keeping track of l2 regularization loss (optional)
    l2_loss = tf.constant(0.0)

    with tf.variable_scope('discriminator'):
        # Embedding layer
      with tf.name_scope("embedding"):
        self.embedding_matrix = tf.Variable(tf.random_normal([vocab_size, embedding_size]), name="embedding_matrix")
          #self.W = tf.constant(self.init_matrix(vocab_size, embedding_size))
        self.embedded_chars = tf.nn.embedding_lookup(self.embedding_matrix, self.input_x)
        self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
        embedded_f0_channel = tf.reshape(self.embedded_chars_expanded,[-1, self.f0, self.s, embedding_size])
      out = self._conv(x=embedded_f0_channel, kernel=self.temp_kernel,stride=self.stride, filters_out=self.num_filters[0], name='conv0')
      out = self.activation(features=out, name='relu0')
      # CONVOLUTION_BLOCK 64 FILTERS
      block_id = 0
      with tf.variable_scope('block' + str(block_id)):
        for conv_id in range(self.conv_depth[self.depth][block_id]):
          out = self._unit_conv_block(out, block_id, conv_id)
      
      #CONVOLUTION_BLOCK 128 FILTERS
      block_id = 1
      with tf.variable_scope('block' + str(block_id)):
        for conv_id in range(self.conv_depth[self.depth][block_id]):
          out = self._unit_conv_block(out, block_id, conv_id)
      # CONVOLUTION_BLOCK  256 FILTERS
      block_id = 2
      with tf.variable_scope('block' + str(block_id)):
        for conv_id in range(self.conv_depth[self.depth][block_id]):
          out = self._unit_conv_block(out, block_id, conv_id)
      # CONVOLUTION_BLOCK 512 FILTERS
      block_id = 3
      with tf.variable_scope('block' + str(block_id)):
        for conv_id in range(self.conv_depth[self.depth][block_id]):
          out = self._unit_conv_block(out, block_id, conv_id)
      max_pool = self._max_pool(out)
      multiplier = max_pool.get_shape()[2].value
      flatten = tf.reshape(max_pool,[-1,self.f0 * multiplier * self.num_filters[block_id]])
      #Fully connected layers (fc)
      #fc1
      fc1 = self._fc(flatten, self.fc1_hidden_size, 'fc1')
      act_fc1 = self.activation(fc1)
      #fc2
      fc2 = self._fc(act_fc1, self.fc2_hidden_size, 'fc2')
      act_fc2 = self.activation(fc2)
      # calculate mean cross-entropy loss
      logits = self._fc(act_fc2, self.num_output, 'softmax')
      self.logits = logits	
      
      with tf.name_scope("output"):
        self.ypred_for_auc = tf.nn.softmax(self.logits)
        self.predictions = tf.argmax(self.logits, 1, name="predictions")
      #CalculateMean cross-entropy loss
      with tf.name_scope("loss"):
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
        #if weighted_loss:
        #losses = tf.divide(losses, tf.reduce_sum(self.input_y, axis=1))
        self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
      with tf.name_scope("accuracy"):
        correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
    self.params = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
    d_optimizer = tf.train.AdamOptimizer(1e-4)
    grads_and_vars = d_optimizer.compute_gradients(self.loss, self.params, aggregation_method=2)
    self.train_op = d_optimizer.apply_gradients(grads_and_vars)
			
			
			
  def _unit_conv_block(self, input_layer, block_id, conv_id):
    unit_id = str(block_id) + "_" + str(conv_id)
    num_filters = self.num_filters[block_id]
    conv = self._conv(x=input_layer, kernel=self.kernel, stride=self.stride, filters_out=num_filters, name='conv' + unit_id)
    norm = self._batch_norm(conv, is_training=tf.cast(True, tf.bool), name='norm' + unit_id)
    act = self.activation(features=norm, name='relu' + unit_id)

    return act

  def _fc(self, x, units_out, name):
    num_units_in = x.get_shape()[1]
    num_units_out = units_out

    weights_initializer = tf.truncated_normal_initializer(stddev=0.0002)

    weights = self._get_variable(name + str(units_out) + 'weights', shape=[num_units_in, num_units_out], initializer=weights_initializer, weight_decay=0.0002)
    biases = self._get_variable(name + str(units_out) + 'biases', shape=[num_units_out], initializer=tf.zeros_initializer())
    x = tf.nn.xw_plus_b(x, weights, biases)

    return x

  def _get_variable(self, name, shape, initializer, weight_decay=0.0, dtype='float', trainable=True):
    if weight_decay > 0:
      regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
      regularizer = None
    collections = [tf.GraphKeys.GLOBAL_VARIABLES, VDCNN_VARIABLES]
    return tf.get_variable(name, shape=shape, initializer=initializer, dtype=dtype, regularizer=regularizer, collections=collections, trainable=trainable)

  def _conv(self, x, kernel, stride, filters_out, name):
    filters_in = x.get_shape()[-1]
    shape = [kernel[0], kernel[1], filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=0.01)
    weights = self._get_variable('weights' + "_" + name, shape=shape, dtype='float', initializer=initializer, weight_decay=0.00000)

    return tf.nn.conv2d(x, weights, [1, stride[0], stride[1], 1], padding='SAME')

  def _max_pool(self, x, ksize=1, stride=1):
    return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding='VALID')

  def _batch_norm(self, x, is_training, name):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    axis = list(range(len(x_shape) - 1))

    beta = self._get_variable('beta' + "_" + name, params_shape, initializer=tf.zeros_initializer())
    gamma = self._get_variable('gamma' + "_" + name, params_shape, initializer=tf.ones_initializer())

    moving_mean = self._get_variable('moving_mean' + "_" + name, params_shape, initializer=tf.zeros_initializer(), trainable=False)
    moving_variance = self._get_variable('moving_variance' + "_" + name, params_shape, initializer=tf.ones_initializer(), trainable=False)

            # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, .99)
    update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, .99)
    #self._extra_train_ops.append(update_moving_mean)
    #self._extra_train_ops.append(update_moving_variance)

    mean, variance = control_flow_ops.cond(is_training, lambda: (mean, variance), lambda: (moving_mean, moving_variance))

    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 1e-6)

    return x			


    """ 
    
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    

    def __init__(self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, pretrained_path=False, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        #self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.pretrained_path = pretrained_path

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        with tf.variable_scope('discriminator'):
            # Embedding layer
            with tf.name_scope("embedding"):
                self.embedding_matrix = tf.Variable(tf.random_normal([vocab_size, embedding_size]), name="embedding_matrix")
                #self.W = tf.constant(self.init_matrix(vocab_size, embedding_size))
                self.embedded_chars = tf.nn.embedding_lookup(self.embedding_matrix, self.input_x)
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for filter_size, num_filter in zip(filter_sizes, num_filters):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, num_filter]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filter]), name="b")
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = sum(num_filters)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

            # Add highway
            with tf.name_scope("highway"):
                self.h_highway = highway(self.h_pool_flat, self.h_pool_flat.get_shape()[1], 1, 0)

            # Add dropout
            #with tf.name_scope("dropout"):
                #self.h_drop = tf.nn.dropout(self.h_highway, self.dropout_keep_prob)

            # Final (unnormalized) scores and predictions
            with tf.name_scope("output"):
                W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
                self.ypred_for_auc = tf.nn.softmax(self.scores)
                self.predictions = tf.argmax(self.scores, 1, name="predictions")

            # CalculateMean cross-entropy loss
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
                self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        self.params = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
        d_optimizer = tf.train.AdamOptimizer(1e-4)
        grads_and_vars = d_optimizer.compute_gradients(self.loss, self.params, aggregation_method=2)
        self.train_op = d_optimizer.apply_gradients(grads_and_vars)

    def init_matrix(self, vocab_size, embedding_size, extend_vocab_size):
        embeddings = []
        # with open(self.pretrained_path) as fs:
        #     for line in fs:
        #         data = line.split()
        #         embeddings.append(map(float, data[1:]))

        embeddings = np.random.random([vocab_size, embedding_size])

        # [UNK], [PAD], [START] and [STOP] get the ids 0,1,2,3.
        # add extend_vocab
        embedding_matrix = np.pad(embeddings,
                                  pad_width=[[4, extend_vocab_size], [0, 0]],
                                  mode='constant',
                                  constant_values=0.).astype(np.float32)
        return embedding_matrix

# An alternative to tf.nn.rnn_cell._linear function, which has been removed in Tensorfow 1.0.1
# The highway layer is borrowed from https://github.com/mkroutikov/tf-lstm-char-cnn
def linear(input_, output_size, scope=None):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * input_[i] ) + Bias[k]
    Args:
    input_: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(input_[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  '''

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """#Highway Network (cf. http://arxiv.org/abs/1505.00387).
    #t = sigmoid(Wy + b)
    #z = t * g(Wy + b) + (1 - t) * y
    #where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """
    output = None
    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))

            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output
    """
