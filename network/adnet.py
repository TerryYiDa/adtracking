import tensorflow as tf
import tensorflow,contrib,slim as slim
class adnet(object):
     ACTIONS = np.array([
        [-1, 0, 0, 0],
        [-2, 0, 0, 0],
        [+1, 0, 0, 0],
        [+2, 0, 0, 0],
        [0, -1, 0, 0],
        [0, -2, 0, 0],
        [0, +1, 0, 0],
        [0, +2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, -1, -1],
        [0, 0, 1, 1]
        # terminated
    ], dtype=np.float16)
    """docstring for ClassName"""
    def __init__(self, train_config, is_training):
        self.is_training = is_training
        self.train_config = train_config
        self.input_tensor = tf.placeholder(shape = [None, None, None, 3], type =  tf.uint8, name = 'instance_input')
        self.label_tensor = tf.placeholder(shape = [None,], type = tf.uint8, name = 'label_tensor')
        self.class_tensor = tf.placeholder(shape = [None,], type = tf.uint8, name = 'class_tensor')
        self.action_history_tensor = tf.placeholder(shape = [None, 1, 1, 110], name = 'action_history_tensor')
    def create_network(self):
       
        # feature extractor - convolutions
        net = slim.convolution(self.input_tensor, 96, [7, 7], 2, padding='VALID', scope='conv1',
                                   activation_fn=tf.nn.relu)
        net = tf.nn.lrn(net, depth_radius=5, bias=2, alpha=1e-4*5, beta=0.75)
        net = slim.pool(net, [3, 3], 'MAX', stride=2, padding='VALID', scope='pool1')
        # net = self.Squeeze_excitation_layer(net, 96, 4, layer_name = 'SE_1')

        net = slim.convolution(net, 256, [5, 5], 2, padding='VALID', scope='conv2',
                                   activation_fn=tf.nn.relu)
        net = tf.nn.lrn(net, depth_radius=5, bias=2, alpha=1e-4*5, beta=0.75)
        net = slim.pool(net, [3, 3], 'MAX', stride=2, padding='VALID', scope='pool2')
        # net = self.Squeeze_excitation_layer(net, 256, 4, layer_name = 'SE_2')

        net = slim.convolution(net, 512, [3, 3], 1, padding='VALID', scope='conv3',
                                   activation_fn=tf.nn.relu)
        # net  = self.Squeeze_excitation_layer(net, 512, 4, layer_name = 'SE_3')
        self.layer_feat = net

        # fc layers
        net = slim.convolution(net, 512, [3, 3], 1, padding='VALID', scope='fc4',
                                   activation_fn=tf.nn.relu)
        # net = self.Squeeze_excitation_layer(net, 512, 4, layer_name = 'SE_4')

        net = slim.dropout(net, keep_prob=0.5, is_training=self.is_training, scope='dropout')

        net = tf.concat([net, self.action_history_tensor], axis=-1)
            
        net = slim.convolution(net, 512, [1, 1], 1, padding='VALID', scope='fc5',
                                   activation_fn=tf.nn.relu)
        net = slim.dropout(net, keep_prob=0.5, is_training=self.is_training, scope='dropout_x16')

        # auxilaries
        out_actions = slim.convolution(net, 11, [1, 1], 1, padding='VALID', scope='fc6_1', activation_fn=None)
        out_scores = slim.convolution(net, 2, [1, 1], 1, padding='VALID', scope='fc6_2', activation_fn=None)
        self.out_actions = flatten_convolution(out_actions)
        self.out_scores = flatten_convolution(out_scores)
        self.layer_actions = tf.nn.softmax(out_actions)
        self.layer_scores = tf.nn.softmax(out_scores)

        # return self.layer_actions, self.layer_scores
    def flatten_convolution(self, tensor_in):
        tendor_in_shape = tensor_in.get_shape()
        tensor_in_flat = tf.reshape(tensor_in, [tendor_in_shape[0].value or -1, np.prod(tendor_in_shape[1:]).value])
        return tensor_in_flat

     def set_global_step(self):
        self.global_step = tf.Variable(initial_value = 0, name = 'global_step', trainable = False, 
                                    collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
    def confirure_learning_rate(self):
        decay_steps = int(self.train_config['train']['num_examples_per_epoch']) / self.train_config['train']['batch_size']
        self.lr =  tf.train.exponential_decay(0.01, self.global_step, decay_steps, 0.8685113737513527)

    def configure_optimizer(self):
        self.optimizer =  tf.train.MomentumOptimizer(self.lr, self.train_config['momentum'], use_nesterov=False, name='Momentum')
        # for action network
    def train_actionloss_sl(self):
        action_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_tensor, logits=self.out_actions)
        # class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.class_tensor, logits=self.layer_scores)
        # total_loss = tf.reduce_mean(tf.add(action_loss, class_loss), name='batch_loss')
        # total_loss = total_loss + tf.add_n(tf.losses.get_regularization_losses(), name='get_regularization_losses')
        score_op =tf.contrib.layers.optimize_loss(loss=action_loss, global_step= self.global_step, learning_rate=self.lr, optimizer=self.optimizer, clip_gradients=None,
                                        summaries=['learning_rate'])
        with tf.control_dependencies([score_op]):
            train_op_action = tf.no_op(name='train')
            return train_op_action, action_loss

    def train_scoreloss_sl(self):
        score_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.class_tensor, logits=self.out_scores)
        score_op =tf.contrib.layers.optimize_loss(loss=score_loss, global_step= self.global_step, learning_rate=self.lr, optimizer=self.optimizer, clip_gradients=None,
                                        summaries=['learning_rate'])
        with tf.control_dependencies([score_op]):
            train_op_score = tf.no_op(name='train')
            return train_op_score

    def build(self):
        self.create_network()
        self.set_global_step()
        self.confirure_learning_rate()
        self.configure_optimizer()
        # self.train_loss()



    


 

   