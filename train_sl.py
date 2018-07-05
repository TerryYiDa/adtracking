import tensorflow as tf
from network.adnet import adnet
import numpy as np
import os
from util.get_samples import get_posneg_samples
from util.get_labels import get_labels
from util.extract_region import extract_region
from util.util import auto_select_gpu, chunker, random_idxs, choices_by_idx
@ex.config
def configurations():
  train_config = configuration.TRAIN_CONFIG

def _load_traindb(train_config):
	datasource_path = train_config['train_db']
	with open(datasource_path, 'rb') as f:
      	imdb = pickle.load(f)

     return imdb

def _get_train_size(train_config):
	traindata = _load_traindb(train_config)
	videos = train_data['videos']
	return len(videos)


def _traindb(train_config, index):
	imdb = _load_traindb(train_config)
	videos = imdb['videos']
	ground_box = imdb['box']
	videos_len = len(self.videos[index])
	shuff_id = np.random.shuffle(np.arange(videos_len))

	return  videos, ground_box, shuff_id


@ex.automain
 def main(train_config):
 	os.environ['CUDA_VISIBLE_DEVICES'] = auto_select_gpu()
 	g = tf.Graph()
 	with g.as_default():
 		sess_config  = tf.ConfigProto(gpu_options = tf.GPUOptions(allow_growth = True))
 		sess = tf.Session(config =  sess_config)
 		Adnetwork = adnet(train_config, is_training = True)
 		Adnetwork.build()
 		action_op, action_loss  = Adnetwork.train_actionloss_op()
	 	for epoch in range(train_config['num_epoch']):
	 		train_size = _get_train_size(train_config)
	 		
	 		for  i in range(train_size):
	 			videos, ground_box, img_id = _traindb( train_config, i)
	 			for j in range(train_config['frame_batch']):

	 				img_path = videos[img_id[j]]
	 				bbox = ground_box[img_id[j]]
	 				img_file = tf.read_file(img_path)
	 				image = tf.image.decode_jpeg(img_file, channels=3, dct_method = 'INTEGER_ACCURATE')
	 				img =  tf.to_float(image)
	 				pos_samples, neg_samples = get_posneg_samples(img, groundbox, train_config['pos_neg'], train_config['neg_size'])
	 				pos_lb_action = get_labels(train_config, pos_samples, bbox)
	 				extract_pos_examples = [extract_region(image, box) for box in pos_samples ]
	 				# feats = []
	 				# for batch in chunker(extract_examples, train_config['batch_size']):
	 				# 	feats_batch = sess.run(Adnetwork.layer_feats, feed_dict={ adnet.input_tensor: batch})
	 				# 	feats.append(feats_batch, feed_dict{adnet.input_tensor: feats_batch})

 					extract_neg_examples = [extract_region(image, box) for box in neg_samples ]
 					iter1 = int(np.ceil(len(pos_lb_action) / train['mini_size']))
 					for index in range(iter):
 						start = index * train_config['mini_size']
 						batch_sa = extract_pos_examples[start: start+ train_config['mini_size']]
 						batch_la = pos_lb_action[start: start+ train_config['mini_size']]

 						_, action_bath_loss = sess.run(action_op, action_loss, feed_dict= {Adnetwork.input_tensor : batch_sa, Adnetwork.label_tensor: batch_la, Adnetwork.action_history_tensor:np.zeros(shape=[train_config['mini_size'], 1,1,110])})

 					total_num = extract_pos_examples + extract_neg_examples
 					batch_class_tenor = [1] * len(extract_pos_examples) + [0] * len(extract_neg_examples)
 					iter2 = int(np.ceil(len(total_num)) / train_config['mini_size'])
 					for index in range(iter2):

 						start = index * train_config['mini_size']
 						batch_sa = total_num[start: start+ train_config['mini_size']]
 						batch_la = batch_class_tenor[start: start+ train_config['mini_size']]
 						_, class_batch_loss = sess.run(score_op, score_loss, feed_dict={Adnetwork.input_tensor : batch_sa, Adnetwork.label_tensor: batch_la, Adnetwork.action_history_tensor:np.zeros(shape=[train_config['mini_size'], 1,1,110])})







