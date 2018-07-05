import tensorflow as tf
from network.adnet import adnet
import numpy as np
import os
from util.get_labels import iou
from util.extract_region import extract_region
from util.util import auto_select_gpu, chunker, random_idxs, choices_by_idx
from tracking import tracking
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

	 			RL_step =  train_config['RL_step']
	 			start = [index for index in range(len(ground_box) - 10)]
	 			end = [inded for index in range(len(ground_box), start=10)]
	 			shuff_start = np.random.shuffle(start)
	 			start = start[shuff_start]
	 			end = end[shuff_start]
	 			num_train_clips = min(train_config['num_batch'], len(start))
	 			action_labels = []
	 			action_labels_pos = []
	 			action_labels_neg = []
	 			out_actions = []
	 			out_scores = []
	 			for clipinx in range(num_train_clips):
	 				franmstart = start[clipinx]
	 				franmend = end[clipinx]
	 				curr_box = ground_box[clipinx]
	 				action_history =  np.zeros(shape = [11, 1])
	 				while(franmstart<franmend):
	 					img_path = tf.read_file(videos_len[franmstart])
	 					img = tf.image.decode_jpeg(img_path, channel =3, dct_method = 'INTEGER_ACCURATE')
	 					curr_gt_bbox = ground_box[franmstart]
	 					num_actions, pred_curr, out_actions, out_scores = tracking(sess, train_config, img, curr_gt_bbox)
	 					action_labels.append(num_actions)
	 					out_actions.append(out_actions)
	 					out_scores.append(out_scores)
	 					curr_gt_bbox = pred_curr
	 					franmstart +=1

	 				if iou(curr_gt_bbox, curr_box) > train_config['pos_thresh'] >0.7:
	 					action_labels_pos.append(action_labels)

	 				esle:
	 					action_labels_neg.append(action_labels_neg)


	 			num_pos = len(action_labels_pos)
        		num_neg = len(action_labels_neg)
	        	train_pos_cnt = 0
		        train_pos = []
		        train_neg_cnt = 0
		        train_neg = []
		        batch_size = train_config['mini_batch']
		        if num_pos > batch_size/2:

		        	remian = train_config['mini_batch'] * num_train_clips
		        	while(remain>0):
		        		if train_pos_cnt == 0:
		        			train_pos_list = np.random.shuffle(range(num_pos))

		        		train_pos += train_pos_list[train_pos_cnt :min(size(train_pos_list), train_pos_cnt + remian)]

		        		train_pos_cnt = min(len(train_pos_list), train_pos_cnt + remian)
		        		train_pos_cnt = np.mod(train_pos_cnt, len(train_pos_list))
		        		remian = train_config['mini_batch'] * num_train_clips - len(train_pos)


		        if num_pos > batch_size/2:

		        	remian = train_config['mini_batch'] * num_train_clips
		        	while(remain>0):
		        		if train_neg_cnt == 0:
		        			train_neg_list = np.random.shuffle(range(num_neg))

		        		train_pos += train_pos_list[train_pos_cnt :min(size(train_neg_list), train_neg_cnt + remian)]

		        		train_neg_cnt = min(len(train_neg_list), train_neg_cnt + remian)
		        		train_neg_cnt = np.mod(train_neg_cnt, len(train_neg_list))
		        		remian = train_config['mini_batch'] * num_train_clips - len(train_pos)

		        with tf.name_scope('loss'):
			        for batchix in num_train_clips:
			        	batch = batchix * train_config['mini_batch']
			        	batch_index = train_pos[batch: batch + train_config['mini_batch']]
			        	pos_labels = action_labels_pos[batch_index]

			        	out_actions_batch = out_actions[batch_index]
			        	neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out_actions, labels=pos_labels)   # this is negative log of chosen action
            			slef.loss = tf.reduce_mean(neg_log_prob * self.tf_vt) 
            	with tf.name_scope('train'):
            		self.train_op = tf.train.AdamOptimizer(train_config['lr']).minimize(self.loss)








	 			






