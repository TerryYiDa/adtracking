train_config = {
	'image_size': 112,

	'roi_zoom': 1.4,

	'lr_config': {'policy': 'exponential',
                'initial_lr': 0.01,
                'num_epochs_per_decay': 1,
                'lr_decay_factor': 0.8685113737513527,
                'staircase': True, },

     'epoch': 30,

     'train_data_config': {'input_imdb': 'data/train_imdb.pickle',
     						'preprocessing_name': 'siamese_fc_color',
     						'num_examples_per_epoch': 5.32e4,
     						'epoch': 50,
     						'batch_size': 8,
                        	'max_frame_dist': 100,  # Maximum distance between any two random frames draw from videos.
                        	'prefetch_threads': 4,
                        	'prefetch_capacity': 15 * 8, },


     'train_data_config': {'input_imdb': 'data/train_imdb.pickle',
                        'preprocessing_name': 'siamese_fc_color',
                        'num_examples_per_epoch': 5.32e4,
                        'epoch': 50,
                        'batch_size': 8,
                        'max_frame_dist': 100,  # Maximum distance between any two random frames draw from videos.
                        'prefetch_threads': 4,
                        'prefetch_capacity': 15 * 8, },

      'optimizer_config': {'optimizer': 'MOMENTUM',  # SGD and MOMENTUM are supported
                       'momentum': 0.9,
                       'use_nesterov': False, },

}