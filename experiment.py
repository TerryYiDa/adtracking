from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURRENT_DIR, '..'))

from configuration import LOG_DIR
from train_siamese_model import ex

if __name__ == '__main__':
  RUN_NAME = 'ADnet-3s-color-scratch'
  ex.run(config_updates={'train_config': {'train_dir': osp.join(LOG_DIR, 'track_model_checkpoints', RUN_NAME), },
                         'track_config': {'log_dir': osp.join(LOG_DIR, 'track_model_inference', RUN_NAME), }
                         },
         options={'--name': RUN_NAME,
                  '--force': True,
                  '--enforce_clean': False,
                  })