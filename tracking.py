 import numpy as np
 from network.adnet import  adnet
 from util.extract_region import extract_region
 from util.util import onehot_flatten
 def tracking(self, sess, img, curr_bbox):
        Adnetwork = adnet(train_config, is_training = True)
        Adnetwork.build()
        self.iteration += 1
        is_tracked = True
        # boxes = []  
        self.latest_score = -1
        # num_action = 20
        num_action = train_config['num_action']
        action_num =  np.zeros((20, 1))
        for track_i in range(num_action):
            patch = extract_region(img, curr_bbox)

            # forward with image & action history = 10
            actions, classes, out_actions, out_scores = sess.run(
                [Adnetwork.layer_actions, Adnetwork.layer_scores, Adnetwork.out_actions, Adnetwork.out_scores],
                feed_dict={
                    Adnetwork.input_tensor: [patch],
                    Adnetwork.action_history_tensor: [onehot_flatten(self.action_histories)],
                    tensor_is_training: False
                }
            )

            latest_score = classes[0][1]
            # thresh_fail = 0.5
            if latest_score < ADNetConf.g()['predict']['thresh_fail']:
                is_tracked = False
                self.action_histories_old = np.copy(self.action_histories)
                self.action_histories = np.insert(self.action_histories, 0, 12)[:-1]
                break
            else:
                self.failed_cnt = 0
            self.latest_score = latest_score

            # move box
            action_idx = np.argmax(actions[0])
            num_action[track_i,:] = action_idx
            self.action_histories = np.insert(self.action_histories, 0, action_idx)[:-1]
            prev_bbox = curr_bbox
            curr_bbox = curr_bbox.do_action(self.imgwh, action_idx)
            if action_idx != ADNetwork.ACTION_IDX_STOP:
                if prev_bbox == curr_bbox:
                    print('action idx', action_idx)
                    print(prev_bbox)
                    print(curr_bbox)
                    raise Exception('box not moved.')

            # oscillation check
            if action_idx != ADNetwork.ACTION_IDX_STOP and curr_bbox in boxes:
                action_idx = ADNetwork.ACTION_IDX_STOP

            if action_idx == ADNetwork.ACTION_IDX_STOP:
                break

            # boxes.append(curr_bbox)
        return num_action, curr_bbox, 