import numpy as np
from network.adnet import adnet
from util.get_samples import fit_image
import collections
Box = collections.namedtuple('Box', ['x', 'y', 'width', 'height'])
def get_labels(train_config, pos_boxes, gt_box):
	labels = []
	for sample in pos_boxes:
		ious = []
        for i in range(11):
            moved_box = do_action(imgwh=None, action_idx=i)
            iou = iou(moved_box, gt_box)
            ious.append(iou)
                # stop_iou = 0.93
        if ious[adnet.ACTION_IDX_STOP] > train_config['stop_iou']:
            labels.append(adnet.ACTION_IDX_STOP)
        if max(ious[:-2]) * 0.99999 <= ious[adnet.ACTION_IDX_STOP]:
            labels.append(np.argmax(ious))
            # return random.choice([i for i, x in enumerate(ious) if x >= max(ious)])
        labels.append(np.argmax(ious[:-2]))

    return labels


def iou(new_box, gt_box):
	if isinstance(new_box, Box) and if isinstance(gt_box, bounding_box):
		x1 = new_box.x
		y1 = new_box.y
		x2 = x1 + new_box.width
		y2 = y1 + new_box.height

		box_x1 = gt_box.x
		box_y1 = gt_box.x
		box_x2 = box_x1 + gt_box.width
		box_y2 = box_x2 + gt_box.height
	else:
		raise

	xa = np.maximum(x1, box_x1)
	ya = np.maximum(y1, box_y1)
	xb = np.minimum(x2,  box_x2)
	yb = np.minimum(y2,  box_y2)

	if xa > xb or ya > yb:
		return 0.0

	interarea = (xb - xa) * (yb - ya)
	boxAarea = new_box.width * new_box.height
	boxBarea = gt_box.width * gt_box.height

	iou = interarea / float(boxAarea + boxBarea - interarea)
	return iou

def do_action(self, imgwh, action_idx):
	action_ratios = tuple([0.03,0.03,0.03,0.03])

	if action_idx < 8:
		deltas_xy = self.wh * action_ratios[:2]
		deltas_xy.max(1)
		actual_deltas = adnet.ACTIONS[action_idx][:2] * (deltas_xy.x, deltas_xy.y)
		moved_xy = self.xy + actual_deltas
		new_box = Box(moved_xy.x, moved_xy.y, self.wh.x, self.wh.y)
	elif action_idx == 8:
		new_box = Box(self.xy.x, self.xy.y, self.wh.x, self.wh.y)
	else:
		deltas_wh = self.wh * action_ratios[2:]
		deltas_wh.max(2)
		deltas_wh_scaled = adnet.ACTIONS[action_idx][2:] * (deltas_wh.x, deltas_wh.y)
		moved_xy = self.xy + -1 * deltas_wh_scaled / 2
		moved_wh = self.wh + deltas_wh_scaled

		new_box = Box(moved_xy.x, moved_xy.y, moved_wh.x, moved_wh.y)

	if imgwh:
		fit_image(imgwh, new_box)
	return new_box