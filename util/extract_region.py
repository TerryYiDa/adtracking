import cv2
def extract_region(train_config, img, bbox):
	xy = [bbox.x, bbox.y]
	wh = [bbox.width, bbox.height]
	xy = np.array(xy)
	wh = np.array(wh)

    xy_center = xy + wh * 0.5

    wh = wh * train_config['roi_zoom']
    xy = xy_center - wh * 0.5
    xy[0] = max(xy[0], 0)
    xy[1] = max(xy[1], 0)

    # crop and resize
    crop = img[xy[1]:xy[1]+wh[1], xy[0]:xy[0]+wh[0], :]
    resize = cv2.resize(crop, (train_config['image_size'], train_config['image_size']))
    return resize