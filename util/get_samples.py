 import tensorflow as tf
import collections
Recangle = collections.namedtuple('Recangle', ['x', 'y', 'width', 'height'])

def fit_image(self, imgwh, box):
    box.x = max(0, box.x)
    box.y = max(0, box.y)

    box.width = max(10, min(box.width, imgwh[0] - 10))
    box.height = max(10, min(box.height, imgwh[1] - 10))
    box.width = min(box.x, imgwh[0] - box.x)
    box.height = min(box.y, imgwh[1] - box.y)

 def gen_noise_samples(self, img, box, noise_type, num, **kwargs):
    imgwh =  tf.shape(img)[0:2]
    center_xy = np.array([box.x, box.y]) + np.array([box.width, box.height]) * 0.5
    mean_wh = sum([box.x, box.y]) / 2.0

    gaussian_translation_f = kwargs.get('gaussian_translation_f', 0.1)
    uniform_translation_f = kwargs.get('uniform_translation_f', 1)
    uniform_scale_f = kwargs.get('uniform_scale_f', 10)

    samples = []
    if noise_type == 'whole':
        grid_x = range(self.box.height // 2, imgwh[0] - self.box.width // 2, self.box.width // 5)
        grid_y = range(self.box.height // 2, imgwh[1] - self.box.height // 2, self.box.height // 5)
        samples_tmp = []
        for dx, dy, ds in itertools.product(grid_x, grid_y, range(-5, 5, 1)):
            boundiing_box = Recangle(dx, dy, self.box.width*(1.05**ds), self.box.height*(1.05**ds))
            fit_image(imgwh, boundiing_box)
            samples_tmp.append(boundiing_box)

        for _ in range(num):
            samples.append(random.choice(samples_tmp))
    else:
        for _ in range(num):
            if noise_type == 'gaussian':
                dx = gaussian_translation_f * mean_wh * minmax(0.5 * random.normalvariate(0, 1), -1, 1)
                dy = gaussian_translation_f * mean_wh * minmax(0.5 * random.normalvariate(0, 1), -1, 1)
                dwh = 1.05 ** (3 * minmax(0.5 * random.normalvariate(0, 1), -1, 1))
            elif noise_type == 'uniform':
                dx = uniform_translation_f * mean_wh * random.uniform(-1.0, 1.0)
                dy = uniform_translation_f * mean_wh * random.uniform(-1.0, 1.0)
                dwh = 1.05 ** (uniform_scale_f * random.uniform(-1.0, 1.0))
            else:
                raise
            new_cxy = center_xy + (dx, dy)
            new_wh = np.array([box.width, box.height]) * dwh
            box = Recangle(new_cxy[0] - new_wh[0] / 2.0, new_cxy[1] - new_wh[1] / 2.0, new_wh[0], new_wh[1])
            fit_image(imgwh, box)
            samples.append(box)

    return samples


def get_posneg_samples(self, img, groundbox, pos_size, neg_size, use_whole=True, **kwargs):
    pos_thresh = kwargs.get('pos_thresh', 0.7) 
    neg_hresh = kwargs.get('neg_thresh', 0.3)

    gaussian_samples = self.gen_noise_samples(img, groundbox, 'gaussian', pos_size * 2, kwargs=kwargs)
    gaussian_samples = [x for x in gaussian_samples if x.iou(self) > pos_thresh]

    uniform_samples = self.gen_noise_samples(img, groundbox,'uniform', neg_size if use_whole else neg_size*2, kwargs=kwargs)
    uniform_samples = [x for x in uniform_samples if x.iou(self) < neg_thresh]

    if use_whole:
        whole_samples = self.gen_noise_samples(img, groundbox, 'whole', neg_size, kwargs=kwargs)
        whole_samples = [x for x in whole_samples if x.iou(self) < neg_thresh]
    else:
        whole_samples = []

    pos_samples = []
    for _ in range(pos_size):
        pos_samples.append(random.choice(gaussian_samples))

    neg_candidates = uniform_samples + whole_samples
    neg_samples = []
    for _ in range(neg_size):
        neg_samples.append(random.choice(neg_candidates))
    return pos_samples, neg_samples
