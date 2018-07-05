import logging
try:
  import pynvml  # nvidia-ml provides utility for NVIDIA management

  HAS_NVML = True
except:
  HAS_NVML = False
def auto_select_gpu():
  """Select gpu which has largest free memory"""
  if HAS_NVML:
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    largest_free_mem = 0
    largest_free_idx = 0
    for i in range(deviceCount):
      handle = pynvml.nvmlDeviceGetHandleByIndex(i)
      info = pynvml.nvmlDeviceGetMemoryInfo(handle)
      if info.free > largest_free_mem:
        largest_free_mem = info.free
        largest_free_idx = i
    pynvml.nvmlShutdown()
    largest_free_mem = largest_free_mem / 1024. / 1024.  # Convert to MB

    idx_to_gpu_id = {}
    for i in range(deviceCount):
      idx_to_gpu_id[i] = '{}'.format(i)

    gpu_id = idx_to_gpu_id[largest_free_idx]
    logging.info('Using largest free memory GPU {} with free memory {}MB'.format(gpu_id, largest_free_mem))
    return gpu_id
  else:
    logging.info('nvidia-ml-py is not installed, automatically select gpu is disabled!')
    return '0'


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def random_idxs(max, k):
    if k >= max:
        return [random.randint(0, max - 1) for _ in range(k)]
    else:
        l = list(range(max))
        random.shuffle(l)
        return l[:k]

def choices_by_idx(seq, idxs):
    return [seq[x] for x in idxs]


def onehot(idxs):
    a = np.zeros(shape=(1, len(idxs), 11), dtype=np.int8)
    # a[0, np.arange(len(idxs)), idxs] = 1
    for i, idx in enumerate(idxs):
        if idx >= 12 or idx < 0:
            continue
        a[0, i, idx] = 1
    return a


def onehot_flatten(idxs):
    a = onehot(idxs)
    a = a.reshape((1, 1, a.shape[1]*a.shape[2]))
    return a