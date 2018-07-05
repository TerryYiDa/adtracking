# tf-adnet
Tensorflow Implementation of 'Action-Decision Networks for Visual Tracking with Deep Reinforcement Learning(CVPR 2017)'

| GIF      | Description                    |
|:---------|:-------------------------------|
| ![freeman1_180202](/data/car/car.gif) | tracking |
* Green : Ground Truth, Blue : ADNet-Fast Tracking Result


## Implementations

- [x] Network Architecture

- [ ] Training Code

  - [ ] Reproduce Paper Result

- [x] Inference Code

  - [x] Converting Original Weights to Tensorflow
  
  - [x] Online Learning(finetuning)

## Run

### OTB100 Dataset

```
$ python experiment.py 
```

## References

- Action-Decision Networks for Visual Tracking with Deep Reinforcement Learning (CVPR201) : http://openaccess.thecvf.com/content_cvpr_2017/papers/Yun_Action-Decision_Networks_for_CVPR_2017_paper.pdf
- ADNet Implmentation in Matlab : https://github.com/hellbell/ADNet/

