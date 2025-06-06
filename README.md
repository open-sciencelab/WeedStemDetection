[![arxiv.org](http://img.shields.io/badge/cs.CV-arXiv%3A2502.06255-B31B1B.svg)](https://arxiv.org/pdf/2502.06255)

<div align="center">
<h1 style="text-align: center; font-size: 2.5rem; font-weight: bolders">
AAAI-2025: Towards Efficient and Intelligent Laser Weeding: Method and Dataset for Weed Stem Detection
</h1>

<font size=4>[[Paper]](https://arxiv.org/pdf/2502.06255)</font> 
</div>

## Quickstart
* Inference:
```
python detect.py --weights best_epoch.pt --source inference_image_dir --name dir_name --project project_name --conf 0.10 
```
* Train
```
python train.py --weights train_model.pt
```

## Data

* Please contact lijinzhe@pjlab.org.cn for data acquisition.

## Citation

```bibtex
@inproceedings{liu2025towards,
  title={Towards Efficient and Intelligent Laser Weeding: Method and Dataset for Weed Stem Detection},
  author={Liu, Dingning and Li, Jinzhe and Su, Haoyang and Cui, Bei and Wang, Zhihui and Yuan, Qingbo and Ouyang, Wanli and Dong, Nanqing},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={27},
  pages={28204--28212},
  year={2025}
}
```

