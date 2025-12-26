# InCNNTD
A lightweight, Computer vision and deep learning experiments.

INSTALLATION
- Python 3.8+
- TensorFlow / Keras
- OpenCV (Headless)
- NumPy
- Pillow (PIL)
- Matplotlib
- Google Colab


pip install -q opencv-python-headless==4.7.0.72

IMPORTS
import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from google.colab import files
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import cv2

GETTING STARTED
1. Open Google Colab
2. Install dependencies
3. Add imports
4. Load dataset
5. Train and evaluate model



## Datasets

This research uses custom curated dataset NHITSNEAI Dataset  and publicly available benchmark datasets to evaluate traffic sign recognition and detection under diverse real-world conditions.

---

### 1. NHITSNEAI Dataset (Custom Indian Traffic Sign Dataset)

**Link:**  
https://

The **NHITSNEAI dataset** is a custom traffic sign dataset manually collected from the Indian road network. It captures real-world driving scenarios commonly observed in India, including variations in illumination, scale, orientation, background complexity, and partial occlusions.  
This dataset is designed to complement publicly available benchmarks by incorporating region-specific traffic signs and visual characteristics.



### 2. German Traffic Sign Recognition Benchmark (GTSRB)

**Official Website:**  
https://benchmark.ini.rub.de/gtsrb_dataset.html

#### Overview
The **German Traffic Sign Recognition Benchmark (GTSRB)** is a widely used public benchmark dataset for traffic sign classification.

- Single-image, multi-class classification problem  
- More than **40 traffic sign classes**  
- Over **50,000 images**  
- Large-scale, lifelike real-world dataset  
- Reliable ground-truth annotations using semi-automatic methods  
- Each physical traffic sign instance appears **only once** in the dataset  

This dataset is extensively used for benchmarking computer vision and deep learning models in intelligent transportation systems.

### 3. CCTSDB2021 (Chinese Traffic Sign Detection Benchmark)

**Weblink:**  
https://github.com/csust7zhangjm/CCTSDB2021

#### Dataset Description
The **CCTSDB2021** dataset is a comprehensive benchmark for traffic sign detection in real-world traffic environments.

- Contains **17,856 images** in the training set and positive sample test set  
- Traffic signs are categorized into three groups:
  - Mandatory  
  - Prohibitory  
  - Warning  

This dataset is widely used for evaluating real-time and lightweight traffic sign detection models.


#### Citation
If you use the GTSRB dataset, please cite:

> J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel,  
> *The German Traffic Sign Recognition Benchmark: A multi-class classification competition*,  
> Proceedings of the IEEE International Joint Conference on Neural Networks (IJCNN),  
> pp. 1453–1460, 2011.

```bibtex
@inproceedings{Stallkamp-IJCNN-2011,
    author    = {Johannes Stallkamp and Marc Schlipsing and Jan Salmen and Christian Igel},
    booktitle = {IEEE International Joint Conference on Neural Networks},
    title     = {The German Traffic Sign Recognition Benchmark: A multi-class classification competition},
    year      = {2011},
    pages     = {1453--1460}
}





#### Citations
If you use the CCTSDB2021 dataset, please cite the following references:
@article{Zhang2024AnchorFreeTSD,
  author  = {Zhang, Jianming and Lv, Yaru and Tao, Jiajun and Huang, Fengxiang and Zhang, Jin},
  title   = {A Robust Real-Time Anchor-Free Traffic Sign Detector with One-Level Feature},
  journal = {IEEE Transactions on Emerging Topics in Computational Intelligence},
  year    = {2024},
  volume  = {8},
  number  = {2},
  pages   = {1437--1451},
  doi     = {10.1109/TETCI.2024.3349464}
}

@article{Zhang2022CCTSDB2021,
  author  = {Zhang, Jianming and Zou, Xin and Kuang, Li-Dan and Wang, Jin and Sherratt, R. Simon and Yu, Xiaofeng},
  title   = {CCTSDB 2021: A More Comprehensive Traffic Sign Detection Benchmark},
  journal = {Human-centric Computing and Information Sciences},
  year    = {2022},
  volume  = {12},
  pages   = {23},
  doi     = {10.22967/HCIS.2022.12.023}
}

@article{Zhang2022ReYOLO,
  author  = {Zhang, Jianming and Zheng, Zhuofan and Xie, Xianding and Gui, Yan and Kim, Gwang-Jun},
  title   = {ReYOLO: A Traffic Sign Detector Based on Network Reparameterization and Features Adaptive Weighting},
  journal = {Journal of Ambient Intelligence and Smart Environments},
  year    = {2022},
  volume  = {14},
  number  = {4},
  pages   = {317--334},
  doi     = {10.3233/AIS-220038}
}

@article{Zhang2022MultiscaleTSD,
  author  = {Zhang, Jianming and Ye, Zi and Jin, Xiaokang and Wang, Jin and Zhang, Jin},
  title   = {Real-Time Traffic Sign Detection Based on Multiscale Attention and Spatial Information Aggregator},
  journal = {Journal of Real-Time Image Processing},
  year    = {2022},
  volume  = {19},
  number  = {6},
  pages   = {1155--1167},
  doi     = {10.1007/s11554-022-01252-w}
}

@article{Zhang2020LightweightTSC,
  author  = {Zhang, Jianming and Wang, Wei and Lu, Chaoquan and Wang, Jin and Sangaiah, Arun Kumar},
  title   = {Lightweight Deep Network for Traffic Sign Classification},
  journal = {Annals of Telecommunications},
  year    = {2020},
  volume  = {75},
  number  = {7--8},
  pages   = {369--379},
  doi     = {10.1007/s12243-019-00731-9}
}

@article{Zhang2020CascadedRCNN,
  author  = {Zhang, Jianming and Xie, Zhipeng and Sun, Juan and Zou, Xin and Wang, Jin},
  title   = {A Cascaded {R-CNN} with Multiscale Attention and Imbalanced Samples for Traffic Sign Detection},
  journal = {IEEE Access},
  year    = {2020},
  volume  = {8},
  pages   = {29742--29754},
  doi     = {10.1109/ACCESS.2020.2972338}
}

