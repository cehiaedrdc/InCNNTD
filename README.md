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

@article{Zhang2022CCTSDB2021,
  author  = {Zhang, Jianming and Zou, Xin and Kuang, Li-Dan and Wang, Jin and Sherratt, R. Simon and Yu, Xiaofeng},
  title   = {CCTSDB 2021: A More Comprehensive Traffic Sign Detection Benchmark},
  journal = {Human-centric Computing and Information Sciences},
  year    = {2022},
  volume  = {12},
  pages   = {23},
  doi     = {10.22967/HCIS.2022.12.023}
}


