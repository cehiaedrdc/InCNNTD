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



### CCTSDB2021 (Chinese Traffic Sign Detection Benchmark)

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

#### Citations
If you use the CCTSDB2021 dataset, please cite the following references:

1. Zhang, J., Lv, Y., Tao, J., Huang, F., & Zhang, J.,  
   *A robust real-time anchor-free traffic sign detector with one-level feature*,  
   **IEEE Transactions on Emerging Topics in Computational Intelligence**, 8(2), 1437–1451, 2024.  
   https://doi.org/10.1109/TETCI.2024.3349464

2. Zhang, J., Zou, X., Kuang, L.-D., Wang, J., Sherratt, R. S., & Yu, X.,  
   *CCTSDB 2021: A more comprehensive traffic sign detection benchmark*,  
   **Human-centric Computing and Information Sciences**, 12, Article 23, 2022.  
   https://doi.org/10.22967/HCIS.2022.12.023

3. Zhang, J., Zheng, Z., Xie, X., Gui, Y., & Kim, G.-J.,  
   *ReYOLO: A traffic sign detector based on network reparameterization and features adaptive weighting*,  
   **Journal of Ambient Intelligence and Smart Environments**, 14(4), 317–334, 2022.  
   https://doi.org/10.3233/AIS-220038

4. Zhang, J., Ye, Z., Jin, X., Wang, J., & Zhang, J.,  
   *Real-time traffic sign detection based on multiscale attention and spatial information aggregator*,  
   **Journal of Real-Time Image Processing**, 19(6), 1155–1167, 2022.  
   https://doi.org/10.1007/s11554-022-01252-w

5. Zhang, J., Wang, W., Lu, C., Wang, J., & Sangaiah, A. K.,  
   *Lightweight deep network for traffic sign classification*,  
   **Annals of Telecommunications**, 75(7–8), 369–379, 2020.  
   https://doi.org/10.1007/s12243-019-00731-9

6. Zhang, J., Xie, Z., Sun, J., Zou, X., & Wang, J.,  
   *A cascaded R-CNN with multiscale attention and imbalanced samples for traffic sign detection*,  
   **IEEE Access**, 8, 29742–29754, 2020.  
   https://doi.org/10.1109/ACCESS.2020.2972338
