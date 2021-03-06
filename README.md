# PF-Net: Pulmonary Fibrosis Segmentation Network with Multi-Scale Guided Dense Attention
This repository provides source code of PF-Net for pulmonary firbrosis segmentation proposed published on IEEE TMI 2021. If you use this code, please cite the following paper:

* [1] G. Wang, S. Zhai et al. [Semi-Supervised Segmentation of Radiation-Induced Pulmonary Fibrosis from Lung CT Scans with Multi-Scale Guided Dense Attention][paper_link], IEEE Transactions on Medical Imaging, 2021.

![pfnet_result](./picture/image_seg.png)

![pf_net](./picture/pfnet.png)
The structure of PF-Net. It combines 2D and 3D convolutions to deal with images with anisotropic resolution. For example, the in-plane resolution is around 4 times of through-plane resolution in our dataset, and we use 2D convolutions for the first two levels and 3D convolutions for the other levels in the encoder. Multi-Scale Guided Dense Attention is introduced in the decoder to deal with lesions with various positions, sizes and shapes. 

# Requirements
* [Pytorch][torch_link] version >=1.0.1.
* [PyMIC][pymic_link], a Pytorch-based toolkit for medical image computing. Version 0.2.4 is required. Install it by `pip install PYMIC==0.2.4`.
* Some basic python packages such as Numpy, Pandas, SimpleITK.

[torch_link]:https://pytorch.org
[pymic_link]:https://github.com/HiLab-git/PyMIC
[paper_link]:https://ieeexplore.ieee.org/document/9558828

# Train and Test
* Prepare your dataset and write .csv files for training, validation and testing. See `config/data_train.csv` for example.
* Edit `config/pfnet.csv`, set the data root and csv files according to your computer. You may also need to set `train_transform` and `test_transform` based on the preprocess strategies required by your dataset. 
* Run the following commands for training and inference:
```
python net_run.py train config/pfnet.cfg
python net_run.py test config/pfnet.cfg
```


