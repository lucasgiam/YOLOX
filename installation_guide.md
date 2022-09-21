## 0. Prerequisities

You should have the following installed on your workstation:
* Anaconda
* Git
* NVIDIA CUDA toolkit and cuDNN driver


## 1. Installation

* Launch the terminal window, create a virtual environment and activate it.

```
conda create -n YOLOX
conda activate YOLOX
```

* Install all the required packages.

```
pip install -r requirements.txt
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

## 2. Loading your own custom dataset

* This section assumes that you have already collected your image data and annnotated the images to the Pascal VOC format.

* Open ```YOLOX/datasets``` folder and create the following directory. Note that <dataset_name> is the name of the dataset, for example ```sp_ppe_voc_all_combinations```.

📁 YOLOX/
📄 ├── datasets/
📄 │   ├── VOCdevkit/
📄 │   │   ├── VOC2012/
📄 │   │   │   ├── <dataset_name>/
📄 │   │   │   │   ├── Annotations/
📄 │   │   │   │   ├── ImageSets/
📄 |   |   |   |   |   ├── Main/
📄 |   |   |   |   |   |   ├── train.txt
📄 |   |   |   |   |   |   ├── val.txt
📄 |   |   |   |   |   |   ├── test.txt
📄 |   |   |   |   |   |   ├── trainval.txt
📄 │   │   │   │   ├── JPEGImages/

* Transfer all the images (.jpg) into the ```JPEGImages``` folder and all the annotations (.xml) into the ```Annotations``` folder.

* XXX