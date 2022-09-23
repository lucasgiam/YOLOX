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


## 2. Load your own custom dataset

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
📄 │   │   │   │   ├── JPEGImages/

* Transfer all the images (.jpg) into the ```JPEGImages``` folder and all the annotations (.xml) into the ```Annotations``` folder.

* Go into ```train_test_split.py``` in the root directory and configure the root_path, seed, train_percent and val_percent, and then run the script.

```python train_test_split.py```


## 3. Download pre-trained weights

* In the root directory, enter the following command to download pre-trained weights into the ```weights``` folder. Note that you can change ```yolo_m``` to ```yolo_s``` or ```yolo_l```, depending on the size of the model that you want.

``` wget.exe -P weights https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_m.pth ```


## 4. Train the model

* Configure training parameters in the "./exps/custom/<datset_name>/<dataset_name>.py" file.

* Run the following command to initiate model training.

```
python tools/train.py -f .\exps\custom\sp_ppe_1\sp_ppe_1.py -d 2 -b 16 --fp16 -o -c .\weights\yolox_m.pth
```

* XXX