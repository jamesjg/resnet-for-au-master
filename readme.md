# Resnet for AU prediction

This repository provides an implementation of an Action Unit (AU) prediction model via ResNet. 

## 0. Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/jamesjg/resnet-for-au-master.git

conda create -n resnet-for-au python=3.8
conda activate resnet-for-au

cd resnet-for-au-master
pip install -r requirements.txt
```


The following dependencies are tested:
- Python 3.8.19
- Pytorch 1.11.0+cu113   torchvision 0.12.0+cu113

## 1. Dataset Paths
Before training or testing the model, you need to download and set up the datasets. The datasets can be downloaded from Google Drive or Baidu Drive: 

| Dataset | Google Drive | Baidu Drive |
| ------- | ------------ | ----------- |
| FEAFA    | [FEAFA]() | [FEAFA]() |
| RAF-DB    | [RAF-DB]() | [RAF-DB]() |
| AFFECTNET    | [AFFECTNET]() | [AFFECTNET]() |


## 2. Get Started

### 2.1 Training

To train the model, you need to provide the paths to the dataset, train json file, validation json file. 

The train json file and validation json file contain the image names and their corresponding AUs. They are in the following format:

```json
[
    {
        "file_name": "image1.jpg", "au":[0,0,0,0,0,0]
        },
    
]
```

To train the model, run the following command:

```bash
python train.py --data_path <path_to_dataset> --train_json_path <path_to_train_json> --val_json_path <path_to_val_json> --save_path <path_to_save>
```

### 2.2 Testing

To test the model, you need to provide the paths to the dataset, val json file, and the trained model. Then,  run the following command:

```bash
python test.py --data_path <path_to_dataset> --val_json_path <path_to_val_json> --model_path <path_to_trained_model>
```

### 2.3 Inference

To infer the AUs for a given face image datacet, you need to provide the path to the dataset and the trained model.

There are 2 kinds of image dataset for inference: "raf" and  "attack". For "raf", the face images are unaligned and you should provide bbox info txt. The format of this text file from the third line onward is as follows:
```
img_name, x1, y1, w, h
```

For "attack", the images are aligned and a json file containing image name and other labels is required. The format of this json file is as follows:
```json
[
    {
        "file_name": "image1.jpg",
        "emo": "Happy"},
]
```


To infer the AUs for "raf" kind dataset , run the following command:

```bash
python inference_v2.py --data_path <path_to_dataset> --model_path <path_to_trained_model> --dataset raf --bbox_txt_path <path_to_bbox_txt> --result_json_path <path_to_result_json>
```

To infer the AUs for "attack" kind dataset , run the following command:

```bash
python inference_v2.py --data_path <path_to_dataset> --model_path <path_to_trained_model> --dataset attack --label_json_path <path_to_val_json> --result_json_path <path_to_result_json>
```


## 3. Related Projects

This project is related to the [SCRFD](https://github.com/deepinsight/insightface/tree/master/detection/scrfd) face detection model, which can be used for preprocessing faces before performing AU prediction.

## 4. Final Model

We trained a Resnet-50  AU regression model on the FEAFA dataset (AUs in 0-1 float) and achieved an average AU mae of 0.00340. The trained model can be downloaded from [Baidu]() or [Google Drive]().

