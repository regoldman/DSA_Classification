# DSA Classification

## System Requirements
The script is tested on a machine with an NVidia A100 40 GB GPU with:

- `Ubuntu 20.04.5` | `Python 3.9.12` | `CUDA 11.6` 
- `Pytorch 2.0.1` | `Torchvision 0.15.2`
- `MONAI 1.2.0`

## DSA Mode Classification
### Data
Script expects data summarized in a .csv file with the following columns:
1. image_file_path: Absolute path to image file
1. label: Text label ('aorta', 'celiac trunk', etc.)
1. numeric_label: Integer from 0 to the number of labels-1
1. diagnostic: Binary: 1: "Key" Image, 0: Non-"key" image
1. train_val_test: label for data split ('train','val','test')   

Individual images in a DSA sequence are expected to be listed sequentially
### Examples

Display all possible options

```bash
python -u DSA_Mode_classification.py -h
```


#### Train
Training a model assuming a csv file at /data/
```bash
python -u DSA_Mode_classification.py \
    --model='resnet50' \
    --optim_lr=1e-5 \
    --best_model='best_model.pt' \
    --epochs=50 \
    --batch_size=100 \
    --image_DB_filepath=/data/DB.csv \
    --key_images_only \
    --logdir=./logs
```

#### Test
Testing a model assuming a csv file at /data/

```bash
python -u DSA_Mode_classification.py \
    --model='resnet50' \
    --test \
    --checkpoint='./logs/best_model.pt' \
    --batch_size=100 \
    --image_DB_filepath=/data/DB.csv \
    --key_images_only \
    --logdir=./logs
```

## DSA MIL Classification

### Data
Script expects data summarized in a .json file with the following structure:
JSON of the formated per [MONAI MIL Model](https://github.com/) `[B, N, C, H, W]`

```
{
    "training": dataset_train,
    "validation": dataset_valid,
    "testing": dataset_test,
}
```

where `dataset_train` is a list as follows: `dataset_train = [ seq_001, seq_002, seq_003,..., seq_p]`

where `dataset_valid` is a list as follows: `dataset_valid = [ seq_p+1, seq_p+2,..., seq_q]`

where `dataset_test` is a list as follows: `dataset_test = [ seq_q+1, seq_q+2, ...,seq_r]`

where `seq_xxx` is a list of dicts as follows:
``` 
seq_001 = [{
            'image': seq_001_001,
            'label': 1
            },
            {
            'image': seq_001_002,
            'label': 1
            },
            {
            'image': seq_001_003,
            'label': 1
            },...
            {
            'image': seq_001_N,
            'label': 1
            }
            ]
```

### Examples
Display all possible options

```bash
python -u DSA_MIL_classification.py -h
```

#### Train

```bash
python -u DSA_MIL_classification.py \
    --dataset_json=/data/DSA_sequence_data.json \
    --amp \
    --mil_mode=att_trans \
    --batch_size=8 \
    --epochs=50 \
    --num_classes=6 \
    --logdir=./logs
```

#### Test
```bash
python -u ./Sequence_Classification/DSA_sequence_classification.py \
    --dataset_json=/data/DSA_sequence_data.json \
    --amp \
    --mil_mode=att_trans \
    --checkpoint=./logs/model_best.pt \
    --test \
    --num_classes=6 \
    --logdir=./logs

```