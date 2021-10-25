# Bird_classification

This repository is the implementation of CodaLab competitions [bird classification](https://reurl.cc/pxr8YQ). 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s), run this command:

```train
python train.py --data_path <train_data_path> --classes_path <classes.txt> --training_labels_path <training_labels.txt> --batch_size 8 --lr 2e-4 --pretrain_model_path <pretrain_model_path> --output_foloder <model output path> --epochs 100
```

* scheduler use `ReduceLROnPlateau`
* optimizer  use `SGD`, momentum = 0.9, weight_decay = 1e-4
* Data augmentation(RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomRotation, GaussianBlur)

## Evaluation

To evaluate my model, run:

```eval
python eval.py --data_path <eval_data_path> --classes_path <classes.txt> --training_labels_path <training_labels.txt> --model_path <model_path>
```

## Reproduceing Submission

```inference
python inference.py --data_path <test_image_path> --classes_path <classes.txt> --test_filename_path <testing_img_order.txt> --model_path <model_path>
```
>📋 Will output `answer.txt`
## Pre-trained Models

You can download pretrained models here:

- [ImageNet-21K Pretraining for the Masses
](https://github.com/Alibaba-MIIL/ImageNet21K) trained on ImageNet.

- Or use timm to load the model

```
model = timm.create_model('vit_base_patch16_224_miil_in21k', pretrained=True)
```

## Results

Our model achieves the following performance on :

### Image Classification

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My best model      |     82.657%     |      NA        |
