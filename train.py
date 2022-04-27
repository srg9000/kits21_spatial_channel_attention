import os
import boto3
# os.system('pip3 install -Iv monai==0.6.0')
# os.system('pip3 install tqdm')
# os.system('pip3 install einops')
# os.system('pip3 install nibabel')
# os.system('pip3 install -Iv pytorch-lightning==1.4.0')
# os.system('pip3 install -Iv lightning-bolts==0.3.4')
# os.system('pip3 install numpy')
# os.system('pip3 install psutil')
# os.system('pip3 install matplotlib')
# os.system('pip3 install -Iv rsa==4.5.0')
# s3 = boto3.client('s3')
# s3.download_file('<path to model/data/etc>')

best_metric_model_file = "last_model.pth"

print(list(os.listdir('./')), flush=True)

import psutil
import argparse
import json
import logging
import os
import sys
from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureType,
    Invertd,
    RandAffined,
    RandShiftIntensityd,
    Rand3DElasticd,
    RandFlipd,
    RandGaussianNoised
)
# from monai.handlers.utils import 
from monai.networks.nets import UNet, UNETR, DynUNet
from SCUnet import UNet as SCUnet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch,SmartCacheDataset
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import numpy as np

def main(root_data='kits21/data/', root_dir='resunet_checkpoints'):

    train_images=[]
    train_labels=[]
    for case in list(sorted(os.listdir(root_data))):
            train_images.append(root_data+"/"+case+"/imaging.nii.gz"),
            train_labels.append(root_data+"/"+case+"/aggregated_AND_seg.nii.gz")

    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(train_images, train_labels)
    ]
    print(len(data_dicts))
    train_files, val_files = data_dicts[28:], data_dicts[:28]
    # val_files.extend([data_dicts[151],data_dicts[156]])
    print("VAL FILES = ", len(val_files), flush=True)
    print("TRAIN FILES = ", len(train_files), flush=True)
    print("MEMORY = ", str(round(psutil.virtual_memory().total / (1024.0 **3)))+" GB", flush=True)
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=(
                2, 1.62, 1.62), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=-80, a_max=305,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(160, 160, 64),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            Rand3DElasticd(
                keys=["image", "label"],
                mode=("bilinear", "nearest"),
                prob=0.5,
                sigma_range=(5, 8),
                magnitude_range=(50, 150),
                spatial_size=(160, 160, 64),
                translate_range=(10, 10, 5),
                rotate_range=(np.pi/36,np.pi/36, np.pi),
                scale_range=(0.1, 0.1, 0.1),
                padding_mode="zeros",
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.25,
            ),
            RandGaussianNoised(keys=["image"], prob=0.25, mean=0.0, std=0.1),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=(
                2, 1.62, 1.62), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],a_min=-80, a_max=305,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    print("CREATING TRAIN DS", flush=True)
    train_ds = SmartCacheDataset(
        data=train_files, transform=train_transforms,
        cache_rate=0.1, replace_rate=0.5)
    print(len(train_ds))
    # train_ds = Dataset(data=train_files, transform=train_transforms)
    print("CREATED TRAIN DS", flush=True)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld
    # to generate 2 x 4 images for network training
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
    print("CREATED TRAIN DATALOADER", flush=True)

    val_ds = SmartCacheDataset(
        data=val_files, transform=val_transforms, cache_rate=0.1,replace_rate=0.5)
    # val_ds = Dataset(data=val_files, transform=val_transforms)
    print("CREATED VAL DS", flush=True)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)
    print("CREATED VAL DATALOADER", flush=True)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')

    print("CREATING MODEL", flush=True)
    model = SCUNet(
        dimensions=3,
        in_channels=1,
        out_channels=4,
        channels=(64, 128, 256, 512, 512),
        strides=(2, 2, 2, 2), 
    )
    # Residual Unet for comparision
    # model = UNet(
    #     dimensions=3,
    #     in_channels=1,
    #     out_channels=4,
    #     channels=(64, 128, 256, 512),
    #     strides=(2, 2, 2, 2),
    #     num_res_units=2,
    #     norm="INSTANCE",
    # ).to(device)
    
    model.load_state_dict(torch.load(best_metric_model_file,
        map_location=torch.device(device)))

    print("CREATED MODEL", flush=True)
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), 1e-4)
    dice_metric = DiceMetric(include_background=False, reduction="mean")


    max_epochs = 250
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=True, n_classes=4)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=True, n_classes=4)])

    for epoch in range(max_epochs):
        print("-" * 10, flush=True)
        print(f"epoch {epoch + 1}/{max_epochs}", flush=True)
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(
                f"{step}/{len(train_ds) // train_loader.batch_size}, "
                f"train_loss: {loss.item():.4f}", flush=True)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}", flush=True)

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    roi_size = (160,160, 64)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(
                        val_inputs, roi_size, sw_batch_size, model)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)

                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()

                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(
                        root_dir, "best_metric_model_"+str(epoch)+"_"+str(f"{metric:.4f}")+".pth"))
                    print("saved new best metric model", flush=True)
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}", flush=True
                )

    torch.save(model.state_dict(), os.path.join(
                        root_dir, "last_model.pth"))
    print(epoch_loss_values, flush=True)
    print(metric_values, flush=True)
    print(
    f"train completed, best_metric: {best_metric:.4f} "
    f"at epoch: {best_metric_epoch}", flush=True)

if __name__=='__main__':
    import os
    import boto3
    parser = argparse.ArgumentParser()
    # s3 = boto3.client('s3')
    # s3.download_file('s3://sagemaker-us-east-1-398089083258', 'requirements.txt', 'requirements.txt')
    # os.command('pip install -r requirements.txt')

    # Data, model, and output directories, environ variables for Sagemaker
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    # parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    # parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    args = parser.parse_args()
    print("ARGS = ", args.data_dir, args.model_dir, args, flush=True)
    main(args.data_dir, args.model_dir)