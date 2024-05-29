import numpy as np
import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from monai.config import print_config
from monai.data import DataLoader,Dataset,decollate_batch
from monai.networks import eval_mode
from monai.networks.nets import resnet34
from monai.metrics import ROCAUCMetric
from monai.transforms import Activations, AsDiscrete, LoadImageD, EnsureChannelFirstD, ScaleIntensityRangeD, RandRotate90D,SpacingD,OrientationD, ResizeD,ScaleIntensityD, Compose
import random

### Directory paths for training and validation data
data_dir_train = r"D:\CTdata\MaskCrop\bg\A\Train"
data_dir_valid = r"D:\CTdata\MaskCrop\bg\A\Validate"

### train data
class_names_train = sorted(x for x in os.listdir(data_dir_train) if os.path.isdir(os.path.join(data_dir_train, x)))

### Load image paths for training
num_class_train = len(class_names_train)
image_files_train = [
    [os.path.join(data_dir_train, class_names_train[i], x) for x in os.listdir(os.path.join(data_dir_train, class_names_train[i]))]
    for i in range(num_class_train)
]

### Load number of images for each class in training set
num_each_train = [len(image_files_train[i]) for i in range(num_class_train)]
image_files_list_train = []
image_class_train = []
for i in range(num_class_train):
    image_files_list_train.extend(image_files_train[i])
    image_class_train.extend([i] * num_each_train[i])
image_class_train = np.array(image_class_train, dtype=np.int64)

### Shuffle training data
random.seed(2023)
templist_train = [i for i in zip(image_files_list_train, image_class_train)]
random.shuffle(templist_train)
image_files_list_train[:], image_class_train[:] = zip(*templist_train)
print(f"Total image count: {len(image_files_list_train)}")
print(f"Label names: {class_names_train}")
print(f"Label counts: {num_each_train}")

### validation data
class_names_valid = sorted(x for x in os.listdir(data_dir_valid) if os.path.isdir(os.path.join(data_dir_valid, x)))

### Load image paths for validation
num_class_valid = len(class_names_valid)
image_files_valid = [
    [os.path.join(data_dir_valid, class_names_valid[i], x ) for x in os.listdir(os.path.join(data_dir_valid, class_names_valid[i]))]
    for i in range(num_class_valid)
]

### Load number of images for each class in validation set
num_each_valid = [len(image_files_valid[i]) for i in range(num_class_valid)]
image_files_list_valid = []
image_class_valid = []
for i in range(num_class_valid):
    image_files_list_valid.extend(image_files_valid[i])
    image_class_valid.extend([i] * num_each_valid[i])
image_class_valid = np.array(image_class_valid, dtype=np.int64)

### Shuffle validation data
random.seed(2023)
templist_valid = [i for i in zip(image_files_list_valid, image_class_valid)]
random.shuffle(templist_valid)
image_files_list_valid[:], image_class_valid[:] = zip(*templist_valid)
print(f"Total image count: {len(image_files_list_valid)}")
print(f"Label names: {class_names_valid}")
print(f"Label counts: {num_each_valid}")

num_class = 4

### Define transformations for training and validation data
train_transform = Compose([
    LoadImageD(keys="image", image_only=True),
    EnsureChannelFirstD(keys="image"),
    ScaleIntensityRangeD(
        keys="image",
        a_min=-230,
        a_max=290,
        b_min=0.0,
        b_max=1.0,
        clip=True,
    ),
    OrientationD(keys="image", axcodes="RAS"),
    SpacingD(keys="image", pixdim=(1, 1, 1), mode="bilinear"),
    ResizeD(keys="image",spatial_size=(64,64,64)),
])

val_transform = Compose([
    LoadImageD(keys="image", image_only=True),
    EnsureChannelFirstD(keys="image"),
    ScaleIntensityRangeD(
        keys="image",
        a_min=-230,
        a_max=290,
        b_min=0.0,
        b_max=1.0,
        clip=True,
    ),
    OrientationD(keys="image", axcodes="RAS"),
    SpacingD(keys="image", pixdim=(1, 1, 1), mode="bilinear"),
    ResizeD(keys="image", spatial_size=(64, 64, 64)),
])

post_pred = Compose([Activations(softmax=True)])
post_label = Compose([AsDiscrete(to_onehot=num_class)])

### Create Dataset objects for training and validation
train_files = [{"image": img, "label": label} for img, label in zip(image_files_list_train, image_class_train)]
val_files = [{"image": img, "label": label} for img, label in zip(image_files_list_valid, image_class_valid)]

tra_ds = Dataset(data=train_files, transform=train_transform)
tra_loader = DataLoader(tra_ds, batch_size=24, num_workers=3, pin_memory=torch.cuda.is_available())
val_ds = Dataset(data=val_files, transform=val_transform)
val_loader = DataLoader(val_ds, batch_size=24, num_workers=3, pin_memory=torch.cuda.is_available())

writer = SummaryWriter()
def train(model, tra_loader, val_loader, loss_function, optimizer, num_epochs, best_acc_metric = -1, best_auc_metric = -1, best_metric_epoch = -1, val_interval = 2):
    for epoch in range(num_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{num_epochs}")
        model.train()
        epoch_loss = 0
        step = 0

        # Training loop
        for batch_data in tra_loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(tra_ds) // tra_loader.batch_size
            #print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

        epoch_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # Validation loop
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                # Compute validation metrics
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                for val_data in val_loader:
                    val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                    y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)

                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)
                y_onehot = [post_label(i) for i in decollate_batch(y, detach=False)]
                y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]
                auc_metric(y_pred_act, y_onehot)
                auc_result = auc_metric.aggregate()
                auc_metric.reset()
                del y_pred_act, y_onehot

                # Update best model and metrics
                if acc_metric > best_acc_metric:
                    best_acc_metric = acc_metric
                    best_acc_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), "best_acc_metric_model_classification3d_dict.pth")
                    print("saved new best acc metric model")
                if auc_result > best_auc_metric:
                    best_auc_metric = auc_result
                    best_auc_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), "best_auc_metric_model_classification3d_dict.pth")
                    print("saved new best auc metric model")
                print(
                    "current epoch: {} current accuracy: {:.4f} current AUC: {:.4f} best acc: {:.4f} at epoch {} best AUC: {:.4f} at epoch {}".format(
                        epoch + 1, acc_metric, auc_result, best_acc_metric, best_acc_metric_epoch, best_auc_metric, best_auc_metric_epoch
                    )
                )
                writer.add_scalar("val_acc_auc", acc_metric, auc_result, epoch + 1)

    # Training completion message
    print(f"train completed, best_acc_metric: {best_acc_metric:.4f} at epoch: {best_acc_metric_epoch}")
    print(f"train completed, best_auc_metric: {best_auc_metric:.4f} at epoch: {best_auc_metric_epoch}")
    writer.close()

# Load pre-trained model and set device
pretrain = torch.load(r"/mnt/D/dataset/MaskCrop/pretrain/resnet_34_23dataset.pth")
pretrain['state_dict'] = {k.replace("module.", ""):v for k, v in pretrain['state_dict'].items()}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model and load pre-trained weights
model = resnet34(spatial_dims=3, n_input_channels=1, num_classes=num_class)
model.load_state_dict(pretrain['state_dict'], strict=False)
if torch.cuda.device_count() > 1:
    print(torch.cuda.device_count(), 'GPUs!')
    model = torch.nn.DataParallel(model)
model.to(device)

loss_function = torch.nn.CrossEntropyLoss()
auc_metric = ROCAUCMetric()

### Fine-tune the last layer for a few epochs
### Freeze model parameters except for the last fully connected layer
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True
optimizer = torch.optim.Adam(model.fc.parameters(), 1e-4, weight_decay=0.0002)
train(model, tra_loader, val_loader, loss_function, optimizer, num_epochs=50)

### Unfreeze all the layers and fine-tune the entire network for a few more epochs
for param in model.parameters():
    param.requires_grad = True
optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=0.0002)
train(model, tra_loader, val_loader, loss_function, optimizer, num_epochs=200)


### Feature extraction
dataframe_list = []

model.eval()
with torch.no_grad():
    for val_data in val_loader:
        val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
        feature = (model(val_images)).cpu()
        feature = torch.squeeze(feature)
        feature = feature.data.numpy()
        print(feature.shape)
        for i in range(len(val_data["label"].numpy())):
            pd_data_featrues = pd.DataFrame({
                'index': [val_data["image"].meta['filename_or_obj'][i]],
                'feature' :[list(feature)[i]]
                    })
            dataframe_list.append(pd_data_featrues)
    dataframe_total = pd.concat(dataframe_list,axis=0)
    dataframe_total = dataframe_total.reset_index(drop=True)
    dataframe_total.to_csv('Feature_nb_ACC_3d_A_tes.csv', index=False)