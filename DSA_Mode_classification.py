#---------------------------------------------------------------------------
# Setup
#---------------------------------------------------------------------------
import os
import sys
import argparse
import shutil
import tempfile
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, accuracy_score, balanced_accuracy_score
from scipy import stats

import monai
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import decollate_batch, DataLoader
from monai.metrics import ROCAUCMetric
from monai.networks.nets import TorchVisionFCModel
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    Resize,
    RandRotate,
    RandZoom,
    ScaleIntensity,
)
from monai.utils import set_determinism

from utils.SI_classification_models import classification_model

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
torch.cuda.empty_cache()

#---------------------------------------------------------------------------
# Function Definitions
#---------------------------------------------------------------------------
def load_DataList(
    image_DB_filepath,
    key_images_only,
    train_val_test,
):
    ''' Load image csv and return list of images and list of labels '''
    df_images = pd.read_csv(image_DB_filepath)
    if key_images_only:
        df_images = df_images.loc[df_images['diagnostic'] == 1]
        df_images.reset_index(inplace=True, drop=True)
    df_images = df_images.loc[df_images['train_val_test'] == train_val_test]
    df_images.reset_index(inplace=True, drop=True)

    image_list = df_images.image_file_path.tolist()
    label_list = df_images.numeric_label.tolist()
    
    ### Sequence Length List
    ordered_uids = pd.DataFrame(df_images['SeriesUID'].drop_duplicates()) # Ordered DataFrame with unique 'SeriesUID' values in the order they appear
    sequence_counts = df_images.groupby('SeriesUID').size().reset_index(name='count') # Group by 'SeriesUID' and count the occurrences
    ordered_sequence_counts = ordered_uids.merge(sequence_counts, on='SeriesUID') # Merge the ordered_uids with the counts to maintain the original order
    sequence_length_list = ordered_sequence_counts['count'].tolist()
            
    return(image_list,label_list,sequence_length_list)

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]

def get_ClassNames(image_DB_filepath):
    df_images = pd.read_csv(image_DB_filepath)
    df_label = df_images.drop_duplicates(subset='label')
    df_label = df_label.sort_values(by='numeric_label')
    class_names = df_label.label.tolist()
    return class_names

def train_epoch(model, loader, optimizer, loss_function, epoch, device, args):
    """One train epoch over the dataset"""

    model.train()
    epoch_loss = 0
    
    step = 0
    for idx, batch_data in enumerate(loader):
        step += 1
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(
        "Train epoch {}/{} ".format(epoch+1, args.epochs),
        "Index {}/{},".format(idx+1, len(loader)),
        "Loss: {:.4f}".format(loss.item()),
        )
        
    epoch_loss /= step
    print(f"Epoch {epoch + 1} Average Training Loss: {epoch_loss:.4f}")
    return epoch_loss

def val_epoch(model, loader, auc_metric, y_trans, y_pred_trans, epoch, device):
    """One validation epoch over the dataset"""
    model.eval()
    with torch.no_grad():
        y_pred = torch.tensor([], dtype=torch.float32, device=device)
        y = torch.tensor([], dtype=torch.long, device=device)
        for val_data in loader:
            val_images, val_labels = (
                val_data[0].to(device),
                val_data[1].to(device),
            )
            y_pred = torch.cat([y_pred, model(val_images)], dim=0)
            y = torch.cat([y, val_labels], dim=0)
        y_onehot = [y_trans(i) for i in decollate_batch(y, detach=False)]
        y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred)]
        auc_metric(y_pred_act, y_onehot)
        auc_result = auc_metric.aggregate()
        auc_metric.reset()
        del y_pred_act, y_onehot
        acc_value = torch.eq(y_pred.argmax(dim=1), y)
        acc_metric = acc_value.sum().item() / len(acc_value)

        print(f"Epoch {epoch + 1} Average Validation AUC: {auc_result:.4f}, Average Validation Accuracy: {acc_metric:.4f}")
        
        return auc_result, acc_metric

def save_checkpoint(model, epoch, args, filename="model.pt", val_metric=0):
    """Save checkpoint"""

    state_dict = model.state_dict()

    save_dict = {"epoch": epoch, "val_metric": val_metric, "state_dict": state_dict}

    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)

def compute_modes(y_true, y_pred, sequence_length_list):
    # Inputs:
    #   y_true: ordered list of manual labels for each tested image
    #   y_predict: ordered list of algorithmically generated labels for each tested image
    #   sequence_length_list_int: ordered list of image sequence lengths
    # Outputs:
    #   y_true_mode: ordered list of manual label for each sequence of images
    #   y_pred_mode: ordered list of predicted label (calculated from the mode) for each sequence of images
    y_true_mode = np.empty(len(sequence_length_list), dtype=int)
    y_pred_mode = np.empty(len(sequence_length_list), dtype=int)

    first = 0
    for i, length in enumerate(sequence_length_list):
        last = first + length
        y_true_mode[i] = y_true[first]
        y_pred_mode[i], _ = stats.mode(y_pred[first:last], keepdims=False)
        first = last
    return y_true_mode, y_pred_mode

def main_worker(gpu, args):

    set_determinism(seed=args.determinism_seed)
    #---------------------------------------------------------------------------
    # Prepare training, validation and test data lists
    #---------------------------------------------------------------------------
    image_DB_filepath = args.image_DB_filepath
    key_images_only = args.key_images_only

    train_images, train_labels, _ = load_DataList(
        image_DB_filepath=args.image_DB_filepath,
        key_images_only=args.key_images_only,
        train_val_test = 'train')

    val_images, val_labels, _ = load_DataList(
        image_DB_filepath = args.image_DB_filepath,
        key_images_only = args.key_images_only,
        train_val_test ='val')

    test_images, test_labels, test_sequence_length_list = load_DataList(
        image_DB_filepath = args.image_DB_filepath,
        key_images_only = args.key_images_only,
        train_val_test ='test')
    
    print("Training count =",len(train_images),"Validation count =", len(val_images), "Test count =",len(test_images))

    if args.model == 'inception_v3': # Requires image resizing
        image_size = (299,299)
        train_transforms = Compose(
            [
                LoadImage(image_only=True),
                EnsureChannelFirst(),
                ScaleIntensity(),
                Resize(image_size),
                RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
                RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
            ]
        )
        val_transforms = Compose(
            [
                LoadImage(image_only=True),
                EnsureChannelFirst(),
                Resize(image_size),
                ScaleIntensity()
            ]
        )
    else: # Models do not require image resizing
        train_transforms = Compose(
            [
                LoadImage(image_only=True),
                EnsureChannelFirst(),
                ScaleIntensity(),
                RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
                RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
            ]
        )

        val_transforms = Compose(
            [
                LoadImage(image_only=True),
                EnsureChannelFirst(),
                ScaleIntensity()
            ]
        )

    y_pred_trans = Compose([Activations(softmax=True)])
    y_trans = Compose([AsDiscrete(to_onehot=args.num_classes)])

    train_ds = ImageDataset(image_files = train_images, labels = train_labels, transforms= train_transforms)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    val_ds = ImageDataset(image_files = val_images, labels = val_labels, transforms= val_transforms)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.workers)

    test_ds = ImageDataset(image_files = test_images, labels = test_labels, transforms= val_transforms)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=args.workers)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = classification_model(model_name = args.model, num_classes = args.num_classes).to(device)

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        if "val_metric" in checkpoint:
            best_val_metric = checkpoint["val_metric"]
        if "epoch" in checkpoint:
            best_metric_epoch = checkpoint["epoch"]
        print("=> loaded checkpoint '{}' (epoch {}) (Metric (AUC) {})".format(args.checkpoint, checkpoint["epoch"]+1, best_val_metric))
    
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.optim_lr)

    #----------------------------------------
    # Scheduler
    #----------------------------------------
    if args.scheduler_type == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    elif args.scheduler_type == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, threshold=0.0001, 
                                 threshold_mode='rel', cooldown=2, min_lr=0, eps=1e-08, verbose=True)
    
    val_interval = args.val_every
    auc_metric = ROCAUCMetric()
    

    if args.test:
        torch.load(args.checkpoint, map_location="cpu")
        print('Evaluation on the Test Data Set')
        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for test_data in test_loader:
                test_images, test_labels = (
                    test_data[0].to(device),
                    test_data[1].to(device),
                )
                pred = model(test_images).argmax(dim=1)
                for i in range(len(pred)):
                    y_true.append(test_labels[i].item())
                    y_pred.append(pred[i].item())
        
        if args.test_type == "single_image":
            print(classification_report(y_true, y_pred, target_names=get_ClassNames(args.image_DB_filepath), digits=4))
        elif args.test_type == "mode":
            y_true, y_pred = compute_modes(y_true, y_pred, test_sequence_length_list)
            print(classification_report(y_true, y_pred, target_names=get_ClassNames(args.image_DB_filepath), digits=4))    
        else:
            print("Test mode type is not recognized. Choose mode or single_image")
            
        if args.save_test_pickle:
            pickle_filename = os.path.join(args.logdir, 'preds_targets_test.pickle') # args.logdir + '/preds_targets_test.pickle'
            print('Saving:',pickle_filename)
            with open(pickle_filename, 'wb') as file:
                pickle.dump((y_pred, y_true), file)
        
        sys.exit(0)

    
    
    #------------------------------------------------------------------------------------------------------------
    # Model Training and Validation
    #------------------------------------------------------------------------------------------------------------

    if args.logdir is not None:
        writer = SummaryWriter(log_dir=args.logdir)
        print("Writing logs to ", writer.log_dir)
    else:
        writer = None
    
    if args.checkpoint is None:
        best_val_metric = -1
        best_metric_epoch = -1
    
    epoch_loss_values = []
    auc_values = []

    for epoch in range(args.epochs):
        
        #---------------------------------------------------------------------
        # Train Epoch
        #---------------------------------------------------------------------
        print("-" * 10)
        
        epoch_loss = train_epoch(model, train_loader, optimizer, loss_function, epoch, device, args)

        if writer is not None:
            writer.add_scalar("train_loss", epoch_loss, epoch)

        #---------------------------------------------------------------------
        # Val Epoch
        #---------------------------------------------------------------------
        b_new_best = False
        
        if (epoch + 1) % val_interval == 0:
            val_auc, val_acc = val_epoch(model, val_loader, auc_metric, y_trans, y_pred_trans, epoch, device)

            if writer is not None:
                writer.add_scalar("val_auc", val_auc, epoch)
                writer.add_scalar("val_acc", val_acc, epoch)
            
            auc_values.append(val_auc)

            if val_auc > best_val_metric:
                best_val_metric = val_auc
                best_metric_epoch = epoch
                b_new_best = True
                
            print(
                f"Current epoch: {epoch + 1}, Current AUC: {val_auc:.4f}"
                f" Current validation accuracy: {val_acc:.4f}"
                f" Best AUC: {best_val_metric:.4f}"
                f" at epoch: {best_metric_epoch+1}"
            )

        if args.logdir is not None:
            save_checkpoint(model, epoch, args, val_metric=val_auc, filename="model_last_epoch.pt")
            if b_new_best:
                print("Saved new best metric model to ", args.best_model)
                save_checkpoint(model, epoch, args, val_metric=val_auc, filename=args.best_model)

        if args.scheduler_type == 'CosineAnnealingLR':
            scheduler.step()
        elif args.scheduler_type == 'ReduceLROnPlateau':
            scheduler.step(val_auc)
                
    print(f"Training Complete!! \nBest_metric (Validation AUC): {best_val_metric:.4f} " f"at epoch: {best_metric_epoch+1}")

def parse_args():

    parser = argparse.ArgumentParser(description="Parser for Single Image Classification for DSA.")
    parser.add_argument(
        "--image_DB_filepath", 
        required=True, 
        help="Path to data csv file"
    )

    parser.add_argument(
        "--model",
        default = 'resnet50',
        help = 'Define classification model from TorchVision Model List'
    )

    parser.add_argument(
        "--best_model",
        default = "best_model.pt",
        help = "Best model state filename.")
    
    parser.add_argument("--checkpoint", default=None, help="load existing checkpoint")
    parser.add_argument("--num_classes", default=6, type=int, help="number of output classes")
    parser.add_argument(
        "--key_images_only",
        action="store_true",
        help="run training on key diagnostic images only",
    )
    
    parser.add_argument("--logdir", default=None, help="path to log directory to store Tensorboard logs")
        
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run only inference on the test set, must specify the checkpoint argument",
    )
    
    parser.add_argument("--test_type",
        default="mode",
        help="Set for reporting inference type: mode or single_image" 
    )

    parser.add_argument("--save_test_pickle",
        action='store_true',
        help="Set TRUE to save a pickle of the TRAGETS & PREGS to --logdir. Must specify logdir argument"
    )

    parser.add_argument("--epochs", default=50, type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", default=100, type=int, help="Batch size")
    parser.add_argument("--optim_lr", default=3e-5, type=float, help="initial learning rate")

    parser.add_argument("--weight_decay", default=0, type=float, help="optimizer weight decay")
    
    parser.add_argument(
        "--val_every",
        default=1,
        type=int,
        help="run validation after this number of epochs, default 1 to run every epoch",
    )
    parser.add_argument("--verbose",action='store_true',help="Outputs Additional Info")
    parser.add_argument("--workers", default=10, type=int, help="number of workers for data loading")
    parser.add_argument("--scheduler_type", default='None', help="Choose from None, CosineAnnealingLR, ReduceLROnPlateau")
    parser.add_argument("--determinism_seed", default=4294967295, help="Set random seed for modules to enable deterministic training.")
    args = parser.parse_args()
    if args.verbose:
        print("Argument values:")
        for k, v in vars(args).items():
            print(k, "=>", v)
        print("-----------------")

    return args


if __name__ == "__main__":

    args = parse_args()
    if args.verbose:
        print_config()
    main_worker(0, args)

