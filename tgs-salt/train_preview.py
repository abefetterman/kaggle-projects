# Direct unet
# add to baseline:
## train-test split with stratification
## lovasz_hinge loss fn
## residual block unet
## train time augmentation

import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

from common import *

from unetres import UnetModel

directory = './data/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_df = pd.read_csv('folds.csv')
train_path = os.path.join(directory, 'train')

fold=2
train_ids = train_df[train_df.fold!=fold]['id'].values
val_ids = train_df[train_df.fold==fold]['id'].values

dataset = TGSSaltDataset(train_path, train_ids, augment=True)
dataset_val = TGSSaltDataset(train_path, val_ids)

model = UnetModel()
model.eval()
model.to(device)

optimizer = {}

load_checkpoint(f'unet-fold{fold}-s2-best.pth', model, optimizer)

# import glob
#
# test_path = os.path.join(directory, 'test')
# test_file_list = glob.glob(os.path.join(test_path, 'images', '*.png'))
# test_file_list = [f.split('/')[-1].split('.')[0] for f in test_file_list]
# test_file_list[:3], test_path
# dataset_test = TGSSaltDataset(test_path, test_file_list, is_test = True)

# all_predictions = []
# for sample in tqdm(data.DataLoader(dataset_test, batch_size = 30)):
#     image = sample['image']
#     image = image.type(torch.float).to(device)
#     y_pred = model(image).cpu().detach().numpy()
#     all_predictions.append(y_pred)
# all_predictions_stacked = unpad_stack(all_predictions)

val_predictions = []
val_masks = []
for sample in tqdm(data.DataLoader(dataset_val, batch_size = 30)):
    image, mask = sample['image'], sample['mask']
    image = image.type(torch.float).to(device)
    y_pred = model(image)[0].cpu().detach().numpy()
    y_pred_flip = flip(model(flip(image,-1))[0],-1).cpu().detach().numpy()
    val_predictions.append((y_pred+y_pred_flip)/2)
    val_masks.append(mask)

val_predictions_stacked = unpad_stack(val_predictions)
val_masks_stacked = unpad_stack(val_masks)
print(val_predictions_stacked.shape)
metric_by_threshold = []
for threshold in np.linspace(-.5, .5, 21):

    # iou_values = []
    metric = iou_numpy((val_predictions_stacked > threshold).astype(int), val_masks_stacked.astype(int))

    print('Threshold: %.2f, Metric: %.3f' % (threshold, metric))
    metric_by_threshold.append((metric, threshold))

best_metric, best_threshold = max(metric_by_threshold)

threshold = 0.1
print('threshold: {}'.format(threshold))
binary_prediction = (all_predictions_stacked > threshold).astype(int)

all_masks = []
for p_mask in list(binary_prediction):
    p_mask = rle_encoding(p_mask)
    all_masks.append(' '.join(map(str, p_mask)))

submit = pd.DataFrame([test_file_list, all_masks]).T
submit.columns = ['id', 'rle_mask']
submit.to_csv('submission.csv.gz', compression = 'gzip', index = False)
