
import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from hybrid import Hybrid

from unetres import UnetModel
from common import *

def train_unet2(fold=1, disable_progress=False):
    directory = './data/'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_df = pd.read_csv('folds.csv')
    train_path = os.path.join(directory, 'train')

    train_ids = train_df[train_df.fold!=fold]['id'].values
    val_ids = train_df[train_df.fold==fold]['id'].values

    dataset = TGSSaltDataset(train_path, train_ids, augment=True)
    dataset_val = TGSSaltDataset(train_path, val_ids)

    stage1 = UnetModel()
    stage1.eval()
    stage1.to(device)

    model = UnetModel()
    model.train()
    model.to(device)

    epoch = 100
    learning_rate = 5e-3
    loss_fn = nn.BCEWithLogitsLoss(reduction = 'none')

    lovasz_fn = LovaszLoss()
    lovasz_coeff = 0.2
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    load_checkpoint(f'unet-fold{fold}-s2-best.pth', stage1, optimizer)
    #load_checkpoint(f'unet-fold{fold}-s2-best.pth', model, optimizer)

    best_iou = 0
    for e in range(epoch):
        train_loss = []
        train_binary_loss = []
        for sample in tqdm(data.DataLoader(dataset, batch_size = 16, shuffle = True), disable=disable_progress):
            image, mask = sample['image'], sample['mask']
            image = image.type(torch.float).to(device)
            mask = mask.to(device)
            stage1_pred, _, _ = stage1(image)
            new_image = pred_merge(image, stage1_pred)
            weights = pred_weights(stage1_pred, mask)
            stage2_pred, _, _ = model(new_image)
            unweighted_loss = loss_fn(stage2_pred, mask)
            weighted_loss = torch.sum(unweighted_loss * weights) / (1e-6 + torch.sum(weights))
            lovasz_loss = lovasz_fn(stage2_pred, mask)
            loss = weighted_loss #+ lovasz_loss * lovasz_coeff

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            train_loss.append(lovasz_loss.item())
            train_binary_loss.append(weighted_loss.item())

        val_loss = []
        val_binary_loss = []
        val_iou = []
        for sample in data.DataLoader(dataset_val, batch_size = 16, shuffle = False):
            image, mask = sample['image'], sample['mask']
            image = image.to(device)
            mask = mask.to(device)
            stage1_pred, _, _ = stage1(image)
            new_image = pred_merge(image, stage1_pred)
            stage2_pred, _, _ = model(new_image)

            y_pred = (stage1_pred + stage2_pred) / 2.0

            weights = pred_weights(stage1_pred, mask)
            unweighted_loss = loss_fn(stage2_pred, mask)
            weighted_loss = torch.sum(unweighted_loss * weights) / (1e-6 + torch.sum(weights))


            lovasz = lovasz_fn(y_pred, mask)
            val_loss.append(lovasz.item())
            val_binary_loss.append(weighted_loss.item())

            iou = iou_pytorch((y_pred > 0).int(), mask.int().to(device)).cpu()
            val_iou.append(iou)

        avg_iou = np.mean(val_iou)
        scheduler.step(np.mean(val_loss))
        print("Epoch: %d, Train: %.3f, Bin: %.3f, Val: %.3f, Val Bin: %.3f, IoU: %.3f" % (e, np.mean(train_loss), np.mean(train_binary_loss), np.mean(val_loss), np.mean(val_binary_loss), avg_iou))
        if avg_iou > best_iou:
            print('saving new best')
            save_checkpoint(f'unet-fold{fold}-unet2-best.pth', model, optimizer)
            best_iou = avg_iou

    print('Best IoU: %.3f' % best_iou)

if __name__ == '__main__':
    try:
        train_unet2()
    except KeyboardInterrupt:
        print('bye')
