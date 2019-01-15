
import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

from unetres import UnetModel
from common import *

def train_s3(fold=3, disable_progress=False):
    directory = './data/'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_df = pd.read_csv('folds.csv')
    train_path = os.path.join(directory, 'train')

    train_ids = train_df[train_df.fold!=fold]['id'].values
    val_ids = train_df[train_df.fold==fold]['id'].values

    dataset = TGSSaltDataset(train_path, train_ids, augment=True)
    dataset_val = TGSSaltDataset(train_path, val_ids)


    model = UnetModel()
    model.train()
    model.to(device)

    epoch = 200
    learning_rate = 1e-3
    loss_fn = LovaszLoss()
    empty_loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=1e-4)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50, eta_min=0.0001)

    load_checkpoint(f'unet-fold{fold}-s2-best.pth', model, optimizer)

    best_iou = 0
    for e in range(epoch):
        train_loss = []
        for sample in tqdm(data.DataLoader(dataset, batch_size = 16, shuffle = True), disable=disable_progress):
            image, mask = sample['image'], sample['mask']
            image = image.type(torch.float).to(device)
            mask = mask.to(device)
            y_pred, y_pred_empty, y_pred_deep = model(image)
            seg_loss = loss_fn(y_pred, mask)
            class_loss = empty_loss_fn(y_pred_empty, empty_mask(mask))
            deep_loss = deep_sup_loss(y_pred_deep, mask)
            loss = seg_loss + class_loss * 0.05 # + deep_loss * 0.10

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            train_loss.append(seg_loss.item())

        val_loss = []
        val_iou = []
        for sample in data.DataLoader(dataset_val, batch_size = 30, shuffle = False):
            image, mask = sample['image'], sample['mask']
            image = image.to(device)
            y_pred, _, _ = model(image)

            loss = loss_fn(y_pred, mask.to(device))
            val_loss.append(loss.item())
            iou = iou_pytorch((y_pred > 0).int(), mask.int().to(device)).cpu()
            val_iou.append(iou)

        avg_iou = np.mean(val_iou)
        scheduler.step(np.mean(val_loss))
        print("Epoch: %d, Train: %.3f, Val: %.3f, IoU: %.3f" % (e, np.mean(train_loss), np.mean(val_loss), avg_iou))
        if avg_iou > best_iou:
            print('saving new best')
            save_checkpoint(f'unet-fold{fold}-s3-best.pth', model, optimizer)
            best_iou = avg_iou

    print('Best IoU: %.3f' % best_iou)

if __name__ == '__main__':
    try:
        train_s3()
    except KeyboardInterrupt:
        print('bye')
