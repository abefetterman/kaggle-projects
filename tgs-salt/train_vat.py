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

import glob

from lovasz import lovasz_hinge
from vat import VATLoss

from unetres import UnetModel
from common import *

def train_vat(fold=2, disable_progress=False):
    directory = './data/'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_df = pd.read_csv('folds.csv')
    train_path = os.path.join(directory, 'train')

    train_ids = train_df[train_df.fold!=fold]['id'].values
    val_ids = train_df[train_df.fold==fold]['id'].values

    dataset = TGSSaltDataset(train_path, train_ids, augment=True)
    dataset_val = TGSSaltDataset(train_path, val_ids)

    test_path = os.path.join(directory, 'test')
    test_file_list = glob.glob(os.path.join(test_path, 'images', '*.png'))
    test_file_list = [f.split('/')[-1].split('.')[0] for f in test_file_list]
    test_file_list[:3], test_path
    dataset_test = TGSSaltDataset(test_path, test_file_list, is_test = True)

    model = UnetModel()
    model.train()
    model.to(device)

    #load_auto(f'auto-new-{num_filters}-s2-best.pth', model, device)
    #set_encoder_train(model, False)

    epoch = 100
    learning_rate = 5e-3
    alpha = 1.0
    #loss_fn = torch.nn.BCEWithLogitsLoss()
    vat_loss = VATLoss(eps=1.0, ip=1)
    empty_loss_fn = nn.BCEWithLogitsLoss()
    ent_loss_fn = EntropyLoss()
    #mask = get_mask().to(device)
    loss_fn = LovaszLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    try:
        load_checkpoint(f'unet-fold{fold}-s2-best.pth', model, optimizer)
    except:
        print('oops no file')

    train_iter = data.DataLoader(dataset, batch_size = 16, shuffle = True)
    test_iter = data.DataLoader(dataset_test, batch_size = 16, shuffle = True)
    best_iou = 0
    for e in range(epoch):
        train_loss = []
        smooth_loss = []
        unlabeled_loss = []
        #for sample in tqdm(data.DataLoader(dataset, batch_size = 30, shuffle = True)):
        for sample, test_sample in zip(tqdm(train_iter),test_iter):
            image, mask = sample['image'], sample['mask']
            image = image.type(torch.float).to(device)
            test_image = test_sample['image']
            test_image = test_image.type(torch.float).to(device)

            lds = 0 # vat_loss(model, image)
            lds_test = 0 # vat_loss(model, test_image)
            y_pred, y_pred_empty, _ = model(image)
            test_pred, test_pred_empty, _ = model(test_image)
            ul_loss = ent_loss_fn(test_pred_empty)
            direct_loss = loss_fn(y_pred, mask.to(device))
            class_loss = empty_loss_fn(y_pred_empty, empty_mask(mask).to(device))
            loss = direct_loss + 0.05 * (class_loss + alpha * (lds + lds_test) + ul_loss)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            train_loss.append(direct_loss.item())
            smooth_loss.append(0) #lds.item() + lds_test.item())
            unlabeled_loss.append(ul_loss.item())

        val_loss = []
        val_iou = []
        for sample in data.DataLoader(dataset_val, batch_size = 16, shuffle = False):
            image, mask = sample['image'], sample['mask']
            image = image.to(device)
            y_pred, _, _ = model(image)

            loss = loss_fn(y_pred, mask.to(device))
            val_loss.append(loss.item())
            iou = iou_pytorch((y_pred > 0).int(), mask.int().to(device)).cpu()
            val_iou.append(iou)

        avg_iou = np.mean(val_iou)
        scheduler.step(np.mean(val_loss))
        print("Epoch: %d, Train: %.3f, Smooth: %.3f, UL: %.3f, Val: %.3f, IoU: %.3f" % (e, np.mean(train_loss), np.mean(smooth_loss), np.mean(unlabeled_loss), np.mean(val_loss), avg_iou))
        if avg_iou > best_iou:
            print('saving new best')
            save_checkpoint(f'unet-fold{fold}-vat-best.pth', model, optimizer)
            best_iou = avg_iou

    print('Best IoU: %.3f' % best_iou)


if __name__ == '__main__':
    try:
        train_vat()
    except KeyboardInterrupt:
        print('bye')
