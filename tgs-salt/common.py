# Adapted from vizualization kernel
import os
import numpy as np
import torch

from torch.utils import data

import cv2
from pathlib import Path
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomContrast,
    RandomGamma,
    RandomBrightness,
    GaussNoise,
)
from albumentations.torch import ToTensor

PAD_SIZE=112

def get_padding(divisible=32, height=101, width=101):
    min_height, min_width = PAD_SIZE, PAD_SIZE

    if height < min_height:
        h_pad_top = int((min_height - height) / 2.0)
        h_pad_bottom = min_height - height - h_pad_top
    else:
        h_pad_top = 0
        h_pad_bottom = 0

    if width < min_width:
        w_pad_left = int((min_width - width) / 2.0)
        w_pad_right = min_width - width - w_pad_left
    else:
        w_pad_left = 0
        w_pad_right = 0

    #return (x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    return (w_pad_left, h_pad_top, w_pad_right, h_pad_bottom)

def get_mask():
    x_min_pad, y_min_pad, x_max_pad, y_max_pad = get_padding()
    mask = torch.zeros(128,128)
    mask[y_min_pad:128 - y_max_pad, x_min_pad:128 - x_max_pad] = 1.0
    return mask

def load_image(path, flip = None, rotate=None, mask = False):
    """
    Load image from a given path and pad it on the sides, so that eash side is divisible by 32 (newtwork requirement)

    if pad = True:
        returns image as numpy.array, tuple with padding in pixels as(x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    else:
        returns image as numpy.array
    """
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(img.shape)
    if (flip != None):
        img = cv2.flip(img, flip)
    if (rotate != None):
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    img = img[:,:,np.newaxis]
    height, width, _ = img.shape

    x_min_pad, y_min_pad, x_max_pad, y_max_pad = get_padding(divisible=32, height=height, width=width)

    img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_REFLECT_101)

    img = img[:,:,np.newaxis]
    if mask:
        # Convert mask to 0 and 1 format
        img = img[:, :, 0:1] // 255
        return torch.from_numpy(img).float().permute([2, 0, 1])
    else:
        img = (img) / 255.0
        return torch.from_numpy(img).float().permute([2, 0, 1])


def unpad_stack(im_array, height=101, width=101):
    # return np.vstack(im_array)[:, 0, :, :]
    x_min_pad, y_min_pad, x_max_pad, y_max_pad = get_padding(divisible=16)

    stacked_im = np.vstack(im_array)[:, 0, :, :]
    return stacked_im[:, y_min_pad:PAD_SIZE - y_max_pad, x_min_pad:PAD_SIZE - x_max_pad]

import random

class TGSSaltDataset(data.Dataset):
    def __init__(self, root_path, file_list, is_test = False, is_val=False, augment = False):
        self.is_test = is_test
        self.augment = augment
        self.root_path = root_path
        self.file_list = file_list
        self.pad = Compose([
            PadIfNeeded(p=1, min_height=PAD_SIZE, min_width=PAD_SIZE),
            ToTensor(),
        ])
        original_height,original_width = 101,101
        self.augmentation = Compose([
            RandomSizedCrop(min_max_height=(50, 101), height=original_height, width=original_width, p=0.9),
            HorizontalFlip(p=0.5),
            GridDistortion(p=0.8),
            RandomContrast(p=0.8),
            RandomBrightness(p=0.8),
            RandomGamma(p=0.8)
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if index not in range(0, len(self.file_list)):
            return self.__getitem__(np.random.randint(0, self.__len__()))

        file_id = self.file_list[index]

        image_folder = os.path.join(self.root_path, "images")
        image_path = os.path.join(image_folder, file_id + ".png")
        image = cv2.imread(image_path)
        data = {'image': image}
        if not self.is_test:
            mask_folder = os.path.join(self.root_path, "masks")
            mask_path = os.path.join(mask_folder, file_id + ".png")
            mask = cv2.imread(mask_path, 0)
            data['mask'] = mask
        if self.augment:
            data = self.augmentation(**data)
        return self.pad(**data)

class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7, mask=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.mask = mask

    def forward(self, input, target):
        y = target
        logit = torch.sigmoid(input)
        logit = logit.clamp(self.eps, 1. - self.eps)

        pt = y * logit + (1 - y) * (1 - logit) # cross entropy
        loss = - torch.log(pt) * (1 - pt) ** self.gamma # focal loss

        if self.mask:
            loss = loss * self.mask

        return loss.mean()


from lovasz import lovasz_hinge

class LovaszLoss(nn.Module):
    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss

class EntropyLoss(nn.Module):
    def forward(self, input_logit):
        return -torch.mean(torch.sigmoid(input_logit)*F.logsigmoid(input_logit) +
                    torch.sigmoid(-input_logit)*F.logsigmoid(-input_logit))

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    # optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

def load_auto(checkpoint_path, model, device='cpu'):
    state = torch.load(checkpoint_path, map_location=device)
    keys = state['state_dict'].keys()
    goodKeys = list(filter(lambda k: k.find('res')==0 or k.find('nl')==0, list(keys)))
    goodState = { k: state['state_dict'][k] for k in goodKeys }
    model.load_state_dict(goodState, strict=False)
    print('auto loaded from %s' % checkpoint_path)

def deep_sup_loss(x, mask, layers=5):
    deep_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    deep_mask = mask.repeat([1,layers,1,1])
    mask_size = deep_mask.shape[-1] * deep_mask.shape[-2]
    deep_loss = deep_loss_fn(x, deep_mask).sum([1,2,3]) / mask_size / layers
    bin_mask = empty_mask(mask)
    avg_loss = torch.sum(deep_loss * bin_mask) / (1e-6 + torch.sum(bin_mask))
    return avg_loss

def pred_merge(image, pred):
    new_image = image.clone()
    new_image[:,1:2] = torch.sigmoid(pred)
    new_image[:,2:] = torch.sigmoid(-pred)*image[:,:1]
    return new_image

def pred_weights(pred, mask):
    return mask * torch.sigmoid(pred) + (1 - mask) * torch.sigmoid(-pred)

import torch.utils.model_zoo as model_zoo
model_url = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'

def load_pretrained(model):
    model.resnet.load_state_dict(model_zoo.load_url(model_url), strict=False)

def set_encoder_train(model, train=False):
    for param in model.res1.parameters():
        param.requires_grad = train
    for param in model.res2.parameters():
        param.requires_grad = train
    for param in model.res3.parameters():
        param.requires_grad = train
    for param in model.res4.parameters():
        param.requires_grad = train
    for param in model.resm.parameters():
        param.requires_grad = train

div_eps = 1e-6

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    # outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    # x_min_pad, y_min_pad, x_max_pad, y_max_pad = get_padding()
    # outputs = outputs[:, :, y_min_pad:128 - y_max_pad, x_min_pad:128 - x_max_pad]
    # labels = labels[:, :, y_min_pad:128 - y_max_pad, x_min_pad:128 - x_max_pad]
    intersection = (outputs & labels).float().sum((1, 2, 3))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2, 3))         # Will be zzero if both are 0

    iou = (intersection + div_eps) / (union + div_eps)  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return thresholded.mean()  # Or thresholded.mean() if you are interested in average across the batch

def empty_mask(a):
    batch_size = a.shape[0]
    return torch.max(a.view(batch_size, -1),-1, keepdim=True)[0]

def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

def iou_numpy(A, B):
    batch_size = A.shape[0]
    metric = []
    for batch in range(batch_size):
        t, p = A[batch], B[batch]
        if np.count_nonzero(t) == 0 and np.count_nonzero(p) > 0:
            metric.append(0)
            continue
        if np.count_nonzero(t) == 0 and np.count_nonzero(p) == 0:
            metric.append(1)
            continue

        iou = jaccard(t, p)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    return np.mean(metric)


def jaccard(y_true, y_pred):
    epsilon = 1e-15
    intersection = (y_pred * y_true).sum()
    union = y_true.sum() + y_pred.sum()

    return ((intersection + epsilon)/ (union - intersection + epsilon)).mean()

# def iou_numpy(A, B):
#     # Numpy version
#
#     batch_size = A.shape[0]
#     metric = 0.0
#     for batch in range(batch_size):
#         t, p = A[batch], B[batch]
#         true = np.sum(t)
#         pred = np.sum(p)
#
#         # deal with empty mask first
#         if true == 0:
#             metric += (pred == 0)
#             continue
#
#         # non empty mask case.  Union is never empty
#         # hence it is safe to divide by its number of pixels
#         intersection = np.sum(t * p)
#         union = true + pred - intersection
#         iou = intersection / union
#
#         # iou metrric is a stepwise approximation of the real iou over 0.5
#         iou = np.floor(max(0, (iou - 0.5)*20)) / 10
#
#         metric += iou
#
#     # teake the average over all images in batch
#     metric /= batch_size
#     return metric

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths
