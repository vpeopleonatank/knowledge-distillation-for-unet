import glob
import torch
import dataset
import numpy as np
from unet import UNet
import torch.nn as nn
from metrics import dice_loss
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR

num_of_epochs = 5

def evaluate(teacher, val_loader):
    teacher.eval().cuda()

    criterion = nn.BCEWithLogitsLoss()
    ll = []
    with torch.no_grad():
        for i,(img,gt) in enumerate(val_loader):
            if torch.cuda.is_available():
                img, gt = img.cuda(), gt.cuda()
            img, gt = Variable(img), Variable(gt)

            output = teacher(img)
            output = output.clamp(min = 0, max = 1)
            gt = gt.clamp(min = 0, max = 1)
            loss = dice_loss(output, gt)
            ll.append(loss.item())


    mean_dice = np.mean(ll)
    print('Eval metrics:\n\tAverabe Dice loss:{}'.format(mean_dice))


def train(teacher, optimizer, train_loader):
    print(' --- teacher training')
    teacher.train().cuda()
    criterion = nn.BCEWithLogitsLoss()
    ll = []
    for i, (img, gt) in enumerate(train_loader):
        print('i', i)
        if torch.cuda.is_available():
            img, gt = img.cuda(), gt.cuda()

        img, gt = Variable(img), Variable(gt)

        output = teacher(img)
        output = output.clamp(min = 0, max = 1)
        gt = gt.clamp(min = 0, max = 1)
        loss = dice_loss(output, gt)
        ll.append(loss.item())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    mean_dice = np.mean(ll)

    print("Average loss over this epoch:\n\tDice:{}".format(mean_dice))

import albumentations as albu
from albumentations.pytorch import ToTensor


def pre_transforms(image_size=320):
    return [albu.Resize(image_size, image_size, p=1)]



def post_transforms():
    # we use ImageNet image normalization
    # and convert it to torch.Tensor
    return [albu.Normalize(), ToTensor()]

def compose(transforms_to_compose):
    # combine all augmentations into single pipeline
    result = albu.Compose([
      item for sublist in transforms_to_compose for item in sublist
    ])
    return result

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_training_augmentation(image_size=320):
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=image_size, min_width=image_size, always_apply=True, border_mode=0),
        albu.RandomCrop(height=image_size, width=image_size, always_apply=True),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    # return albu.Compose(train_transform)
    return train_transform


def get_validation_augmentation(image_size=320):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(image_size, image_size, p=1),
        albu.PadIfNeeded(384, 480)
    ]
    # return albu.Compose(test_transform)
    return test_transform


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


if __name__ == "__main__":

    teacher = UNet(channel_depth = 32, n_channels = 3, n_classes=1)

    optimizer = torch.optim.Adam(teacher.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size = 100, gamma = 0.2)

    #load teacher and student model

    #NV: add val folder
    # train_list = glob.glob('/content/drive/MyDrive/Data/polyp_data/CVC-ClinicDB')
    # val_list = glob.glob('/home/nirvi/Internship_2020/Carvana dataset/val/val1/*jpg')
    #
    # tf = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    #
    # #2 tensors -> img_list and gt_list. for batch_size = 1 --> img: (1, 3, 320, 320); gt: (1, 1, 320, 320)
    # train_loader = torch.utils.data.DataLoader(
    #     dataset.listDataset(train_list,
    #     shuffle = False,
    #     transform = tf,
    #     ),
    #     batch_size = 1
    # )
    #
    #
    # val_loader = torch.utils.data.DataLoader(
    #     dataset.listDataset(val_list,
    #     shuffle = False,
    #     transform = tf,
    #     ),
    #     batch_size = 1
    # )

    from pathlib import Path

    # ROOT = Path("segmentation_data/")

    ROOT = Path('/content/drive/MyDrive/Data/polyp_data/CVC-ClinicDB')

    train_image_path = ROOT / 'Original'
    train_mask_path = ROOT / 'Ground Truth'

    ALL_IMAGES = sorted(train_image_path.glob("*.tif"))

    ALL_MASKS = sorted(train_mask_path.glob("*.tif"))

    train_transforms = compose([
        pre_transforms(384),
        get_training_augmentation(384),
        post_transforms()
    ])
    valid_transforms = compose([
        pre_transforms(384),
        get_validation_augmentation(384),
        post_transforms()
    ])

    from utils import get_loaders

    loaders = get_loaders(
        images=ALL_IMAGES,
        masks=ALL_MASKS,
        random_state=42,
        train_transforms_fn=train_transforms,
        valid_transforms_fn=valid_transforms,
        batch_size=8,
        # to_mask2d=True      # Convert mask to 2d array
    )

    for epoch in range(num_of_epochs):
        print(' --- teacher training: epoch {}'.format(epoch+1))
        train(teacher, optimizer, loaders["train"])

        #evaluate for one epoch on validation set
        evaluate(teacher, loaders["valid"])

        #if val_metric is best, add checkpoint

        torch.save(teacher.state_dict(), 'teacher_checkpoints/32/CP_32_{}.pth'.format(epoch+1))
        print("Checkpoint {} saved!".format(epoch+1))
        scheduler.step()
