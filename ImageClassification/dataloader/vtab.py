import torch.utils.data as data

from PIL import Image
import os
import os.path
from torchvision import transforms
import torch

from timm.data import create_transform

def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split()
            imlist.append((impath, int(imlabel)))

    return imlist


class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None, target_transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imlist)


def get_data(basedir, name, logger=None, evaluate=True, train_aug=False, batch_size=64):
    root = os.path.join(basedir, name)
    
    if train_aug:
        aug_transform = create_transform(
                input_size=(224, 224),
                is_training=True,
                color_jitter=0.4,
                auto_augment='rand-m9-mstd0.5-inc1',
                re_prob=0.0,
                re_mode='pixel',
                re_count=1,
                interpolation='bicubic',
            )
        aug_transform.transforms[0] = transforms.Resize((224, 224), interpolation=3)
    else:
        aug_transform = None

    # print(aug_transform)

    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_transform = aug_transform if aug_transform else transform
    # train_transform = transform

    if logger is not None:
        logger.info(f'Data transform, train:\n{train_transform}')
        logger.info(f'Data transform, test:\n{transform}')

    if evaluate:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(root=root, flist=root + "/train800val200.txt",
                transform=train_transform),
            batch_size=batch_size, shuffle=True, drop_last=False,
            num_workers=4, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            ImageFilelist(root=root, flist=root + "/test.txt",
                transform=transform),
            batch_size=512, shuffle=False,
            num_workers=4, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(root=root, flist=root + "/train800.txt",
                transform=train_transform),
            batch_size=batch_size, shuffle=True, drop_last=True,
            num_workers=4, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            ImageFilelist(root=root, flist=root + "/val200.txt",
                transform=transform),
            batch_size=512, shuffle=False,
            num_workers=4, pin_memory=True)
    return train_loader, val_loader