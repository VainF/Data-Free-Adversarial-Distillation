from torchvision import datasets, transforms
import torch
from dataset.caltech import Caltech101
from dataset.camvid import CamVid
from dataset.nyu import NYUv2, NYUv2Depth
from utils import ext_transforms

def get_dataloader(args):
    if args.dataset.lower()=='mnist':
        train_loader = torch.utils.data.DataLoader( 
            datasets.MNIST(args.data_root, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Resize((32, 32)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader( 
            datasets.MNIST(args.data_root, train=False, download=True,
                      transform=transforms.Compose([
                          transforms.Resize((32, 32)),
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)

    elif args.dataset.lower()=='cifar10':
        train_loader = torch.utils.data.DataLoader( 
            datasets.CIFAR10(args.data_root, train=True, download=True,
                       transform=transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader( 
            datasets.CIFAR10(args.data_root, train=False, download=True,
                       transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
    elif args.dataset.lower()=='cifar100':
        train_loader = torch.utils.data.DataLoader( 
            datasets.CIFAR100(args.data_root, train=True, download=True,
                       transform=transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader( 
            datasets.CIFAR100(args.data_root, train=False, download=True,
                       transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
    elif args.dataset.lower()=='caltech101':
        train_loader = torch.utils.data.DataLoader(
            Caltech101(args.data_root, train=True, download=args.download,
                        transform=transforms.Compose([
                            transforms.Resize(128),
                            transforms.RandomCrop(128),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(
            Caltech101(args.data_root, train=False, download=args.download, 
                        transform=transforms.Compose([
                            transforms.Resize(128),
                            transforms.CenterCrop(128),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                        ])), 
            batch_size=args.test_batch_size, shuffle=False, num_workers=2)
    elif args.dataset.lower()=='imagenet':
        train_loader = None # not required
        test_loader = torch.utils.data.DataLoader( 
            datasets.ImageNet(args.data_root, split='val', download=True,
                      transform=transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]),
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=4) # shuffle for visualization

    ############ Segmentation       
    elif args.dataset.lower()=='camvid':
        train_loader = torch.utils.data.DataLoader(
            CamVid(args.data_root, split='train',
                        transform=ext_transforms.ExtCompose([
                            ext_transforms.ExtResize(256),
                            ext_transforms.ExtRandomCrop(128, pad_if_needed=True),
                            ext_transforms.ExtRandomHorizontalFlip(),
                            ext_transforms.ExtToTensor(),
                            ext_transforms.ExtNormalize((0.5,), (0.5,))
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(
            CamVid(args.data_root, split='test',
                        transform=ext_transforms.ExtCompose([
                            ext_transforms.ExtResize(256),
                            ext_transforms.ExtToTensor(),
                            ext_transforms.ExtNormalize((0.5,), (0.5,))
                        ])),
            batch_size=args.test_batch_size, shuffle=False, num_workers=2)
    elif args.dataset.lower() in ['nyuv2']:
        train_loader = torch.utils.data.DataLoader(
            NYUv2(args.data_root, split='train',
                        transform=ext_transforms.ExtCompose([
                            ext_transforms.ExtResize(256),
                            ext_transforms.ExtRandomCrop(128, pad_if_needed=True),
                            ext_transforms.ExtRandomHorizontalFlip(),
                            ext_transforms.ExtToTensor(),
                            ext_transforms.ExtNormalize((0.5,), (0.5,))
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(
            NYUv2(args.data_root, split='test',
                        transform=ext_transforms.ExtCompose([
                            ext_transforms.ExtResize(256),
                            ext_transforms.ExtToTensor(),
                            ext_transforms.ExtNormalize((0.5,), (0.5,))
                        ])),
            batch_size=args.test_batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader