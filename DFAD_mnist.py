from __future__ import print_function
import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os
import network
#from cvkit.vis import VisdomPlotter
#from cvkit.misc import pack_images
from dataloader import get_dataloader
import random
import numpy as np

import torchvision

#vp = VisdomPlotter('15550', env='DFAD-mnist')

def train(args, teacher, student, generator, device, train_loader, optimizer, epoch):
    teacher.eval()
    student.train()
    generator.train()
    optimizer_S, optimizer_G = optimizer

    for i in range( args.epoch_itrs ):
        for k in range(5):
            z = torch.randn( (args.batch_size, args.nz, 1, 1) ).to(device)
            optimizer_S.zero_grad()
            fake = generator(z).detach()
            t_logit = teacher(fake)
            s_logit = student(fake)
            loss_S = F.l1_loss(s_logit, t_logit.detach())
            
            loss_S.backward()
            optimizer_S.step()

        z = torch.randn( (args.batch_size, args.nz, 1, 1) ).to(device)
        optimizer_G.zero_grad()
        generator.train()
        fake = generator(z)
        t_logit = teacher(fake) 
        s_logit = student(fake)

        #loss_G = - torch.log( F.l1_loss( s_logit, t_logit )+1) 
        loss_G = - F.l1_loss( s_logit, t_logit ) 

        loss_G.backward()
        optimizer_G.step()

        if args.verbose and i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tG_Loss: {:.6f} S_loss: {:.6f}'.format(
                epoch, i, args.epoch_itrs, 100*float(i)/float(args.epoch_itrs), loss_G.item(), loss_S.item()))
            #vp.add_scalar('Loss_S', (epoch-1)*args.epoch_itrs+i, loss_S.item())
            #vp.add_scalar('Loss_G', (epoch-1)*args.epoch_itrs+i, loss_G.item())

def test(args, student, generator, device, test_loader, epoch=0):
    student.eval()
    generator.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            z = torch.randn( (data.shape[0], args.nz, 1, 1), device=data.device, dtype=data.dtype )
            fake = generator(z)
            output = student(data)
            #if i==0:
            #    vp.add_image( 'input', pack_images( ((data+1)/2).clamp(0,1).detach().cpu().numpy(), col=8 ) )
            #    vp.add_image( 'generated', pack_images( ((fake+1)/2).clamp(0,1).detach().cpu().numpy(), col=8 ) )

            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Epoch {} Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        epoch, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    acc = correct/len(test_loader.dataset)
    return acc

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='DFAD MNIST')
    parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train (default: 40)')
    parser.add_argument('--lr_S', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--lr_G', type=float, default=1e-3,
                        help='learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=1e-4)


    parser.add_argument('--epoch_itrs', type=int, default=50)
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist'],
                        help='dataset name (default: mnist)')
    parser.add_argument('--model', type=str, default='lenet5_half', choices=['lenet5_half'],
                        help='model name (default: lenet5_half)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--ckpt', type=str, default='checkpoint/teacher/mnist-lenet5.pt')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--test-only', action='store_true', default=False)
    parser.add_argument('--download', action='store_true', default=False)
    parser.add_argument('--step_size', type=int, default=100, metavar='S')
    parser.add_argument('--scheduler', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    os.makedirs('checkpoint/student', exist_ok=True)
    print(args)

    train_loader, test_loader = get_dataloader(args)
    teacher = network.lenet.LeNet5()
    student = network.lenet.LeNet5Half()
    generator = network.gan.GeneratorA(nz=args.nz, nc=1, img_size=32)

    teacher.load_state_dict( torch.load( args.ckpt ) )

    print("Teacher restored from %s"%(args.ckpt))
    teacher = teacher.to(device)
    student = student.to(device)
    generator = generator.to(device)
    teacher.eval()

    optimizer_S = optim.SGD( student.parameters(), lr=args.lr_S, weight_decay=args.weight_decay, momentum=0.9 )
    optimizer_G = optim.Adam( generator.parameters(), lr=args.lr_G )
    
    if args.scheduler:
        scheduler_S = optim.lr_scheduler.StepLR(optimizer_S, args.step_size, 0.1)
        scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, args.step_size, 0.1)
    best_acc = 0
    if args.test_only:
        acc = test(args, student, generator, device, test_loader)
        return
    acc_list = []
    for epoch in range(1, args.epochs + 1):
        # Train
        if args.scheduler:
            scheduler_S.step()
            scheduler_G.step()

        train(args, teacher=teacher, student=student, generator=generator, device=device, train_loader=train_loader, optimizer=[optimizer_S, optimizer_G], epoch=epoch)
        # Test
        acc = test(args, student, generator, device, test_loader, epoch)
        acc_list.append(acc)
        if acc>best_acc:
            best_acc = acc
            torch.save(student.state_dict(),"checkpoint/student/%s-%s.pt"%('mnist', 'lenet5_half'))
            torch.save(generator.state_dict(),"checkpoint/student/%s-%s-generator.pt"%('mnist', 'lenet5_half'))
        #vp.add_scalar('Acc', epoch, acc)
    print("Best Acc=%.6f"%best_acc)

    import csv
    os.makedirs('log', exist_ok=True)
    with open('log/DFAD-mnist.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(acc_list)

if __name__ == '__main__':
    main()