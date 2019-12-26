from __future__ import print_function
import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
import torchvision
import network
from utils import soft_cross_entropy, kldiv
from utils.visualizer import VisdomPlotter
from utils.misc import pack_images, denormalize
from dataloader import get_dataloader
from utils.stream_metrics import StreamSegMetrics
import random, os
import numpy as np

vp = VisdomPlotter('15550', env='DFAD-camvid')

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
        fake = generator(z)
        t_logit = teacher(fake)
        s_logit = student(fake)

        loss_G = -torch.log( F.l1_loss(s_logit, t_logit)+1 )
        #loss_G = -F.l1_loss(s_logit, t_logit)
        
        loss_G.backward()
        optimizer_G.step()

        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tG_Loss: {:.6f} S_loss: {:.6f}'.format(
                epoch, i, args.epoch_itrs, 100*float(i)/float(args.epoch_itrs), loss_G.item(), loss_S.item()))
            vp.add_scalar('Loss_S', (epoch-1)*args.epoch_itrs+i, loss_S.item())
            vp.add_scalar('Loss_G', (epoch-1)*args.epoch_itrs+i, loss_G.item())

from PIL import Image
import os

def test(args, student, teacher, generator, device, test_loader):
    student.eval()
    generator.eval()
    teacher.eval()

    seg_metrics = StreamSegMetrics(11)
    if args.save_img:
        os.makedirs('results/DFAD-camvid', exist_ok=True)
    img_idx=0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            z = torch.randn( (data.shape[0], args.nz, 1, 1), device=data.device, dtype=data.dtype )
            fake = generator(z)
            output = student(data)

            if args.save_img:
                t_out = teacher(data)
                
                input_imgs = (((data+1)/2)*255).clamp(0,255).detach().cpu().numpy().transpose(0,2,3,1).astype('uint8')
                colored_preds = test_loader.dataset.decode_target( output.max(1)[1].detach().cpu().numpy() ).astype('uint8')
                colored_teacher_preds = test_loader.dataset.decode_target( t_out.max(1)[1].detach().cpu().numpy() ).astype('uint8')
                colored_targets = test_loader.dataset.decode_target( target.detach().cpu().numpy() ).astype('uint8')
                for _pred, _img, _target, _tpred in zip( colored_preds, input_imgs, colored_targets, colored_teacher_preds  ):
                    Image.fromarray( _pred ).save('results/DFAD-camvid/%d_pred.png'%img_idx)
                    Image.fromarray( _img ).save('results/DFAD-camvid/%d_img.png'%img_idx)
                    Image.fromarray( _target ).save('results/DFAD-camvid/%d_target.png'%img_idx)
                    Image.fromarray( _tpred ).save('results/DFAD-camvid/%d_teacher.png'%img_idx)
                    img_idx+=1
            
            if i==0:
                t_out = teacher(data)
                t_out_onfake = teacher(fake)
                s_out_onfake = student(fake)
                vp.add_image( 'input', pack_images( ((data+1)/2).clamp(0,1).detach().cpu().numpy() ) )
                vp.add_image( 'generated', pack_images( ((fake+1)/2).clamp(0,1).detach().cpu().numpy() ) )
                vp.add_image( 'target', pack_images( test_loader.dataset.decode_target(target.cpu().numpy()), channel_last=True ).astype('uint8') )
                vp.add_image( 'pred',   pack_images( test_loader.dataset.decode_target(output.max(1)[1].detach().cpu().numpy().astype('uint8')), channel_last=True ).astype('uint8') )
                vp.add_image( 'teacher',   pack_images( test_loader.dataset.decode_target(t_out.max(1)[1].detach().cpu().numpy().astype('uint8')), channel_last=True ).astype('uint8') )
                vp.add_image( 'teacher-onfake',   pack_images( test_loader.dataset.decode_target(t_out_onfake.max(1)[1].detach().cpu().numpy().astype('uint8')), channel_last=True ).astype('uint8') )
                vp.add_image( 'student-onfake',   pack_images( test_loader.dataset.decode_target(s_out_onfake.max(1)[1].detach().cpu().numpy().astype('uint8')), channel_last=True ).astype('uint8') )
            seg_metrics.update(output.max(1)[1].detach().cpu().numpy().astype('uint8'), target.detach().cpu().numpy().astype('uint8'))

    results = seg_metrics.get_results()
    print('\nTest set: Acc= %.6f, mIoU: %.6f\n'%(results['Overall Acc'],results['Mean IoU']))
    return results

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='DFAD CamVid')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='N',
                        help='input batch size for testing (default: 16)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--epoch_itrs', type=int, default=50)
    parser.add_argument('--lr_S', type=float, default=0.1, metavar='LR',
                        help='learning rate of the student model (default: 0.1)')
    parser.add_argument('--lr_G', type=float, default=1e-3,
                        help='learning rate of the teacher model (default: 1e-3)')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='camvid', choices=['camvid'],
                        help='dataset name (default: camvid)')
    parser.add_argument('--model', type=str, default='deeplabv3_resnet50', choices=['deeplabv3_resnet50'],
                        help='model name (default: deeplabv3_resnet50)')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--ckpt', type=str, default='checkpoint/teacher/camvid-deeplabv3_resnet50-256.pt')
    parser.add_argument('--stu_ckpt', type=str, default=None)
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--nz', type=int, default=256)
    parser.add_argument('--test_only', action='store_true', default=False)
    parser.add_argument('--download', action='store_true', default=False)
    parser.add_argument('--step_size', type=int, default=100, metavar='S')
    parser.add_argument('--scheduler', action='store_true', default=False)
    parser.add_argument('--save_img', action='store_true', default=False)
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
    teacher = network.segmentation.deeplabv3.deeplabv3_resnet50(num_classes=11)
    student = network.segmentation.deeplabv3.deeplabv3_mobilenet(num_classes=11, dropout_p=0.5, pretrained_backbone=False)
    generator = network.gan.GeneratorB(nz=args.nz, img_size=128)
    
    teacher.load_state_dict( torch.load( args.ckpt ) )
    print("Teacher restored from %s"%(args.ckpt))

    if args.stu_ckpt is not None:
        student.load_state_dict( torch.load( args.stu_ckpt ) )
        try:
            generator.load_state_dict( torch.load( args.stu_ckpt[:-3]+'-generator.pt' ) )
        except:
            print("Generator ckpt not found")
        
        print('student loaded from %s'%args.stu_ckpt)
    
    teacher = teacher.to(device)
    student = student.to(device)
    generator = generator.to(device)    

    teacher.eval()

    optimizer_S = optim.SGD( student.parameters(), lr=args.lr_S, weight_decay=args.weight_decay, momentum=0.9)
    optimizer_G = optim.Adam( generator.parameters(), lr=args.lr_G)
    
    if args.scheduler:
        scheduler_S =  optim.lr_scheduler.StepLR(optimizer_S, args.step_size, gamma=0.1)
        scheduler_G =  optim.lr_scheduler.StepLR(optimizer_G, args.step_size, gamma=0.1)
    
    best_result = 0
    if args.test_only:
        results = test(args, student, teacher, generator, device, test_loader)
        return
    miou_list = []
    for epoch in range(1, args.epochs + 1):
        # Train
        train(args, teacher=teacher, student=student, generator=generator, device=device, train_loader=train_loader, optimizer=[optimizer_S, optimizer_G], epoch=epoch)
        # Test
        results = test(args, student, teacher, generator, device, test_loader)
        
        if results['Mean IoU']>best_result:
            best_result = results['Mean IoU']
            torch.save(student.state_dict(),"checkpoint/student/%s-%s.pt"%('camvid', 'deeplabv3_mobilenet'))
            torch.save(generator.state_dict(),"checkpoint/student/%s-%s-generator.pt"%('camvid', 'deeplabv3_mobilenet'))
        vp.add_scalar('mIoU', epoch, results['Mean IoU'])
        miou_list.append( results['Mean IoU'] )
        if args.scheduler:
            scheduler_S.step()
            scheduler_G.step()

    print("Best mIoU=%.6f"%best_result)

    import csv
    os.makedirs('log', exist_ok=True)
    with open('log/DFAD-camvid.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(miou_list)

if __name__ == '__main__':
    main()
