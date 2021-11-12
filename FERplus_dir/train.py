import argparse
import os,sys,shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
#import transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import math
from resnet_loss import resnet18,resnet34,resnet50,resnet101
import resnet_loss
from iresnet_loss import iresnet18,iresnet34,iresnet50,iresnet100
import iresnet_loss
#from part_attentioon_sample_fly import MsCelebDataset, CaffeCrop
from load_dataset_train import MsCelebDataset, CaffeCrop
import scipy.io as sio  
import numpy as np
import pdb
import time
from torch.utils.tensorboard import SummaryWriter
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4'

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

current_time=time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())

parser = argparse.ArgumentParser(description='PyTorch CelebA Training')
parser.add_argument('--img_dir', metavar='DIR', default='./dataset/FERPlus_reshape_crop_folder/Train', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='iresnet50', choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: alexnet)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=80, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-b_t', '--batch-size_t', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
#./model/SOTA/ijba_res18_naive.pth.tar
parser.add_argument('--pretrained', default='/home/DataBase4/sunpeiwen/RAN/FERplus_dir/pretrain/faces_emore_iresnet50/backbone.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--model_dir','-m', default='/home/DataBase4/sunpeiwen/RAN/FERplus_dir/model/mytrain/'+current_time+os.sep, type=str)
parser.add_argument('--end2end', default=True,
                    help='if true, using end2end with dream block, else, using naive architecture')

best_prec1 = 0

writer = SummaryWriter('/home/DataBase4/sunpeiwen/RAN/FERplus_dir/log/'+current_time)
def main():
    global args, best_prec1
    args = parser.parse_args()
    print(args)
    print("In tensorflow35 type \"tensorboard --logdir=/home/DataBase4/sunpeiwen/RAN/FERplus_dir/log/"+current_time+" \"")
    # pdb.set_trace()
    args_img_dir='./dataset/FERPlus_reshape_crop_folder/Train'

    train_list_file = './dataset/FERPlus_list/dlib_ferplus_train_center_crop_range_list.txt'
    train_label_file = './dataset/FERPlus_list/dlib_ferplus_train_center_crop_range_label.txt'

    caffe_crop = CaffeCrop('train')
    train_dataset =  MsCelebDataset(args_img_dir, train_list_file, train_label_file, 
            transforms.Compose([caffe_crop,transforms.ToTensor()]))

    
    args_img_dir_val='./dataset/FERPlus_reshape_crop_folder/Test'
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
   
    caffe_crop = CaffeCrop('test')
    val_list_file = './dataset/FERPlus_list/dlib_ferplus_val_center_crop_range_list.txt'
    val_label_file = './dataset/FERPlus_list/dlib_ferplus_val_center_crop_range_label.txt'

    val_dataset =  MsCelebDataset(args_img_dir_val, val_list_file, val_label_file, 
            transforms.Compose([caffe_crop,transforms.ToTensor()]))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size_t, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    # pdb.set_trace()
    #assert(train_dataset.max_label == val_dataset.max_label)

    
    # prepare model
    model = None
    assert(args.arch in ['resnet18','resnet34','resnet50','resnet101','iresnet18','iresnet34','iresnet50','iresnet100'])
    if args.arch == 'resnet18':
        #model = Res()
        model = resnet18(end2end=args.end2end)
   #     model = resnet18(pretrained=False, nverts=nverts_var,faces=faces_var,shapeMU=shapeMU_var,shapePC=shapePC_var,num_classes=class_num, end2end=args.end2end)
    elif args.arch == 'resnet34':
        model = resnet34(end2end=args.end2end)    
    elif args.arch == 'resnet50':
        model = resnet50(end2end=args.end2end)
    elif args.arch == 'resnet101':
        model = resnet101(end2end=args.end2end)
    elif args.arch == 'iresnet18':
        model = iresnet18()
    elif args.arch == 'iresnet34':
        model = iresnet34()
    elif args.arch == 'iresnet50':
        model = iresnet50()
    elif args.arch == 'iresnet100':
        model = iresnet100()
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    if args.arch[0:6] == "resnet":
        criterion1 = resnet_loss.MyLoss().cuda()
    elif args.arch[0:7] == "iresnet":
        criterion1 = iresnet_loss.MyLoss().cuda()
    #criterion=Cross_Entropy_Sample_Weight.CrossEntropyLoss_weight().cuda()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), args.lr,
                                 momentum=args.momentum,
                                 weight_decay=args.weight_decay)

   # optionally resume from a checkpoint
    
    if args.pretrained:
        
        checkpoint = torch.load(args.pretrained)
        try: 
            pretrained_state_dict = checkpoint['state_dict']
        except:
            pretrained_state_dict = checkpoint
        model_state_dict = model.state_dict()
        # pdb.set_trace()
        
        for key in pretrained_state_dict:
            if  ((key=='module.fc.weight')|(key=='module.fc.bias')):
                pass
            else:    
                model_state_dict[key] = pretrained_state_dict[key]

        model.load_state_dict(model_state_dict, strict = False)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    
    # for name, value in mymodel.named_parameters():
    #     #print(name)
    #     #print(value.requires_grad)
    #     #print(value)
    #     if name == 'conv1.weight':
    #         #confient
    #         value.requires_grad= False
    #     print(name)
    #     print(value.requires_grad)

    #pdb.set_trace()
    print ('args.evaluate',args.evaluate)
    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate_fixed(optimizer, epoch)
        writer.add_scalar("learning rate",optimizer.state_dict()['param_groups'][0]['lr'],epoch)
        # train for one epoch
        train(train_loader, model, criterion, criterion1, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, criterion1)

        writer.add_scalar("prec1(validate)",prec1,epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        
        best_prec1 = max(prec1.cuda().item(), best_prec1)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best.item())

def train(train_loader, model, criterion, criterion1, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    cla_losses = AverageMeter()
    yaw_losses = AverageMeter()
    losses = AverageMeter()
    weights_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input_first, target_first, input_second,target_second, input_third, target_third, input_forth, target_forth, input_fifth, target_fifth, input_sixth, target_sixth) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        input = torch.zeros([input_first.shape[0],input_first.shape[1],input_first.shape[2],input_first.shape[3],6])
        #input = torch.cat((input_first,input_second),0)
        #input = torch.cat((input,input_third),0)


        input[:,:,:,:,0] = input_first
        input[:,:,:,:,1] = input_second
        input[:,:,:,:,2] = input_third
        input[:,:,:,:,3] = input_forth
        input[:,:,:,:,4] = input_fifth
        input[:,:,:,:,5] = input_sixth
        #input[:,:,:,:,6] = input_seventh
        #input[:,:,:,:,7] = input_eigth
        
        target = target_first
         

        target = target.cuda()
 
        
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        
        # pdb.set_trace() torch.Size([64, 3, 224, 224, 6])
        # compute output
        pred_score, alphas_part_max, alphas_org = model(input_var)
        # pdb.set_trace()
        weights_loss = criterion1(alphas_part_max, alphas_org)
        loss = criterion(pred_score, target_var) + weights_loss
        
        # pdb.set_trace()
        
        # measure accuracy and record loss
        prec1 = accuracy(pred_score.data, target, topk=(1,))
        # pdb.set_trace()
        losses.update(loss.item(), input.size(0))
        try:
            weights_losses.update(weights_loss.item(), input.size(0))
        except:
            pdb.set_trace()
        top1.update(prec1[0], input.size(0))
        

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val} ({batch_time.avg})\t'
                  'Data {data_time.val} ({data_time.avg})\t'
                  'Loss {loss.val} ({loss.avg})\t'
                  'weights_loss {weights_loss.val} ({weights_loss.avg})\t'
                  'Prec@1 {top1.val} ({top1.avg})\t'
                                                              .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, weights_loss=weights_losses, top1=top1))
        writer.add_scalar("prec1(train)",top1.avg,epoch)
        writer.add_scalar("loss(train)",losses.avg,epoch)


def validate(val_loader, model, criterion, criterion1):
    batch_time = AverageMeter()
    cla_losses = AverageMeter()
    yaw_losses = AverageMeter()
    losses = AverageMeter()
    weights_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input_first, target_first, input_second,target_second, input_third, target_third, input_forth, target_forth, input_fifth, target_fifth, input_sixth, target_sixth) in enumerate(val_loader):
        # target = target.cuda(async=True)
        # input_var = torch.autograd.Variable(input, volatile=True)
        # target_var = torch.autograd.Variable(target, volatile=True)
        # compute output
        input = torch.zeros([input_first.shape[0],input_first.shape[1],input_first.shape[2],input_first.shape[3],6])
        #input = torch.cat((input_first,input_second),0)
        #input = torch.cat((input,input_third),0)


        input[:,:,:,:,0] = input_first
        input[:,:,:,:,1] = input_second
        input[:,:,:,:,2] = input_third
        input[:,:,:,:,3] = input_forth
        input[:,:,:,:,4] = input_fifth
        input[:,:,:,:,5] = input_sixth
        #input[:,:,:,:,6] = input_seventh
        #input[:,:,:,:,7] = input_eigth
        
        target = target_first
         

        target = target.cuda()
 
        
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        pred_score, alphas_part_max, alphas_org = model(input_var)
        loss = criterion(pred_score, target_var) + criterion1(alphas_part_max, alphas_org)
        # try:
        weights_loss = criterion1(alphas_part_max, alphas_org)

        # measure accuracy and record loss
        prec1 = accuracy(pred_score.data, target, topk=(1,))
        losses.update(loss.data[0], input.size(0))
        weights_losses.update(weights_loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val} ({batch_time.avg})\t'
                  'Loss {loss.val} ({loss.avg})\t'
                  'weights_loss {weights_loss.val} ({weights_loss.avg})\t'
                  'Prec@1 {top1.val} ({top1.avg})\t'
                  .format(
                   i, len(val_loader), batch_time=batch_time, loss=losses, weights_loss=weights_losses,  
                   top1=top1))

    print(' * Prec@1 {top1.avg} '
          .format(top1=top1))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):

    full_filename = os.path.join(args.model_dir, filename)
    full_bestname = os.path.join(args.model_dir, 'model_best.pth.tar')
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    torch.save(state, full_filename)
    epoch_num = state['epoch']
    if epoch_num%5==0 and epoch_num>=0:
        torch.save(state, full_filename.replace('checkpoint','checkpoint_'+str(epoch_num).zfill(3)))
    if is_best:
        shutil.copyfile(full_filename, full_bestname)


class AverageMeter(object): 
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate_proportion(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #lr = args.lr * (0.1 ** (epoch // 30))
    if epoch in [int(args.epochs*0.3), int(args.epochs*0.5), int(args.epochs*0.8)]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1

def adjust_learning_rate_fixed(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #lr = args.lr * (0.1 ** (epoch // 30))
    if epoch in [15, 25, 40]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
