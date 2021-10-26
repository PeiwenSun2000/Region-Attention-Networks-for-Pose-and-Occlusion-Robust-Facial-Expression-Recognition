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
#from ResNet_MN_Val_all import resnet18, resnet50, resnet101
from attention_rank_loss import resnet18, resnet34
from val_part_attention_sample import MsCelebDataset, CaffeCrop
# from part_attention_sample import MsCelebDataset, CaffeCrop
import scipy.io as sio   
import numpy as np
import pdb
import torch._utils
import numpy as np
import time
import pandas as pd
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CelebA Training')
parser.add_argument('--img_dir_val', metavar='DIR', default='./dataset/FERPlus_reshape_crop_folder/Test/', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='./model/mytrain/2021-10-20-11:43:56/checkpoint_060.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--model_dir','-m', default='./model/mytrain/'+time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())+os.sep, type=str)
parser.add_argument('--end2end', default=True,\
        help='if true, using end2end with dream block, else, using naive architecture')

def get_dict_val_data():
    txt_path = './dataset/FERPlus_test_txts/'
    df1 = pd.read_csv("./dataset/FERPlus_list/dlib_ferplus_val_center_crop_range_list.txt", sep=' ', header=None)
    df2 = pd.read_csv("./dataset/FERPlus_list/dlib_ferplus_val_center_crop_range_label.txt", sep=' ', header=None)
    df_all = pd.merge(df1,df2,left_index=True,right_index=True,how='outer')
    df_all.columns=['path','num_pics','label']
    dict_val_data={}
    for index,row in df_all.iterrows():
        dict_val_data[row['path']]=[str(row["num_pics"]),str(row["label"])]
    return dict_val_data

def get_val_data(list_txt,dict_data,frame_num):
    # list_txt是一个不带后缀的文件名
    caffe_crop = CaffeCrop('test')
    # txt_path = '/media/sdc/kwang/ferplus/pose_test/test_txt/'

    # pdb.set_trace()
    # args.img_dir_val,是个路径，val_list_file, val_label_file,这两个文件都要打开的
    val_dataset =  MsCelebDataset(args.img_dir_val,list_txt,dict_data, 
        transforms.Compose([caffe_crop,transforms.ToTensor()]))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,batch_size=frame_num, shuffle=False,num_workers=args.workers, pin_memory=True)
    
    # pdb.set_trace()
    return val_loader


def main(arch,resume):
    global args
    args = parser.parse_args()
    arch = arch.split('_')[0]
    model = None
    assert(arch in ['resnet18','resnet34','resnet50','resnet101'])
    if arch == 'resnet18':
        model = resnet18(end2end=args.end2end)
    if arch == 'resnet34':
        model = resnet34(end2end=args.end2end)
    if arch == 'resnet50':
        model = resnet50(nverts=nverts,faces=faces,shapeMU=shapeMU,shapePC=shapePC, num_classes=class_num, end2end=args.end2end)
    if arch == 'resnet101':
        model = resnet101(pretrained=False, num_classes=class_num,\
                extract_feature=True, end2end=end2end)


    # params=model.state_dict() #获得模型的原始状态以及参数。
    # for name,parameters in model.named_parameters():
    #     print(name,':',parameters.size())
    # pdb.set_trace()



    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    assert(os.path.isfile(resume))
    #pdb.set_trace()
    checkpoint = torch.load(resume)
    #pdb.set_trace()
    model.load_state_dict(checkpoint['state_dict'])

    cudnn.benchmark = True

    val_nn_txt = './dataset/FERPlus_list/val_ferplus_mn.txt'
    val_nn_files = open(val_nn_txt,'r')
    correct = 0
    video_num = 0
    output_task1 = open('ferplus_mn_score.txt','w+')
    
    dict_val_data = get_dict_val_data()

    for val_nn_file in val_nn_files:
        
        record = val_nn_file.strip().split()
        #pdb.set_trace()
        list_txt = record[0]
        label_txt = record[1]
        # frame_num = record[2]
        frame_num = 6
        video_num = video_num +1
        video_name = list_txt

        index_xiahua = video_name.find('_')
        video_name = list(video_name)
        video_name[index_xiahua] = '/'
        video_name = ''.join(video_name)

        index_xiahua = label_txt.find('_')
        label_txt = list(label_txt)
        label_txt[index_xiahua] = '/'
        label_txt = ''.join(label_txt)
        #pdb.set_trace()
        # 下面这行源码有
        video_name = video_name[0:-4]
        
        print('video_name',video_name)
        # video_name=fer0032279 0_fer0032241_label.txt frame_num 感觉本地化应该一直是0
        val_loader = get_val_data(video_name,dict_val_data,int(frame_num))
        
        for i,(input,label) in enumerate(val_loader):
            label = label.numpy()
            with torch.no_grad():
                input_var = torch.autograd.Variable(input)
            # pdb.set_trace()
            ## 下面是新加的
            input_var = torch.unsqueeze(input_var,4)
            #output, f_need_fix, feature_standard = model(input_var)
            input_var=input_var.transpose(0, 4).contiguous()
            print("input_var.shape",input_var.shape)
            output = model(input_var)

            output_write = output
            output_write =output_write[0]
            output_write = output_write.cpu().data.numpy()
            print('output_write',output_write)
            # pdb.set_trace()
            output_of_softmax = F.softmax(output[0],dim=1)
            output_of_softmax_ = output_of_softmax.cpu().data.numpy()
            pred_class = np.argmax(output_of_softmax_)
            #output_of_softmax_ = output_of_softmax_[0]
            #output_task1.write(video_name+' '+str(output_of_softmax_[0])+' '+str(output_of_softmax_[1])+' '+str(output_of_softmax_[2])+' '+str(output_of_softmax_[3])+' '+str(output_of_softmax_[4])+' '+str(output_of_softmax_[5])+' '+str(output_of_softmax_[6])+'\n')
            output_task1.write(video_name+' '+str(pred_class)+'\n')
            pred_final = output_of_softmax[0].data.max(0,keepdim=True)[1]
            #pdb.set_trace()
            #pred_final = pred_final.cpu().data.numpy()
            pred_final = pred_final.cpu().numpy()
            if int(label[0]) == int(pred_final[0]):
               correct = correct +1
               print('predict right label',label[0])
    print('accuracy', float(correct)/video_num)
    print('correct',correct)
    print('video_num',video_num)

if __name__ == '__main__':
    
    infos = [ ('resnet18_naive', './model/mytrain/2021-10-20-11:43:56/model_best.pth.tar'), 
               ]


    for arch, model_path in infos:
        print("{} {}".format(arch, model_path))
        main(arch, model_path)
        
        print()