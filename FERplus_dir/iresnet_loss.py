import torch
from torch import nn
import pdb
import numpy as np
import torch.nn.functional as F
__all__ = ['iresnet18', 'iresnet34', 'iresnet50', 'iresnet100', 'iresnet200']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05,)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class IResNet(nn.Module):
    fc_scale = 7 * 7
    def __init__(self,
                 block, layers, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False):
        super(IResNet, self).__init__()
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05,)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.alpha = nn.Sequential(nn.Linear(512, 1),nn.Sigmoid())
        self.beta = nn.Sequential(nn.Linear(1024, 1),nn.Sigmoid())
        self.fc = nn.Linear(1024,8)

        # self.dropout = nn.Dropout(p=dropout, inplace=True)
        # self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        # self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        # nn.init.constant_(self.features.weight, 1.0)
        # self.features.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05, ),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        vs = []
        alphas = []
        for i in range(x.shape[4]):
            f = x[:,:,:,:,i]
            with torch.cuda.amp.autocast(self.fp16):
                # pdb.set_trace()
                f = self.conv1(f)
                f = self.bn1(f)
                f = self.prelu(f)
                f = self.layer1(f)
                f = self.layer2(f)
                f = self.layer3(f)
                f = self.layer4(f)
                # print("layer4-",f.shape)
                # f = self.bn2(f)
                # print("bn2-",f.shape)
                f = self.avgpool(f)
                # print("bn2-",f.shape)
                f = f.squeeze(3).squeeze(2)
                # f = torch.flatten(f, 1)
                # print("flatten-",f.shape)
                # f = self.dropout(f)
                # print("dropout-",f.shape)
                vs.append(f)
            # self.alpha(f).shape torch.Size([64, 1])
                alphas.append(self.alpha(f))
            
        vs_stack = torch.stack(vs, dim=2)
        alphas_stack = torch.stack(alphas, dim=2)
        alphas_stack = F.softmax(alphas_stack,dim=2)
        #pdb.set_trace()
        alphas_part_max = alphas_stack[:,:,0:5].max(dim=2)[0]
        # alphas_part_max = alphas_stack[:,:,0:3].mean(dim=2)
        # 论文中指出的原图是在0的位置，而代码中应该是在5的位置,下面这个原本是5
        alphas_org = alphas_stack[:,:,0]
        vm = vs_stack.mul(alphas_stack).sum(2).div(alphas_stack.sum(2))
        # pdb.set_trace()
        for i in range(len(vs)):
            vs[i] = torch.cat([vs[i], vm], dim=1)
        vs_stack_4096 = torch.stack(vs, dim=2)
        # pdb.set_trace()
        betas = []
        for index, v in enumerate(vs):
            betas.append(self.beta(v))
        betas_stack = torch.stack(betas, dim=2)
        betas_stack = F.softmax(betas_stack,dim=2)

        output = vs_stack_4096.mul(betas_stack*alphas_stack).sum(2).div((betas_stack*alphas_stack).sum(2))
        output = output.view(output.size(0), -1)
        # print(output.shape)
        pred_score = self.fc(output)

        return pred_score, alphas_part_max, alphas_org

def _iresnet(arch, block, layers, pretrained, progress, **kwargs):
    model = IResNet(block, layers, **kwargs)
    if pretrained:
        raise ValueError()
    return model


def iresnet18(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet18', IBasicBlock, [2, 2, 2, 2], pretrained,
                    progress, **kwargs)


def iresnet34(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet34', IBasicBlock, [3, 4, 6, 3], pretrained,
                    progress, **kwargs)


def iresnet50(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet50', IBasicBlock, [3, 4, 14, 3], pretrained,
                    progress, **kwargs)


def iresnet100(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet100', IBasicBlock, [3, 13, 30, 3], pretrained,
                    progress, **kwargs)


def iresnet200(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet200', IBasicBlock, [6, 26, 60, 6], pretrained,
                    progress, **kwargs)

class MyLoss(nn.Module):    
    def __init__(self):        
        super(MyLoss, self).__init__()          
    def forward(self, alphas_part_max, alphas_org):
        size = alphas_org.shape[0]
        loss_wt = 0.0
        for i in range(size):
            loss_wt += max(torch.Tensor([0]).cuda(), 0.1 - (alphas_part_max[i] - alphas_org[i]))
        return  loss_wt/size