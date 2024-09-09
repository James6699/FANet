import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.sph_utils import xy2angle, pruned_inf, to_3dsphere, get_face
from utils.sph_utils import face_to_cube_coord, norm_to_cube
from spherical_distortion.functional import create_tangent_images, tangent_images_to_equirectangular
import math as m



__all__ = ['ResNet', 'resnet50']

models_path = {'resnet50': './pretrain_weights/resnet50.pth',
                }


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding_mode='replicate',
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, 2, 2]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               padding_mode='replicate', bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        # multi_level_features = [l1]
        # multi_level_features = [l1, l2, l3]
        multi_level_features = [l1, l2, l3,l4]

        return multi_level_features

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = torch.load(models_path[arch])
        model.load_state_dict(state_dict)
    return model

def resnet50(pretrained=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    # return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, **kwargs)#？
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, **kwargs)#？

class CA(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False),
                                   nn.ReLU(inplace=True),
                                   )
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.selectattention = nn.Sequential(nn.Conv2d(in_planes + in_planes, in_planes, kernel_size=1, bias=False),
                                             nn.BatchNorm2d(in_planes),
                                             )

    def forward(self, x):

        avg_out = self.fc2(self.fc1(self.avg_pool(x)))
        max_out = self.fc2(self.fc1(self.max_pool(x)))

        x1 = torch.cat((avg_out, max_out),dim=1)
        attention = self.sigmoid(self.selectattention(x1))
        return attention


class Cube2Equi(nn.Module):
    def __init__(self):
        super(Cube2Equi, self).__init__()
        self.scale_c = 1

    def _config(self, input_w):
        in_width = input_w * self.scale_c
        out_w = in_width * 4
        out_h = in_width * 2

        face_map = torch.zeros((out_h, out_w))

        YY, XX = torch.meshgrid(torch.arange(out_h), torch.arange(out_w))

        theta, phi = xy2angle(XX, YY, out_w, out_h)
        theta = pruned_inf(theta)
        phi = pruned_inf(phi)

        _x, _y, _z = to_3dsphere(theta, phi, 1)
        face_map = get_face(_x, _y, _z, face_map)
        x_o, y_o = face_to_cube_coord(face_map, _x, _y, _z)

        out_coord = torch.stack([x_o, y_o], dim=2) 
        out_coord = norm_to_cube(out_coord, in_width)

        return out_coord, face_map

    def forward(self, input_data):
        warpout_list = []
        for input in torch.split(input_data, 1 , dim=0):
            inputdata = input.squeeze(dim=0).clone().detach().cuda()
            assert inputdata.size()[2] == inputdata.size()[3]
            gridf, face_map = self._config(inputdata.size()[2])
            gridf = gridf.clone().detach().cuda()
            face_map = face_map.clone().detach().cuda()
            out_w = int(gridf.size(1))
            out_h = int(gridf.size(0))
            in_width = out_w/4
            depth = inputdata.size(1)
            warp_out = torch.zeros([1, depth, out_h, out_w], dtype=torch.float32).clone().detach().requires_grad_().cuda()
            gridf = (gridf-torch.max(gridf)/2)/(torch.max(gridf)/2)

            for f_idx in range(0, 6):
                face_mask = face_map == f_idx
                expanded_face_mask = face_mask.expand(1, inputdata.size(1), face_mask.size(0), face_mask.size(1))
                warp_out[expanded_face_mask] = nn.functional.grid_sample(torch.unsqueeze(inputdata[f_idx], 0), torch.unsqueeze(gridf, 0), align_corners=True)[expanded_face_mask]

            warpout_list.append(warp_out)
        return torch.cat(warpout_list, dim=0)

class Tan2Equi(nn.Module):
    def __init__(self):
        super(Tan2Equi, self).__init__()
        self.base_order = 0
 
    def forward(self,tan):
        # tan = F.interpolate(tan, size=[128,128], mode='bilinear', align_corners=True)
        # print(tan.shape)
        [n,b,c,h,w] = tan.shape
        # self.shape = [h,w]
        self.shape = [h*2,w*4]
        self.sample_order = m.log2(h) + self.base_order # ?????
        # self.sample_order = 7
        #cube = cube.view(self.batch,b//self.batch,c,h,w)
        tan = tan.permute(0,2,1, 3,4).contiguous()

        erp = tangent_images_to_equirectangular(tan.float(), self.shape,
                                                 self.base_order, self.sample_order)                                    
        return erp.float()



class C2EB(nn.Module):
    def __init__(self, inter_channel):
        super(C2EB, self).__init__()
        self.projection = Cube2Equi()
        self.fusion = nn.Sequential(nn.Conv2d(inter_channel, inter_channel, kernel_size=3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channel),
                                    )

    def forward(self, cubefeature_list):
        cubefeatures = torch.stack(cubefeature_list, dim=1)
        equi_feature = self.projection(cubefeatures)
        #print(equi_feature.size())
        output_equi_feature = self.fusion(equi_feature)
        return output_equi_feature

class T2EB(nn.Module):
    def __init__(self, inter_channel):
        super(T2EB, self).__init__()
        self.projection = Tan2Equi()
        self.fusion = nn.Sequential(nn.Conv2d(inter_channel, inter_channel, kernel_size=3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channel),
                                    )

    def forward(self, cubefeature_list):
        cubefeatures = torch.stack(cubefeature_list, dim=1)
        equi_feature = self.projection(cubefeatures)
        #print(equi_feature.size())
        output_equi_feature = self.fusion(equi_feature)
        return output_equi_feature

class MSF(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(MSF, self).__init__()
        dilations = [1, 6, 12, 18]
        # print(inplanes,outplanes)
        self.atrous_conv1 = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=1, padding=0, dilation=dilations[0], bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )
        self.atrous_conv2 = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )
        self.atrous_conv3 = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )
        self.atrous_conv4 = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=dilations[3], dilation=dilations[3], bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )
        #self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, outplanes, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(outplanes),
                                             nn.ReLU(inplace=True))
        self.fusionconv = nn.Sequential(nn.Conv2d(inplanes * 5, outplanes, 1, bias=False),
                                        nn.BatchNorm2d(outplanes),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(0.5)
                                        )

    def forward(self, x):
        x1 = self.atrous_conv1(x)
        x2 = self.atrous_conv2(x)
        x3 = self.atrous_conv3(x)
        x4 = self.atrous_conv4(x)
        # print(x.size())
        # x_51 = self.pool(x)
        # print(x_51.size())
        # 去掉了池化
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        out = self.fusionconv(x)

        return out


class PFA(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(PFA, self).__init__()
        self.c2eprojection = C2EB(inplanes)

        self.feature_selection = nn.Sequential(nn.Conv2d(inplanes * 2, outplanes, kernel_size=3, stride=1, padding=1, bias=False),
                                               nn.BatchNorm2d(outplanes),
                                               )

        self.msfusion = MSF(outplanes, outplanes)
        self.channelattention = CA(outplanes)
       
        self.spatialattention = nn.Sequential(nn.Conv2d(outplanes, 2, kernel_size=1, stride=1, bias=False),
                                              nn.BatchNorm2d(2),
                                              nn.Softmax(dim=1),
                                              ) # 这里输出应该是(1,2,h,w)

    def forward(self, x):
        proj_feature = self.c2eprojection(x[1])  # cube 2 erp feature
        equi_feature = F.interpolate(x[0], size=proj_feature.size()[2:], mode='bilinear', align_corners=True)
        msfeature = self.msfusion(self.feature_selection(torch.cat((equi_feature, proj_feature), dim=1)))

        channelatten =self.channelattention(msfeature) # (1,128,1,1)

        spa_atten0 = self.spatialattention(msfeature)[:, 0, :, :].unsqueeze(dim=1)
        spa_atten1 = self.spatialattention(msfeature)[:, 1, :, :].unsqueeze(dim=1)
        fusedfeature = self.msfusion(self.feature_selection(torch.cat((channelatten * equi_feature * spa_atten0,
                                                                       channelatten * proj_feature * spa_atten1), dim=1)))
        out = self.feature_selection(torch.cat((msfeature, fusedfeature), dim=1))
        return out

# 特征金字塔
class FPNFusion(nn.Module):
    def __init__(self, inchannel):
        super(FPNFusion, self).__init__()
        # self.latlayer4 = nn.Sequential(nn.Conv2d(2048, inchannel, kernel_size=1, stride=1, bias=False),
        #                                nn.BatchNorm2d(inchannel),
        #                                )  
        self.latlayer3 = nn.Sequential(nn.Conv2d(1024, inchannel, kernel_size=1, stride=1, bias=False),
                                       nn.BatchNorm2d(inchannel),
                                       )
        self.latlayer2 = nn.Sequential(nn.Conv2d(512, inchannel, kernel_size=1, stride=1, bias=False),
                                       nn.BatchNorm2d(inchannel),
                                       )
        self.latlayer1 = nn.Sequential(nn.Conv2d(256, inchannel, kernel_size=1, stride=1, bias=False),
                                       nn.BatchNorm2d(inchannel),
                                       )

    def forward(self, x):
        l1 = self.latlayer1(x[0])
        l2 = self.latlayer2(x[1])
        l3 = self.latlayer3(x[2])
        # l4 = self.latlayer4(x[3])
        # print(l1.size(),l2.size(),l3.size(),l4.size())
        # l4Feature = l4
        # l3Feature = l3 + F.interpolate(l4, size=l3.size()[2:], mode='bilinear', align_corners=True)
        l3Feature = l3
        l2Feature = l2 + F.interpolate(l3Feature, size=l2.size()[2:], mode='bilinear', align_corners=True)
        #l2Feature = l2
        l1Feature = l1 + F.interpolate(l2Feature, size=l1.size()[2:], mode='bilinear', align_corners=True)
        #l1Feature = l1
        # print(l1Feature.size(),l2Feature.size(),l3Feature.size(),l4Feature.size())
        return [l1Feature, l2Feature, l3Feature]
        # return [l1Feature, l2Feature, l3Feature,l4Feature]
        # return [l1Feature]


class MLFA(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MLFA, self).__init__()
        self.fusion = nn.Sequential(nn.Conv2d(in_channel * 3, in_channel, 1, bias=False),
                                    nn.BatchNorm2d(in_channel),
                                    )

        self.spa_attention = nn.Sequential(nn.Conv2d(in_channel, 1, 1, bias=False),
                                                  nn.BatchNorm2d(1),
                                                  nn.Sigmoid())

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.weight_level = nn.Sequential(nn.Conv2d(in_channel, 3, kernel_size=1, bias=False),
                                          nn.BatchNorm2d(3),
                                          nn.Dropout2d(p=0.5),
                                          )

        self.selectattention = nn.Sequential(nn.Conv2d(6, 3, kernel_size=1, bias=False),
                                             nn.BatchNorm2d(3),
                                             nn.Softmax(dim=1),
                                             )

        self.ConcateFusion = nn.Sequential(nn.Conv2d(in_channel+in_channel, in_channel, 3, padding=1, bias=False),
                                           nn.BatchNorm2d(in_channel),
                                           nn.ReLU(inplace=True),
                                           nn.Dropout2d(p=0.5),
                                           nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
                                           nn.BatchNorm2d(out_channel),
                                           )

    def forward(self, x1,x2,x3):
        x12features = torch.cat((x1,
                                 F.interpolate(x2, size=x1.size()[2:], mode='bilinear', align_corners=True)), dim=1)
        x123features = torch.cat((x12features, F.interpolate(x3, size=x12features.size()[2:], mode='bilinear', align_corners=True)), dim=1)
        # x1234features = torch.cat((x123features,
        #                            F.interpolate(x4, size=x123features.size()[2:], mode='bilinear', align_corners=True)), dim=1)
        fused_feat = self.fusion(x123features)
        #fused_feat = self.fusion(x1)

        level_weight = self.selectattention(torch.cat((self.weight_level(self.avg_pool(fused_feat)),
                                                        self.weight_level(self.max_pool(fused_feat))), dim=1))
        level_weight1 = level_weight[:, 0, :, :].unsqueeze(dim=1).expand(-1, x1.size(1), -1, -1)
        level_weight2 = level_weight[:, 1, :, :].unsqueeze(dim=1).expand(-1, x2.size(1), -1, -1)
        level_weight3 = level_weight[:, 2, :, :].unsqueeze(dim=1).expand(-1, x3.size(1), -1, -1)
        # level_weight4 = level_weight[:, 3, :, :].unsqueeze(dim=1).expand(-1, x4.size(1), -1, -1)
        levelweights = torch.cat((level_weight1, level_weight2, level_weight3), dim=1)
        #levelweights = torch.cat((level_weight1), dim=1)
        # levelweights  = level_weight1
        fusedspa_attention = self.spa_attention(fused_feat)
        weighted_fusedfeat = self.fusion(x123features*fusedspa_attention*levelweights)
        #weighted_fusedfeat = self.fusion(x1*fusedspa_attention*levelweights)

        Concatefused_output = self.ConcateFusion(torch.cat((fused_feat, weighted_fusedfeat), dim=1))
        return Concatefused_output


class FANet(nn.Module):
    def __init__(self, num_classes):
        super(FANet, self).__init__()
        self.inter_channel = 64
        self.basenet = resnet50()

        self.FPN = FPNFusion(self.inter_channel)

        self.PFFusion = PFA(self.inter_channel, self.inter_channel)
        self.MLFCombination = MLFA(self.inter_channel, num_classes)

        self.midfeature_output = nn.Sequential(nn.Dropout2d(p=0.5),
                                               nn.Conv2d(self.inter_channel, num_classes, kernel_size=1, stride=1, bias=False),
                                               nn.BatchNorm2d(num_classes),
                                               )
        self._init_weight()

    def _upsample_sum(self, x, y):
        _, _, H, W = x.size()
        return x + F.interpolate(y, size=(H, W), mode='bilinear', align_corners=True)

    def forward(self, x):
        equifeatures = self.basenet(x[0])

        tanfeatures_0 = self.basenet(x[1])
        tanfeatures_1 = self.basenet(x[2])
        tanfeatures_2 = self.basenet(x[3])
        tanfeatures_3 = self.basenet(x[4])
        tanfeatures_4 = self.basenet(x[5])
        tanfeatures_5 = self.basenet(x[6])
        tanfeatures_6 = self.basenet(x[7])
        tanfeatures_7 = self.basenet(x[8])
        tanfeatures_8 = self.basenet(x[9])
        tanfeatures_9 = self.basenet(x[10])
        tanfeatures_10 = self.basenet(x[11])
        tanfeatures_11 = self.basenet(x[12])
        tanfeatures_12 = self.basenet(x[13])
        tanfeatures_13 = self.basenet(x[14])
        tanfeatures_14 = self.basenet(x[15])
        tanfeatures_15 = self.basenet(x[16])
        tanfeatures_16 = self.basenet(x[17])
        tanfeatures_17 = self.basenet(x[18])
        tanfeatures_18 = self.basenet(x[19])
        tanfeatures_19 = self.basenet(x[20])
 

        # FPN输出融合特征通道数为inter_channel
        equi_FPN_feature_list = self.FPN([equifeatures[0],
                                          equifeatures[1],
                                          equifeatures[2]
                                                 ])


        tan0_FPN_feature_list = self.FPN([tanfeatures_0[0],tanfeatures_0[1],tanfeatures_0[2]])
        tan1_FPN_feature_list = self.FPN([tanfeatures_1[0],tanfeatures_1[1],tanfeatures_1[2]])
        tan2_FPN_feature_list = self.FPN([tanfeatures_2[0],tanfeatures_2[1],tanfeatures_2[2]])
        tan3_FPN_feature_list = self.FPN([tanfeatures_3[0],tanfeatures_3[1],tanfeatures_3[2]])
        tan4_FPN_feature_list = self.FPN([tanfeatures_4[0],tanfeatures_4[1],tanfeatures_4[2]])
        tan5_FPN_feature_list = self.FPN([tanfeatures_5[0],tanfeatures_5[1],tanfeatures_5[2]])
        tan6_FPN_feature_list = self.FPN([tanfeatures_6[0],tanfeatures_6[1],tanfeatures_6[2]])
        tan7_FPN_feature_list = self.FPN([tanfeatures_7[0],tanfeatures_7[1],tanfeatures_7[2]])
        tan8_FPN_feature_list = self.FPN([tanfeatures_8[0],tanfeatures_8[1],tanfeatures_8[2]])
        tan9_FPN_feature_list = self.FPN([tanfeatures_9[0],tanfeatures_9[1],tanfeatures_9[2]])
        tan10_FPN_feature_list = self.FPN([tanfeatures_10[0],tanfeatures_10[1],tanfeatures_10[2]])
        tan11_FPN_feature_list = self.FPN([tanfeatures_11[0],tanfeatures_11[1],tanfeatures_11[2]])
        tan12_FPN_feature_list = self.FPN([tanfeatures_12[0],tanfeatures_12[1],tanfeatures_12[2]])
        tan13_FPN_feature_list = self.FPN([tanfeatures_13[0],tanfeatures_13[1],tanfeatures_13[2]])
        tan14_FPN_feature_list = self.FPN([tanfeatures_14[0],tanfeatures_14[1],tanfeatures_14[2]])
        tan15_FPN_feature_list = self.FPN([tanfeatures_15[0],tanfeatures_15[1],tanfeatures_15[2]])
        tan16_FPN_feature_list = self.FPN([tanfeatures_16[0],tanfeatures_16[1],tanfeatures_16[2]])
        tan17_FPN_feature_list = self.FPN([tanfeatures_17[0],tanfeatures_17[1],tanfeatures_17[2]])
        tan18_FPN_feature_list = self.FPN([tanfeatures_18[0],tanfeatures_18[1],tanfeatures_18[2]])
        tan19_FPN_feature_list = self.FPN([tanfeatures_19[0],tanfeatures_19[1],tanfeatures_19[2]])       

        # l4_tanfeatures = [tan0_FPN_feature_list[3],tan1_FPN_feature_list[3],tan2_FPN_feature_list[3],tan3_FPN_feature_list[3],tan4_FPN_feature_list[3],
        #                   tan5_FPN_feature_list[3],tan6_FPN_feature_list[3],tan7_FPN_feature_list[3],tan8_FPN_feature_list[3],tan9_FPN_feature_list[3],
        #                   tan10_FPN_feature_list[3],tan11_FPN_feature_list[3],tan12_FPN_feature_list[3],tan13_FPN_feature_list[3],tan14_FPN_feature_list[3],
        #                   tan15_FPN_feature_list[3],tan16_FPN_feature_list[3],tan17_FPN_feature_list[3],tan18_FPN_feature_list[3],tan19_FPN_feature_list[3]]

        l3_tanfeatures = [tan0_FPN_feature_list[2],tan1_FPN_feature_list[2],tan2_FPN_feature_list[2],tan3_FPN_feature_list[2],tan4_FPN_feature_list[2],
                          tan5_FPN_feature_list[2],tan6_FPN_feature_list[2],tan7_FPN_feature_list[2],tan8_FPN_feature_list[2],tan9_FPN_feature_list[2],
                          tan10_FPN_feature_list[2],tan11_FPN_feature_list[2],tan12_FPN_feature_list[2],tan13_FPN_feature_list[2],tan14_FPN_feature_list[2],
                          tan15_FPN_feature_list[2],tan16_FPN_feature_list[2],tan17_FPN_feature_list[2],tan18_FPN_feature_list[2],tan19_FPN_feature_list[2]]

        l2_tanfeatures = [tan0_FPN_feature_list[1],tan1_FPN_feature_list[1],tan2_FPN_feature_list[1],tan3_FPN_feature_list[1],tan4_FPN_feature_list[1],
                          tan5_FPN_feature_list[1],tan6_FPN_feature_list[1],tan7_FPN_feature_list[1],tan8_FPN_feature_list[1],tan9_FPN_feature_list[1],
                          tan10_FPN_feature_list[1],tan11_FPN_feature_list[1],tan12_FPN_feature_list[1],tan13_FPN_feature_list[1],tan14_FPN_feature_list[1],
                          tan15_FPN_feature_list[1],tan16_FPN_feature_list[1],tan17_FPN_feature_list[1],tan18_FPN_feature_list[1],tan19_FPN_feature_list[1]]

        l1_tanfeatures = [tan0_FPN_feature_list[0],tan1_FPN_feature_list[0],tan2_FPN_feature_list[0],tan3_FPN_feature_list[0],tan4_FPN_feature_list[0],
                          tan5_FPN_feature_list[0],tan6_FPN_feature_list[0],tan7_FPN_feature_list[0],tan8_FPN_feature_list[0],tan9_FPN_feature_list[0],
                          tan10_FPN_feature_list[0],tan11_FPN_feature_list[0],tan12_FPN_feature_list[0],tan13_FPN_feature_list[0],tan14_FPN_feature_list[0],
                          tan15_FPN_feature_list[0],tan16_FPN_feature_list[0],tan17_FPN_feature_list[0],tan18_FPN_feature_list[0],tan19_FPN_feature_list[0]]

        # fused_l4feature = self.PFFusion([equi_FPN_feature_list[3], l4_cubefeatures])
        # fused_l3feature = self.PFFusion([equi_FPN_feature_list[2], l3_cubefeatures])
        # fused_l2feature = self.PFFusion([equi_FPN_feature_list[1], l2_cubefeatures])
        # fused_l1feature = self.PFFusion([equi_FPN_feature_list[0], l1_cubefeatures])

        # fused_l4feature = self.PFFusion([equi_FPN_feature_list[3], l4_tanfeatures])
        # fused_l3feature = self.PFFusion([equi_FPN_feature_list[2], l3_tanfeatures])
        fused_l3feature = self.PFFusion([equi_FPN_feature_list[2], l3_tanfeatures])
        fused_l2feature = self.PFFusion([equi_FPN_feature_list[1], l2_tanfeatures])
        fused_l1feature = self.PFFusion([equi_FPN_feature_list[0], l1_tanfeatures])

        # Concatefused_output = self.MLFCombination(fused_l1feature, fused_l2feature, fused_l3feature, fused_l4feature)
        Concatefused_output = self.MLFCombination(fused_l1feature, fused_l2feature, fused_l3feature)
        midl1out = self.midfeature_output(fused_l1feature)
        midl2out = self.midfeature_output(fused_l2feature)
        midl3out = self.midfeature_output(fused_l3feature)
        # midl4out = self.midfeature_output(fused_l4feature)

        # output_list = [Concatefused_output, midl1out,midl2out,midl3out,midl4out]
        output_list = [Concatefused_output, midl1out,midl2out,midl3out]
        return output_list
    def _init_weight(self):
        for name, m in self.named_modules():
            if 'basenet' not in name:
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
