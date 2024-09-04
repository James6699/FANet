import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.sph_utils import xy2angle, pruned_inf, to_3dsphere, get_face
from utils.sph_utils import face_to_cube_coord, norm_to_cube
import copy
import numpy as np
from scipy import ndimage
import utils.gnomonic_projection as gp
import utils.spherical_coordinates as sc
import utils.polygon as polygon
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
        # l4 = self.layer4(l3)

        multi_level_features = [l1, l2,l3]
        # multi_level_features = [l1, l2, l3, l4]

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
    def __init__(self,inter_channel,padding_size,blend):
        super(Tan2Equi, self).__init__()
        self.base_order = 0
        self.inter_channel = inter_channel
        self.padding_size = padding_size
        self.blend = blend

    def get_icosahedron_parameters(self,triangle_index, padding_size=0.0):
        """
        Get icosahedron's tangent face's paramters.
        Get the tangent point theta and phi. Known as the theta_0 and phi_0.
        The erp image origin as top-left corner

        :return the tangent face's tangent point and 3 vertices's location.
        """
        # reference: https://en.wikipedia.org/wiki/Regular_icosahedron
        radius_circumscribed = np.sin(2 * np.pi / 5.0)
        radius_inscribed = np.sqrt(3) / 12.0 * (3 + np.sqrt(5))
        radius_midradius = np.cos(np.pi / 5.0)

        # the tangent point
        theta_0 = None
        phi_0 = None

        # the 3 points of tangent triangle in spherical coordinate
        triangle_point_00_theta = None
        triangle_point_00_phi = None
        triangle_point_01_theta = None
        triangle_point_01_phi = None
        triangle_point_02_theta = None
        triangle_point_02_phi = None

        # triangles' row/col range in the erp image
        # erp_image_row_start = None
        # erp_image_row_stop = None
        # erp_image_col_start = None
        # erp_image_col_stop = None

        theta_step = 2.0 * np.pi / 5.0
        # 1) the up 5 triangles
        if 0 <= triangle_index <= 4:
            # tangent point of inscribed spheric
            theta_0 = - np.pi + theta_step / 2.0 + triangle_index * theta_step
            phi_0 = np.pi / 2 - np.arccos(radius_inscribed / radius_circumscribed)
            # the tangent triangle points coordinate in tangent image
            triangle_point_00_theta = -np.pi + triangle_index * theta_step
            triangle_point_00_phi = np.arctan(0.5)
            triangle_point_01_theta = -np.pi + np.pi * 2.0 / 5.0 / 2.0 + triangle_index * theta_step
            triangle_point_01_phi = np.pi / 2.0
            triangle_point_02_theta = -np.pi + (triangle_index + 1) * theta_step
            triangle_point_02_phi = np.arctan(0.5)

            # # availied area of ERP image
            # erp_image_row_start = 0
            # erp_image_row_stop = (np.pi / 2 - np.arctan(0.5)) / np.pi
            # erp_image_col_start = 1.0 / 5.0 * triangle_index_temp
            # erp_image_col_stop = 1.0 / 5.0 * (triangle_index_temp + 1)

        # 2) the middle 10 triangles
        # 2-0) middle-up triangles
        if 5 <= triangle_index <= 9:
            triangle_index_temp = triangle_index - 5
            # tangent point of inscribed spheric
            theta_0 = - np.pi + theta_step / 2.0 + triangle_index_temp * theta_step
            phi_0 = np.pi / 2.0 - np.arccos(radius_inscribed / radius_circumscribed) - 2 * np.arccos(radius_inscribed / radius_midradius)
            # the tangent triangle points coordinate in tangent image
            triangle_point_00_theta = -np.pi + triangle_index_temp * theta_step
            triangle_point_00_phi = np.arctan(0.5)
            triangle_point_01_theta = -np.pi + (triangle_index_temp + 1) * theta_step
            triangle_point_01_phi = np.arctan(0.5)
            triangle_point_02_theta = -np.pi + theta_step / 2.0 + triangle_index_temp * theta_step
            triangle_point_02_phi = -np.arctan(0.5)

            # # availied area of ERP image
            # erp_image_row_start = (np.arccos(radius_inscribed / radius_circumscribed) + np.arccos(radius_inscribed / radius_midradius)) / np.pi
            # erp_image_row_stop = (np.pi / 2.0 + np.arctan(0.5)) / np.pi
            # erp_image_col_start = 1 / 5.0 * triangle_index_temp
            # erp_image_col_stop = 1 / 5.0 * (triangle_index_temp + 1)

        # 2-1) the middle-down triangles
        if 10 <= triangle_index <= 14:
            triangle_index_temp = triangle_index - 10
            # tangent point of inscribed spheric
            theta_0 = - np.pi + triangle_index_temp * theta_step
            phi_0 = -(np.pi / 2.0 - np.arccos(radius_inscribed / radius_circumscribed) - 2 * np.arccos(radius_inscribed / radius_midradius))
            # the tangent triangle points coordinate in tangent image
            triangle_point_00_phi = -np.arctan(0.5)
            triangle_point_00_theta = - np.pi - theta_step / 2.0 + triangle_index_temp * theta_step
            if triangle_index_temp == 10:
                # cross the ERP image boundary
                triangle_point_00_theta = triangle_point_00_theta + 2 * np.pi
            triangle_point_01_theta = -np.pi + triangle_index_temp * theta_step
            triangle_point_01_phi = np.arctan(0.5)
            triangle_point_02_theta = - np.pi + theta_step / 2.0 + triangle_index_temp * theta_step
            triangle_point_02_phi = -np.arctan(0.5)

            # # availied area of ERP image
            # erp_image_row_start = (np.pi / 2.0 - np.arctan(0.5)) / np.pi
            # erp_image_row_stop = (np.pi - np.arccos(radius_inscribed / radius_circumscribed) - np.arccos(radius_inscribed / radius_midradius)) / np.pi
            # erp_image_col_start = 1.0 / 5.0 * triangle_index_temp - 1.0 / 5.0 / 2.0
            # erp_image_col_stop = 1.0 / 5.0 * triangle_index_temp + 1.0 / 5.0 / 2.0

        # 3) the down 5 triangles
        if 15 <= triangle_index <= 19:
            triangle_index_temp = triangle_index - 15
            # tangent point of inscribed spheric
            theta_0 = - np.pi + triangle_index_temp * theta_step
            phi_0 = - (np.pi / 2 - np.arccos(radius_inscribed / radius_circumscribed))
            # the tangent triangle points coordinate in tangent image
            triangle_point_00_theta = - np.pi - theta_step / 2.0 + triangle_index_temp * theta_step
            triangle_point_00_phi = -np.arctan(0.5)
            triangle_point_01_theta = - np.pi + theta_step / 2.0 + triangle_index_temp * theta_step
            # cross the ERP image boundary
            if triangle_index_temp == 15:
                triangle_point_01_theta = triangle_point_01_theta + 2 * np.pi
            triangle_point_01_phi = -np.arctan(0.5)
            triangle_point_02_theta = - np.pi + triangle_index_temp * theta_step
            triangle_point_02_phi = -np.pi / 2.0

            # # spherical coordinate (0,0) is in the center of ERP image
            # erp_image_row_start = (np.pi / 2.0 + np.arctan(0.5)) / np.pi
            # erp_image_row_stop = 1.0
            # erp_image_col_start = 1.0 / 5.0 * triangle_index_temp - 1.0 / 5.0 / 2.0
            # erp_image_col_stop = 1.0 / 5.0 * triangle_index_temp + 1.0 / 5.0 / 2.0

        tangent_point = [theta_0, phi_0]

        # the 3 vertices in tangent image's gnomonic coordinate
        triangle_points_tangent = []
        triangle_points_tangent.append(gp.gnomonic_projection(triangle_point_00_theta, triangle_point_00_phi, theta_0, phi_0))
        triangle_points_tangent.append(gp.gnomonic_projection(triangle_point_01_theta, triangle_point_01_phi, theta_0, phi_0))
        triangle_points_tangent.append(gp.gnomonic_projection(triangle_point_02_theta, triangle_point_02_phi, theta_0, phi_0))

        # pading the tangent image
        triangle_points_tangent_no_pading = copy.deepcopy(triangle_points_tangent)  # Needed for NN blending
        triangle_points_tangent_pading = polygon.enlarge_polygon(triangle_points_tangent, padding_size)

        # print("triangle_points_tangent_no_pading\n\n",triangle_points_tangent_no_pading)
        # if padding_size != 0.0:
        triangle_points_tangent = copy.deepcopy(triangle_points_tangent_pading)

        # the points in spherical location
        triangle_points_sph = []
        for index in range(3):
            tri_pading_x, tri_pading_y = triangle_points_tangent_pading[index]  # tangent平面上
            triangle_point_theta, triangle_point_phi = gp.reverse_gnomonic_projection(tri_pading_x, tri_pading_y, theta_0, phi_0)
            triangle_points_sph.append([triangle_point_theta, triangle_point_phi])

        # compute bounding box of the face in spherical coordinate
        availied_sph_area = []
        availied_sph_area = np.array(copy.deepcopy(triangle_points_sph))


        triangle_points_tangent_pading = np.array(triangle_points_tangent_pading)

        # print("triangle_points_tangent_pading\n\n",triangle_points_tangent_pading)

        # 横坐标选最中间，纵坐标选一个最中间
        point_insert_x = np.sort(triangle_points_tangent_pading[:, 0])[1]
        
        point_insert_y = np.sort(triangle_points_tangent_pading[:, 1])[1]

        # print(point_insert_x,point_insert_y)
        availied_sph_area = np.append(availied_sph_area, [gp.reverse_gnomonic_projection(point_insert_x, point_insert_y, theta_0, phi_0)], axis=0)

        # print("availied_sph_area\n\n",availied_sph_area)


        # the bounding box of the face with spherical coordinate
        availied_ERP_area_sph = []  # [min_longitude, max_longitude, min_latitude, max_lantitude]

        if 0 <= triangle_index <= 4:
            if padding_size > 0.0:
                availied_ERP_area_sph.append(-np.pi)
                availied_ERP_area_sph.append(np.pi)
            else:
                availied_ERP_area_sph.append(np.amin(availied_sph_area[:, 0]))
                availied_ERP_area_sph.append(np.amax(availied_sph_area[:, 0]))

            availied_ERP_area_sph.append(np.pi / 2.0)
            availied_ERP_area_sph.append(np.amin(availied_sph_area[:, 1]))  # the ERP Y axis direction as down

        elif 15 <= triangle_index <= 19:
            if padding_size > 0.0:
                availied_ERP_area_sph.append(-np.pi)
                availied_ERP_area_sph.append(np.pi)
            else:
                availied_ERP_area_sph.append(np.amin(availied_sph_area[:, 0]))
                availied_ERP_area_sph.append(np.amax(availied_sph_area[:, 0]))
            availied_ERP_area_sph.append(np.amax(availied_sph_area[:, 1]))
            availied_ERP_area_sph.append(-np.pi / 2.0)
            
        else:
            availied_ERP_area_sph.append(np.amin(availied_sph_area[:, 0]))
            availied_ERP_area_sph.append(np.amax(availied_sph_area[:, 0]))
            availied_ERP_area_sph.append(np.amax(availied_sph_area[:, 1]))
            availied_ERP_area_sph.append(np.amin(availied_sph_area[:, 1]))

        # else:
        #     triangle_points_sph.append([triangle_point_00_theta, triangle_point_00_theta])
        #     triangle_points_sph.append([triangle_point_01_theta, triangle_point_01_theta])
        #     triangle_points_sph.append([triangle_point_02_theta, triangle_point_02_theta])

        #     availied_ERP_area.append(erp_image_row_start)
        #     availied_ERP_area.append(erp_image_row_stop)
        #     availied_ERP_area.append(erp_image_col_start)
        #     availied_ERP_area.append(erp_image_col_stop)

        return {"tangent_point": tangent_point, "triangle_points_tangent": triangle_points_tangent,
                "triangle_points_sph": triangle_points_sph,
                "triangle_points_tangent_nopad": triangle_points_tangent_no_pading, "availied_ERP_area": availied_ERP_area_sph}


    def erp2ico_image(self,erp_image, tangent_image_width, padding_size=0.0, full_face_image=False):
        """Project the equirectangular image to 20 triangle images.
        Project the equirectangular image to level-0 icosahedron.
        :param erp_image: the input equirectangular image, RGB image should be 3 channel [H,W,3], depth map' shape should be [H,W].
        :type erp_image: numpy array, [height, width, 3]
        :param tangent_image_width: the output triangle image size, defaults to 480
        :type tangent_image_width: int, optional
        :param padding_size: the output face image' padding size
        :type padding_size: float
        :param full_face_image: If yes project all pixels in the face image, no just project the pixels in the face triangle, defaults to False
        :type full_face_image: bool, optional
        :param depthmap_enable: if project depth map, return the each pixel's 3D points location in current camera coordinate system.
        :type depthmap_enable: bool
        :return: If erp is rgb image:
                    1) a list contain 20 triangle images, the image is 4 channels, invalided pixel's alpha is 0, others is 1.
                    2)
                    3) None.
                If erp is depth map:
                    1) a list contain 20 triangle images depth maps in tangent coordinate system.  The subimage's depth is 3D point could depth value.
                    2) 
                    3) 3D point cloud in tangent coordinate system. The pangent point cloud coordinate system is same as the world coordinate system. +y down, +x right and +z forward.
        :rtype: 
        """
        # if full_face_image:
        #     log.debug("Generate rectangle tangent image.")
        # else:
        #     log.debug("Generating triangle tangent image.")
            
        # ERP image size
        depthmap_enable = False
        if len(erp_image.shape) == 3:
            if np.shape(erp_image)[2] == 4:
                # log.info("project ERP image is 4 channels RGB map")
                erp_image = erp_image[:, :, 0:3]
            # log.info("project ERP image 3 channels RGB map")
        elif len(erp_image.shape) == 2:
            # log.info("project ERP image is single channel depth map")
            erp_image = np.expand_dims(erp_image, axis=2)
            depthmap_enable = True

        erp_image_height = np.shape(erp_image)[0]
        erp_image_width = np.shape(erp_image)[1]
        channel_number = np.shape(erp_image)[2]

        if erp_image_width != erp_image_height * 2:
            raise Exception("the ERP image dimession is {}".format(np.shape(erp_image)))

        tangent_image_list = []
        tangent_image_gnomonic_xy = [] # [x[height, width], y[height, width]]
        tangent_3dpoints_list = []
        tangent_sphcoor_list = []

        tangent_image_height = int((tangent_image_width / 2.0) / np.tan(np.radians(30.0)) + 0.5)

        # generate tangent images
        for triangle_index in range(0, 20):
            # log.debug("generate the tangent image {}".format(triangle_index))
            triangle_param = self.get_icosahedron_parameters(triangle_index, padding_size)

            tangent_triangle_vertices = np.array(triangle_param["triangle_points_tangent"])
            # the face gnomonic range in tangent space
            gnomonic_x_min = np.amin(tangent_triangle_vertices[:, 0], axis=0)
            gnomonic_x_max = np.amax(tangent_triangle_vertices[:, 0], axis=0)
            gnomonic_y_min = np.amin(tangent_triangle_vertices[:, 1], axis=0)
            gnomonic_y_max = np.amax(tangent_triangle_vertices[:, 1], axis=0)
            gnom_range_x = np.linspace(gnomonic_x_min, gnomonic_x_max, num=tangent_image_width, endpoint=True)
            gnom_range_y = np.linspace(gnomonic_y_max, gnomonic_y_min, num=tangent_image_height, endpoint=True)
            gnom_range_xv, gnom_range_yv = np.meshgrid(gnom_range_x, gnom_range_y)

            # the tangent triangle points coordinate in tangent image
            inside_list = np.full(gnom_range_xv.shape[:2], True, dtype=np.bool)
            if not full_face_image:
                gnom_range_xyv = np.stack((gnom_range_xv.flatten(), gnom_range_yv.flatten()), axis=1)
                pixel_eps = (gnomonic_x_max - gnomonic_x_min) / (tangent_image_width)
                inside_list = gp.inside_polygon_2d(gnom_range_xyv, tangent_triangle_vertices, on_line=True, eps=pixel_eps)
                inside_list = inside_list.reshape(gnom_range_xv.shape)

            # project to tangent image
            tangent_point = triangle_param["tangent_point"]
            tangent_triangle_theta_, tangent_triangle_phi_ = gp.reverse_gnomonic_projection(gnom_range_xv[inside_list], gnom_range_yv[inside_list], tangent_point[0], tangent_point[1])

            tangent_sphcoor_list.append(
                np.stack((tangent_triangle_theta_.reshape(gnom_range_xv.shape), tangent_triangle_phi_.reshape(gnom_range_xv.shape)))
            )

            # tansform from spherical coordinate to pixel location
            tangent_triangle_erp_pixel_x, tangent_triangle_erp_pixel_y = sc.sph2erp(tangent_triangle_theta_, tangent_triangle_phi_, erp_image_height, sph_modulo=True)

            # get the tangent image pixels value
            tangent_gnomonic_range = [gnomonic_x_min, gnomonic_x_max, gnomonic_y_min, gnomonic_y_max]
            tangent_image_x, tangent_image_y = gp.gnomonic2pixel(gnom_range_xv[inside_list], gnom_range_yv[inside_list],
                                                                0.0, tangent_image_width, tangent_image_height, tangent_gnomonic_range)

            if depthmap_enable:
                tangent_image = np.full([tangent_image_height, tangent_image_width, channel_number], -1.0)
            else:
                tangent_image = np.full([tangent_image_height, tangent_image_width, channel_number], 255.0)
            for channel in range(0, np.shape(erp_image)[2]):
                tangent_image[tangent_image_y, tangent_image_x, channel] = \
                    ndimage.map_coordinates(erp_image[:, :, channel], [tangent_triangle_erp_pixel_y, tangent_triangle_erp_pixel_x], order=1, mode='wrap', cval=255.0)

            # if the ERP image is depth map, get camera coordinate system 3d points
            tangent_3dpoints = None
            if depthmap_enable:
                # convert the spherical depth map value to tangent image coordinate depth value  
                center2pixel_length = np.sqrt(np.square(gnom_range_xv[inside_list])  + np.square(gnom_range_yv[inside_list]) + np.ones_like(gnom_range_yv[inside_list]))
                center2pixel_length = center2pixel_length.reshape((tangent_image_height, tangent_image_width, channel_number))
                tangent_3dpoints_z = np.divide(tangent_image , center2pixel_length)
                tangent_image = tangent_3dpoints_z

                # get x and y
                tangent_3dpoints_x = np.multiply(tangent_3dpoints_z , gnom_range_xv[inside_list].reshape((tangent_image_height, tangent_image_width, channel_number)))
                tangent_3dpoints_y = np.multiply(tangent_3dpoints_z , gnom_range_yv[inside_list].reshape((tangent_image_height, tangent_image_width, channel_number)))
                tangent_3dpoints = np.concatenate([tangent_3dpoints_x, -tangent_3dpoints_y, tangent_3dpoints_z], axis =2)
                
            # set the pixels outside the boundary to transparent
            # tangent_image[:, :, 3] = 0
            # tangent_image[tangent_image_y, tangent_image_x, 3] = 255
            tangent_image_list.append(tangent_image)
            tangent_3dpoints_list.append(tangent_3dpoints)

        # get the tangent image's gnomonic coordinate
        tangent_image_gnomonic_x = gnom_range_xv[inside_list].reshape((tangent_image_height, tangent_image_width))
        tangent_image_gnomonic_xy.append(tangent_image_gnomonic_x)
        tangent_image_gnomonic_y = gnom_range_yv[inside_list].reshape((tangent_image_height, tangent_image_width))
        tangent_image_gnomonic_xy.append(tangent_image_gnomonic_y)

        return tangent_image_list, tangent_sphcoor_list, [tangent_3dpoints_list, tangent_image_gnomonic_xy]


    def ico2erp_image(self,tangent_images, erp_image_height, padding_size=0.0, blender_method=None):

        """Stitch the level-0 icosahedron's tangent image to ERP image.

        blender_method:
            - None: just sample the triangle area;
            - Mean: the mean value on the overlap area.

        TODO there are seam on the stitched erp image.

        :param tangent_images: 20 tangent images in order.
        :type tangent_images: a list of numpy
        :param erp_image_height: the output erp image's height.
        :type erp_image_height: int
        :param padding_size: the face image's padding size
        :type padding_size: float
        :param blender_method: the method used to blend sub-images. 
        :type blender_method: str
        :return: the stitched ERP image
        :type numpy
        """
        # if len(tangent_images) != 20:
        #     log.error("The tangent's images triangle number is {}.".format(len(tangent_images)))

        if len(tangent_images[0].shape) == 3:
            images_channels_number = tangent_images[0].shape[2]
            if images_channels_number == 4:
                # log.debug("the face image is RGBA image, convert the output to RGB image.")
                images_channels_number = 3
                
        elif len(tangent_images[0].shape) == 2:
            # log.info("project single channel disp or depth map")
            images_channels_number = 1

        erp_image_width = erp_image_height * 2
        erp_image = np.full([erp_image_height, erp_image_width, images_channels_number], 0, np.float64)

        tangent_image_height = tangent_images[0].shape[0]
        tangent_image_width = tangent_images[0].shape[1]

        erp_weight_mat = np.zeros((erp_image_height, erp_image_width), dtype=np.float64)
        # stitch all tangnet images to ERP image
        for triangle_index in range(0, 20):
            # log.debug("stitch the tangent image {}".format(triangle_index))
            triangle_param = self.get_icosahedron_parameters(triangle_index, padding_size)

            # 1) get all tangent triangle's available pixels coordinate
            availied_ERP_area = triangle_param["availied_ERP_area"]
            erp_image_col_start, erp_image_row_start = sc.sph2erp(availied_ERP_area[0], availied_ERP_area[2], erp_image_height, sph_modulo=False)
            erp_image_col_stop, erp_image_row_stop = sc.sph2erp(availied_ERP_area[1], availied_ERP_area[3], erp_image_height, sph_modulo=False)

            # process the image boundary
            erp_image_col_start = int(erp_image_col_start) if int(erp_image_col_start) > 0 else int(erp_image_col_start - 0.5)
            erp_image_col_stop = int(erp_image_col_stop + 0.5) if int(erp_image_col_stop) > 0 else int(erp_image_col_stop)
            erp_image_row_start = int(erp_image_row_start) if int(erp_image_row_start) > 0 else int(erp_image_row_start - 0.5)
            erp_image_row_stop = int(erp_image_row_stop + 0.5) if int(erp_image_row_stop) > 0 else int(erp_image_row_stop)

            triangle_x_range = np.linspace(erp_image_col_start, erp_image_col_stop, erp_image_col_stop - erp_image_col_start + 1)
            triangle_y_range = np.linspace(erp_image_row_start, erp_image_row_stop, erp_image_row_stop - erp_image_row_start + 1)
            triangle_xv, triangle_yv = np.meshgrid(triangle_x_range, triangle_y_range)
            # process the wrap around
            triangle_xv = np.remainder(triangle_xv, erp_image_width)
            triangle_yv = np.remainder(triangle_yv, erp_image_height)

            # 2) sample the pixel value from tanget image
            # project spherical coordinate to tangent plane
            spherical_uv = sc.erp2sph([triangle_xv, triangle_yv], erp_image_height=erp_image_height, sph_modulo=False)
            theta_0 = triangle_param["tangent_point"][0]
            phi_0 = triangle_param["tangent_point"][1]
            tangent_xv, tangent_yv = gp.gnomonic_projection(spherical_uv[0, :, :], spherical_uv[1, :, :], theta_0, phi_0)

            # the pixels in the tangent triangle
            triangle_points_tangent = np.array(triangle_param["triangle_points_tangent"])
            gnomonic_x_min = np.amin(triangle_points_tangent[:, 0], axis=0)
            gnomonic_x_max = np.amax(triangle_points_tangent[:, 0], axis=0)
            gnomonic_y_min = np.amin(triangle_points_tangent[:, 1], axis=0)
            gnomonic_y_max = np.amax(triangle_points_tangent[:, 1], axis=0)

            tangent_gnomonic_range = [gnomonic_x_min, gnomonic_x_max, gnomonic_y_min, gnomonic_y_max]
            pixel_eps = abs(tangent_xv[0, 0] - tangent_xv[0, 1]) / (2 * tangent_image_width)

            if len(tangent_images[0].shape) == 2:
                tangent_images_subimage = np.expand_dims(tangent_images[triangle_index], axis=2)
            else:
                tangent_images_subimage = tangent_images[triangle_index]

            if blender_method is None:
                available_pixels_list = gp.inside_polygon_2d(np.stack((tangent_xv.flatten(), tangent_yv.flatten()), axis=1),
                                                            triangle_points_tangent, on_line=True, eps=pixel_eps).reshape(tangent_xv.shape)

                # the tangent available gnomonic coordinate sample the pixel from the tangent image
                tangent_xv, tangent_yv = gp.gnomonic2pixel(tangent_xv[available_pixels_list], tangent_yv[available_pixels_list],
                                                        0.0, tangent_image_width, tangent_image_height, tangent_gnomonic_range)

                for channel in range(0, images_channels_number):
                    erp_image[triangle_yv[available_pixels_list].astype(np.int), triangle_xv[available_pixels_list].astype(np.int), channel] = \
                        ndimage.map_coordinates(tangent_images_subimage[:, :, channel], [tangent_yv, tangent_xv], order=1, mode='constant', cval=255)

            elif blender_method == "max":
                triangle_points_tangent = [[gnomonic_x_min, gnomonic_y_max],
                                        [gnomonic_x_max, gnomonic_y_max],
                                        [gnomonic_x_max, gnomonic_y_min],
                                        [gnomonic_x_min, gnomonic_y_min]]
                available_pixels_list = gp.inside_polygon_2d(np.stack((tangent_xv.flatten(), tangent_yv.flatten()), axis=1),
                                                            triangle_points_tangent, on_line=True, eps=pixel_eps).reshape(tangent_xv.shape)

                tangent_xv, tangent_yv = gp.gnomonic2pixel(tangent_xv[available_pixels_list], tangent_yv[available_pixels_list],
                                                        0.0, tangent_image_width, tangent_image_height, tangent_gnomonic_range)
                
                for channel in range(0, images_channels_number):
                    erp_face_image = ndimage.map_coordinates(tangent_images_subimage[:, :, channel], [tangent_yv, tangent_xv], order=1, mode='constant', cval=255)
                    flag_erp = erp_image[triangle_yv[available_pixels_list].astype(np.int), triangle_xv[available_pixels_list].astype(np.int), channel]
                    erp_image[triangle_yv[available_pixels_list].astype(np.int), triangle_xv[available_pixels_list].astype(np.int), channel] = np.maximum(erp_face_image.astype(np.float64), flag_erp.astype(np.float64))

            elif blender_method == "mean":
                triangle_points_tangent = [[gnomonic_x_min, gnomonic_y_max],
                                        [gnomonic_x_max, gnomonic_y_max],
                                        [gnomonic_x_max, gnomonic_y_min],
                                        [gnomonic_x_min, gnomonic_y_min]]
                available_pixels_list = gp.inside_polygon_2d(np.stack((tangent_xv.flatten(), tangent_yv.flatten()), axis=1),
                                                            triangle_points_tangent, on_line=True, eps=pixel_eps).reshape(tangent_xv.shape)

                tangent_xv, tangent_yv = gp.gnomonic2pixel(tangent_xv[available_pixels_list], tangent_yv[available_pixels_list],
                                                        0.0, tangent_image_width, tangent_image_height, tangent_gnomonic_range)
                for channel in range(0, images_channels_number):
                    erp_face_image = ndimage.map_coordinates(tangent_images_subimage[:, :, channel], [tangent_yv, tangent_xv], order=1, mode='constant', cval=255)
                    erp_image[triangle_yv[available_pixels_list].astype(np.int), triangle_xv[available_pixels_list].astype(np.int), channel] += erp_face_image.astype(np.float64)

                face_weight_mat = np.ones(erp_face_image.shape, np.float64)
                erp_weight_mat[triangle_yv[available_pixels_list].astype(np.int64), triangle_xv[available_pixels_list].astype(np.int64)] += face_weight_mat

        # compute the final optical flow base on weight
        if blender_method == "mean":
            # erp_flow_weight_mat = np.full(erp_flow_weight_mat.shape, erp_flow_weight_mat.max(), np.float) # debug
            non_zero_weight_list = erp_weight_mat != 0
            # if not np.all(non_zero_weight_list):
            #     log.warn("the optical flow weight matrix contain 0.")
            for channel_index in range(0, images_channels_number):
                erp_image[:, :, channel_index][non_zero_weight_list] = erp_image[:, :, channel_index][non_zero_weight_list] / erp_weight_mat[non_zero_weight_list]

        return erp_image



    def forward(self,tan):
        
        [n,b,c,h,w] = tan.shape
        # list1=[]
        # tan = tan.permute(0,2,1, 3,4).contiguous()
        # for i in range (0,n):
        #     list=[]
        #     for j in range (0,self.inter_channel):
        #         output = tan[i, j, :, :,: ].detach().cpu().numpy()
        #         output=self.ico2erp_image(output,w*4,self.padding_size,self.blend)
        #         list.append(output)
        #     list = np.array(list)
        #     list1.append(list)
        # list1 = np.array(list1)
        # erp = torch.from_numpy(list1).squeeze()
        
        self.shape = [h*4,w*8]
        self.sample_order = m.log2(h)  # ?????
        #cube = cube.view(self.batch,b//self.batch,c,h,w)
        tan = tan.permute(0,2,1, 3,4).contiguous()
        erp = tangent_images_to_equirectangular(tan.float(), self.shape,
                                                    self.base_order, self.sample_order)                                    
        return erp.float()

    # def forward(self,tan):
    #     [n,b,c,h,w] = tan.shape
        
    #     self.shape = [h*4,w*8]
    #     self.sample_order = m.log2(h)  # ?????
    #     #cube = cube.view(self.batch,b//self.batch,c,h,w)
    #     tan = tan.permute(0,2,1, 3,4).contiguous()
    #     erp = tangent_images_to_equirectangular(tan.float(), self.shape,
    #                                              self.base_order, self.sample_order)                                    
    #     return erp.float()



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
    def __init__(self, inter_channel,padding_size,blend):
        super(T2EB, self).__init__()
        self.projection = Tan2Equi(inter_channel,padding_size,blend)
        self.fusion = nn.Sequential(nn.Conv2d(inter_channel, inter_channel, kernel_size=3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channel),
                                    )

    def forward(self, cubefeature_list):
        cubefeatures = torch.stack(cubefeature_list, dim=1)
        equi_feature = self.projection(cubefeatures)
        #print(equi_feature.size())
        equi_feature = equi_feature.to('cuda')
        # equi_feature = equi_feature.unsqueeze(0)
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
    def __init__(self, inplanes, outplanes,padding_size,blend):
        super(PFA, self).__init__()
        self.c2eprojection = T2EB(inplanes,padding_size,blend)

        self.feature_selection = nn.Sequential(nn.Conv2d(inplanes * 2, outplanes, kernel_size=3, stride=1, padding=1, bias=False),
                                               nn.BatchNorm2d(outplanes),
                                               )

        self.msfusion = MSF(outplanes, outplanes)
        self.channelattention = CA(outplanes)

        self.spatialattention_v1 = nn.Sequential(nn.Conv2d(outplanes, 1, kernel_size=1, stride=1, bias=False),
                                              nn.BatchNorm2d(1),
                                              nn.Softmax(dim=1),
                                              ) # 这里输出应该是(1,2,h,w)
        # self.spatialattention = nn.Sequential(nn.Conv2d(outplanes, 2, kernel_size=1, stride=1, bias=False),
        #                                       nn.BatchNorm2d(2),
        #                                       nn.Softmax(dim=1),
        #                                       ) # 这里输出应该是(1,2,h,w)
        self.Conv1x1 = nn.Sequential(nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
                                            nn.BatchNorm2d(outplanes))

    def forward(self, x):
        proj_feature = self.c2eprojection(x[1])  # cube 2 erp feature
        equi_feature = F.interpolate(x[0], size=proj_feature.size()[2:], mode='bilinear', align_corners=True)
        channelatten =self.channelattention(equi_feature)
        spa_atten0 = self.spatialattention_v1(equi_feature)[:, 0, :, :].unsqueeze(dim=1)
        equi_feature_ = channelatten * equi_feature * spa_atten0
        equi_feature_ = self.feature_selection(torch.cat((equi_feature_,equi_feature),dim=1))
        out = self.Conv1x1(self.msfusion(self.feature_selection(torch.cat((equi_feature_, proj_feature), dim=1))))
        # msfeature = self.msfusion(self.feature_selection(torch.cat((equi_feature, proj_feature), dim=1)))

        # channelatten =self.channelattention(msfeature) # (1,128,1,1)

        # spa_atten0 = self.spatialattention(msfeature)[:, 0, :, :].unsqueeze(dim=1)
        # spa_atten1 = self.spatialattention(msfeature)[:, 1, :, :].unsqueeze(dim=1)
        # fusedfeature = self.msfusion(self.feature_selection(torch.cat((channelatten * equi_feature * spa_atten0,
        #                                                                channelatten * proj_feature * spa_atten1), dim=1)))
        # out = self.feature_selection(torch.cat((msfeature, fusedfeature), dim=1))
        # out = out.mean(dim=0)
        # out = torch.cat((proj_feature,equi_feature),dim=1)
        # out = self.ConcateFusion(out)
        # out = (proj_feature+equi_feature)/2
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

        # l4Feature = l4
        #l3Feature = l3 + F.interpolate(l4, size=l3.size()[2:], mode='bilinear', align_corners=True)
        l3Feature = l3 
        l2Feature = l2 + F.interpolate(l3Feature, size=l2.size()[2:], mode='bilinear', align_corners=True)
        # l2Feature = l2
        l1Feature = l1 + F.interpolate(l2Feature, size=l1.size()[2:], mode='bilinear', align_corners=True)

        #return [l1Feature, l2Feature, l3Feature, l4Feature]
        return [l1Feature, l2Feature,l3Feature]


class MLFA(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MLFA, self).__init__()
        # self.fusion = nn.Sequential(nn.Conv2d(in_channel*3, in_channel, 1, bias=False),
        #                             nn.BatchNorm2d(in_channel),
        #                             )

        # self.spa_attention = nn.Sequential(nn.Conv2d(in_channel, 1, 1, bias=False),
        #                                           nn.BatchNorm2d(1),
        #                                           nn.Sigmoid())

        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)

        # self.weight_level = nn.Sequential(nn.Conv2d(in_channel, 3, kernel_size=1, bias=False),
        #                                   nn.BatchNorm2d(3),
        #                                   nn.Dropout2d(p=0.5),
        #                                   )

        # self.selectattention = nn.Sequential(nn.Conv2d(6, 3, kernel_size=1, bias=False),
        #                                      nn.BatchNorm2d(3),
        #                                      nn.Softmax(dim=1),
        #                                      )

        # self.ConcateFusion = nn.Sequential(nn.Conv2d(in_channel+in_channel, in_channel, 3, padding=1, bias=False),
        #                                    nn.BatchNorm2d(in_channel),
        #                                    nn.ReLU(inplace=True),
        #                                    nn.Dropout2d(p=0.5),
        #                                    nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
        #                                    nn.BatchNorm2d(out_channel),
        #                                    )
        # self.feature_selection = nn.Sequential(nn.Conv2d(in_channel * 2, in_channel, kernel_size=3, stride=1, padding=1, bias=False),
        #                                        nn.BatchNorm2d(in_channel),
        #                                        nn.ReLU(inplace=True),
        #                                        nn.Conv2d(in_channel, in_channel, kernel_size=1, bias=False),
        #                                        nn.BatchNorm2d(in_channel)
        #                                        )
        self.feature_selection = nn.Sequential(nn.Conv2d(in_channel*2, in_channel, kernel_size=1, bias=False),
                                               nn.BatchNorm2d(in_channel)
                                               )
        self.conv1X1_1 = nn.Sequential(nn.Conv2d(in_channel*3, in_channel, kernel_size=1, bias=False),
                                                nn.BatchNorm2d(in_channel))
        self.conv1X1_2 = nn.Sequential(nn.Conv2d(in_channel*2, in_channel, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(in_channel))
        self.conv1X1 = nn.Sequential(nn.Conv2d(in_channel*3, out_channel, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(out_channel))
        # self.feature_selection_end = nn.Sequential(nn.Conv2d(in_channel * 2, in_channel, kernel_size=3, stride=1, padding=1, bias=False),
        #                                        nn.BatchNorm2d(in_channel),
        #                                        nn.ReLU(inplace=True),
        #                                        nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
        #                                        nn.BatchNorm2d(out_channel)
        #                                        )

    def forward(self, x1, x2,x3):
        x111 = self.conv1X1_1(torch.cat((x1
                          ,F.interpolate(x2, size=x1.size()[2:], mode='bilinear', align_corners=True)
                          ,F.interpolate(x3, size=x1.size()[2:], mode='bilinear', align_corners=True)),dim=1))
        x222 = self.conv1X1_2(torch.cat((x2
                          ,F.interpolate(x3, size=x2.size()[2:], mode='bilinear', align_corners=True)),dim=1))
        x333 = x3
        x123 = self.conv1X1(torch.cat((x111
                         ,F.interpolate(x222, size=x111.size()[2:], mode='bilinear', align_corners=True)
                         ,F.interpolate(x333, size=x111.size()[2:], mode='bilinear', align_corners=True)),dim=1))
        Concatefused_output = x123
        # x23features = torch.cat((x2,
        #                          F.interpolate(x3, size=x2.size()[2:], mode='bilinear', align_corners=True)), dim=1)
        # x23features = self.feature_selection(x23features)
        # x123features = torch.cat((x1,
        #                          F.interpolate(x23features, size=x1.size()[2:], mode='bilinear', align_corners=True)), dim=1)
        # x123features = self.feature_selection(x123features)
        # Concatefused_output = self.conv1X1(x123features)


        # x12features = torch.cat((x1,
        #                          F.interpolate(x2, size=x1.size()[2:], mode='bilinear', align_corners=True)), dim=1)
        # x123features = torch.cat((x12features, F.interpolate(x3, size=x12features.size()[2:], mode='bilinear', align_corners=True)), dim=1)
        # # x1234features = torch.cat((x123features,
        # #                            F.interpolate(x4, size=x123features.size()[2:], mode='bilinear', align_corners=True)), dim=1)
        # #fused_feat = self.fusion(x1234features)
        # fused_feat = self.fusion(x123features)

        # level_weight = self.selectattention(torch.cat((self.weight_level(self.avg_pool(fused_feat)),
        #                                                 self.weight_level(self.max_pool(fused_feat))), dim=1))
        # level_weight1 = level_weight[:, 0, :, :].unsqueeze(dim=1).expand(-1, x1.size(1), -1, -1)
        # level_weight2 = level_weight[:, 1, :, :].unsqueeze(dim=1).expand(-1, x2.size(1), -1, -1)
        # level_weight3 = level_weight[:, 2, :, :].unsqueeze(dim=1).expand(-1, x3.size(1), -1, -1)
        # # level_weight4 = level_weight[:, 3, :, :].unsqueeze(dim=1).expand(-1, x4.size(1), -1, -1)
        # # levelweights = torch.cat((level_weight1, level_weight2, level_weight3, level_weight4), dim=1)
        # levelweights = torch.cat((level_weight1, level_weight2, level_weight3), dim=1)
        # fusedspa_attention = self.spa_attention(fused_feat)
        # # weighted_fusedfeat = self.fusion(x1234features*fusedspa_attention*levelweights)
        # weighted_fusedfeat = self.fusion(x123features*fusedspa_attention*levelweights)

        # Concatefused_output = self.ConcateFusion(torch.cat((fused_feat, weighted_fusedfeat), dim=1))
        return Concatefused_output

  # 0924：修改MFLA模块为DSS部分
class FANet(nn.Module):
    def __init__(self, num_classes,padding_size,blend):
        super(FANet, self).__init__()
        self.inter_channel = 64
        self.basenet = resnet50()

        self.FPN = FPNFusion(self.inter_channel)

        self.PFFusion = PFA(self.inter_channel, self.inter_channel,padding_size,blend)
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
        # cubefeatures_B = self.basenet(x[1])
        # cubefeatures_D = self.basenet(x[2])
        # cubefeatures_F = self.basenet(x[3])
        # cubefeatures_L = self.basenet(x[4])
        # cubefeatures_R = self.basenet(x[5])
        # cubefeatures_T = self.basenet(x[6])

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
                                                 equifeatures[1],equifeatures[2]
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

        # cubeB_FPN_feature_list = self.FPN([cubefeatures_B[0],
        #                                           cubefeatures_B[1],
        #                                           cubefeatures_B[2],
        #                                           cubefeatures_B[3]])

        # cubeD_FPN_feature_list = self.FPN([cubefeatures_D[0],
        #                                           cubefeatures_D[1],
        #                                           cubefeatures_D[2],
        #                                           cubefeatures_D[3]])

        # cubeF_FPN_feature_list = self.FPN([cubefeatures_F[0],
        #                                           cubefeatures_F[1],
        #                                           cubefeatures_F[2],
        #                                           cubefeatures_F[3]])

        # cubeL_FPN_feature_list = self.FPN([cubefeatures_L[0],
        #                                           cubefeatures_L[1],
        #                                           cubefeatures_L[2],
        #                                           cubefeatures_L[3]])

        # cubeR_FPN_feature_list = self.FPN([cubefeatures_R[0],
        #                                           cubefeatures_R[1],
        #                                           cubefeatures_R[2],
        #                                           cubefeatures_R[3]])

        # cubeT_FPN_feature_list = self.FPN([cubefeatures_T[0],
        #                                           cubefeatures_T[1],
        #                                           cubefeatures_T[2],
        #                                           cubefeatures_T[3]])

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

        # l4_cubefeatures = [cubeB_FPN_feature_list[3],
        #                    cubeD_FPN_feature_list[3],
        #                    cubeF_FPN_feature_list[3],
        #                    cubeL_FPN_feature_list[3],
        #                    cubeR_FPN_feature_list[3],
        #                    cubeT_FPN_feature_list[3]]
        # l3_cubefeatures = [cubeB_FPN_feature_list[2],
        #                    cubeD_FPN_feature_list[2],
        #                    cubeF_FPN_feature_list[2],
        #                    cubeL_FPN_feature_list[2],
        #                    cubeR_FPN_feature_list[2],
        #                    cubeT_FPN_feature_list[2]]
        # l2_cubefeatures = [cubeB_FPN_feature_list[1],
        #                    cubeD_FPN_feature_list[1],
        #                    cubeF_FPN_feature_list[1],
        #                    cubeL_FPN_feature_list[1],
        #                    cubeR_FPN_feature_list[1],
        #                    cubeT_FPN_feature_list[1]]
        # l1_cubefeatures = [cubeB_FPN_feature_list[0],
        #                    cubeD_FPN_feature_list[0],
        #                    cubeF_FPN_feature_list[0],
        #                    cubeL_FPN_feature_list[0],
        #                    cubeR_FPN_feature_list[0],
        #                    cubeT_FPN_feature_list[0]]
        # fused_l4feature = self.PFFusion([equi_FPN_feature_list[3], l4_cubefeatures])
        # fused_l3feature = self.PFFusion([equi_FPN_feature_list[2], l3_cubefeatures])
        # fused_l2feature = self.PFFusion([equi_FPN_feature_list[1], l2_cubefeatures])
        # fused_l1feature = self.PFFusion([equi_FPN_feature_list[0], l1_cubefeatures])

        # fused_l4feature = self.PFFusion([equi_FPN_feature_list[3], l4_tanfeatures])
        fused_l3feature = self.PFFusion([equi_FPN_feature_list[2], l3_tanfeatures])
        fused_l2feature = self.PFFusion([equi_FPN_feature_list[1], l2_tanfeatures])
        fused_l1feature = self.PFFusion([equi_FPN_feature_list[0], l1_tanfeatures])

        # Concatefused_output = self.MLFCombination(fused_l1feature, fused_l2feature, fused_l3feature, fused_l4feature)
        Concatefused_output = self.MLFCombination(fused_l1feature, fused_l2feature, fused_l3feature)
        midl1out = self.midfeature_output(fused_l1feature)
        midl2out = self.midfeature_output(fused_l2feature)
        midl3out = self.midfeature_output(fused_l3feature)
        # midl4out = self.midfeature_output(fused_l4feature)

        output_list = [Concatefused_output, midl1out, midl2out,midl3out]
        return output_list

    def _init_weight(self):
        for name, m in self.named_modules():
            if 'basenet' not in name:
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
