import numpy as np
import cv2
import math as m
import collections
from spherical_distortion.functional import create_tangent_images, tangent_images_to_equirectangular
from scipy.interpolate import RegularGridInterpolator as interp2d
from scipy.interpolate import interp1d
from pylab import *
from utils.sph_utils import rotx, roty, rotz
import torch.nn.functional as F


class Equi2Tan:
    def __init__(self,out_dim,base_order):
        
        # self.sample_order = int(m.log2(out_dim)+base_order)
        self.sample_order = 7

        self.base_order = base_order

    def ToTanTensor(self,equi):
        batch = equi.shape[0]
        tan_imgs = create_tangent_images(equi,self.base_order,self.sample_order,
                                        return_mask=False)
        # print(tan_imgs.size())
        # tan_imgs = F.interpolate(tan_imgs, size=[256,256], mode='bilinear', align_corners=True)
        # tan_imgs = F.interpolate(tan_imgs, size=[128,128], mode='bilinear', align_corners=True)

        return tan_imgs
        # tan_imgs = tan_imgs.permute(2,0, 1, 3,4).contiguous()
        # c,h,w = tan_imgs.shape[-3:]
        # tan_imgs = tan_imgs.view(-1,c,h,w)

        # return tan_imgs.contiguous()





class Equi2Cube:
    def __init__(self, output_width, in_image, vfov=90):

        self.out = {}
        assert in_image.shape[0]*2 == in_image.shape[1]
        self.cube_out = np.array([])
        views = [[180, 0, 0], 
                 [0, -90, 0], 
                 [0, 0, 0],  
                 [-90, 0, 0],
                 [90, 0, 0], 
                 [0, 90, 0]] 

        self.inXs = []
        self.inYs = []

        vfov = vfov*np.pi/180
        views = np.array(views)*np.pi/180
        output_width = output_width
        output_height = output_width
        input_width = in_image.shape[1]
        input_height = in_image.shape[0]

        self.in_image = in_image
        self.views = views
        self.output_width = output_width
        self.output_height = output_height
        self.input_width = input_width
        self.input_height = input_height

        topLeft = np.array([-m.tan(vfov/2)*(output_width/output_height), -m.tan(vfov/2), 1])
        uv = np.array([-2*topLeft[0]/output_width, -2*topLeft[1]/output_height, 0])

        res_acos = 2*input_width
        res_atan = 2*input_height
        step_acos = np.pi / res_acos
        step_atan = np.pi / res_atan
        lookup_acos = np.append(np.array(-np.cos(np.array(np.arange(0, res_acos))*step_acos)), 1.)
        lookup_atan = np.append(np.append(np.tan(step_atan/2-pi/2),
                                np.tan(np.array(np.arange(1, res_atan))*step_atan-pi/2)), np.tan(-step_atan/2+pi/2))

        X, Y = np.meshgrid(range(output_width), range(output_height))
        X = X.flatten()
        Y = Y.flatten()
        self.X = X
        self.Y = Y
        XImage, YImage = np.meshgrid(range(input_height), range(input_width))

        for idx in range(views.shape[0]):
            yaw = views[idx, 0] 
            pitch = views[idx, 1] 
            roll = views[idx, 2]
            transform = np.dot(np.dot(roty(yaw), rotx(pitch)), rotz(roll))

            points = np.concatenate((np.concatenate((topLeft[0] + uv[0]*np.expand_dims(X, axis=0),
                                                     topLeft[1] + uv[1]*np.expand_dims(Y, axis=0)), axis=0),
                                                     topLeft[2] + uv[2]*np.ones((1, X.shape[0]))), axis=0)
            moved_points = np.dot(transform, points)

            x_points = moved_points[0, :]
            y_points = moved_points[1, :]
            z_points = moved_points[2, :]

            nxz = sqrt(x_points**2 + z_points**2)
            phi = zeros(X.shape[0])
            theta = zeros(X.shape[0])

            ind = nxz < 10e-10 
            phi[ind & (y_points > 0)] = pi/2
            phi[ind & (y_points <= 0)] = -pi/2

            ind = np.logical_not(ind)
            phi_interp = interp1d(lookup_atan, np.arange(0, res_atan+1), 'linear')
            phi[ind] = phi_interp(y_points[ind]/nxz[ind])*step_atan - (pi/2) 
            theta_interp = interp1d(lookup_acos, np.arange(0, res_acos+1), 'linear')
            theta[ind] = theta_interp(-z_points[ind]/nxz[ind])*step_acos
            theta[ind & (x_points < 0)] = -theta[ind & (x_points < 0)]

            inX = (theta / pi) * (input_width/2) + (input_width/2) + 1
            inY = (phi / (pi/2)) * (input_height/2) + (input_height/2) + 1

            inX[inX < 1] = 1
            inX[inX >= input_width-1] = input_width - 1 
            inY[inY < 1] = 1
            inY[inY >= input_height-1] = input_height-1
            self.inXs.append(inX)
            self.inYs.append(inY)

    def to_cube(self, in_image):
        for idx in range(self.views.shape[0]):
            out = self.out
            out[idx] = np.zeros((self.output_height, self.output_width, in_image.shape[2]), in_image.dtype)

            out_pix = zeros((self.X.shape[0], in_image.shape[2]))

            inX = self.inXs[idx].reshape(self.output_width, self.output_height).astype('float32')
            inY = self.inYs[idx].reshape(self.output_width, self.output_height).astype('float32')
            for c in range(in_image.shape[2]):
                out[idx][:, :, c] = cv2.remap(in_image[:, :, c], inX, inY, cv2.INTER_LINEAR)
        return out
