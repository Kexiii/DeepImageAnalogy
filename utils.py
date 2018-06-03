from __future__ import print_function, division

import cv2
import numpy as np
import torch

def ts2np(x):
    x = x.squeeze(0)
    x = x.cpu().numpy()
    x = x.transpose(1,2,0)
    return x

def np2ts(x):
    x = x.transpose(2,0,1)
    x = torch.from_numpy(x)
    x = x.unsqueeze(0)
    return x.cuda()

def load_image(file,resize_ratio=1.0):
    ori_img = cv2.imread(file)
    res = cv2.resize(ori_img, (int(ori_img.shape[0]*resize_ratio), int(ori_img.shape[1]*resize_ratio)),
                    interpolation=cv2.INTER_CUBIC)
    print("Output Image Shape: {}".format(res.shape))
    return res
    
def normalize(feature_map):
    response = torch.sum(feature_map*feature_map, dim=1, keepdim=True)
    normed_feature_map = feature_map/torch.sqrt(response)
    response = (response-torch.min(response))/(torch.max(response)-torch.min(response))
    return  normed_feature_map, response
    
def blend(response, f_a, r_bp, alpha=0.8, tau=0.05):
    """Equotion(4) stated in the paper
    We use the indicator function instead of sigmoid here according to the official implementation:
    https://github.com/msracver/Deep-Image-Analogy
    """
    weight = (response > tau).type(torch.FloatTensor) * alpha
    weight = weight.expand(1, f_a.size(1), weight.size(2), weight.size(3))
    weight = weight.cuda()
    f_ap = f_a*weight + r_bp*(1. - weight)
    return f_ap

