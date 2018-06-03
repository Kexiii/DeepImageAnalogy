import torch
import numpy as np
import copy
from PatchMatchOrig import init_nnf, upSample_nnf, avg_vote, propagate, reconstruct_avg
from VGG19 import VGG19
from utils import *

def deep_image_analogy(A,BP,config):
    alphas = config['alpha']
    nnf_patch_size = config['nnf_patch_size']
    radii = config['radii']
    params = config['params']
    lr = config['lr']
    
    # preparing data
    img_A_tensor = torch.FloatTensor(A.transpose(2, 0, 1)).cuda()
    img_BP_tensor = torch.FloatTensor(BP.transpose(2, 0, 1)).cuda()
    # fake a batch dimension
    img_A_tensor = img_A_tensor.unsqueeze(0)
    img_BP_tensor = img_BP_tensor.unsqueeze(0)

    # 4.1 Preprocessing Step
    model = VGG19()
    F_A, F_A_size = model.get_features(img_tensor=img_A_tensor.clone(), layers=params['layers'])
    F_BP, F_B_size = model.get_features(img_tensor=img_BP_tensor.clone(), layers=params['layers'])
    
    # Init AP&B 's feature maps with F_A&F_BP
    F_AP = copy.deepcopy(F_A)
    F_B = copy.deepcopy(F_BP)
    
    #Note that the feature_maps now is in the order of [5,4,3,2,1,input]
    for curr_layer in range(5):
    
        #ANN init step, coarsest layer is initialized randomly,
        #Other layers is initialized using upsample technique described in the paper
        if curr_layer == 0:
            ann_AB = init_nnf(F_A_size[curr_layer][2:], F_B_size[curr_layer][2:])
            ann_BA = init_nnf(F_B_size[curr_layer][2:], F_A_size[curr_layer][2:])
        else:
            ann_AB = upSample_nnf(ann_AB, F_A_size[curr_layer][2:])
            ann_BA = upSample_nnf(ann_BA, F_B_size[curr_layer][2:])

        # According to Equotion(2), we need to normalize F_A and F_BP
        # response denotes the M in Equotion(6)
        F_A_BAR, response_A = normalize(F_A[curr_layer])
        F_BP_BAR, response_BP = normalize(F_BP[curr_layer])
        
        # F_AP&F_B is reconstructed according to Equotion(4)
        # Note that we reuse the varibale F_AP here,
        # it denotes the RBprime as is stated in the  Equotion(4) which is calculated
        # at the end of the previous iteration
        F_AP[curr_layer] = blend(response_A, F_A[curr_layer], F_AP[curr_layer], alphas[curr_layer])
        F_B[curr_layer] = blend(response_BP, F_BP[curr_layer], F_B[curr_layer], alphas[curr_layer])
        
        # Normalize F_AP&F_B as well
        F_AP_BAR, _ = normalize(F_AP[curr_layer])
        F_B_BAR, _ = normalize(F_B[curr_layer])

        # Run PatchMatch algorithm to get mapping AB and BA
        ann_AB, _ = propagate(ann_AB, ts2np(F_A_BAR), ts2np(F_AP_BAR), ts2np(F_B_BAR), ts2np(F_BP_BAR),
                        nnf_patch_size[curr_layer],params['iter'], radii[curr_layer])
        ann_BA, _ = propagate(ann_BA, ts2np(F_BP_BAR), ts2np(F_B_BAR), ts2np(F_AP_BAR), ts2np(F_A_BAR),
                        nnf_patch_size[curr_layer],params['iter'], radii[curr_layer])

        if curr_layer >= 4:
            break
        
        # The code below is used to initialize the F_AP&F_B in the next layer,
        # it generates the R_B' and R_A as is stated in Equotion(4)
        # R_B' is stored in F_AP, R_A is stored in F_B
       
        # using backpropagation to approximate feature
        
        # About why we add 2 here:
        # https://github.com/msracver/Deep-Image-Analogy/issues/30
        next_layer = curr_layer + 2
        ann_AB_upnnf2 = upSample_nnf(ann_AB, F_A_size[next_layer][2:])
        ann_BA_upnnf2 = upSample_nnf(ann_BA, F_B_size[next_layer][2:])
        F_AP_np = avg_vote(ann_AB_upnnf2, ts2np(F_BP[next_layer]), nnf_patch_size[next_layer], F_A_size[next_layer][2:],
                              F_B_size[next_layer][2:])
        F_B_np = avg_vote(ann_BA_upnnf2, ts2np(F_A[next_layer]), nnf_patch_size[next_layer], F_B_size[next_layer][2:],
                             F_A_size[next_layer][2:])
       
        # Initialize  R_B' and R_A
        F_AP[next_layer] = np2ts(F_AP_np)
        F_B[next_layer] = np2ts(F_B_np)
        
        # Warp F_BP using ann_AB, Warp F_A using ann_BA
        target_BP_np = avg_vote(ann_AB, ts2np(F_BP[curr_layer]), nnf_patch_size[curr_layer], F_A_size[curr_layer][2:],
                                F_B_size[curr_layer][2:])
        target_A_np = avg_vote(ann_BA, ts2np(F_A[curr_layer]), nnf_patch_size[curr_layer], F_B_size[curr_layer][2:],
                               F_A_size[curr_layer][2:])

        target_BP = np2ts(target_BP_np)
        target_A = np2ts(target_A_np)
        
        #LBFGS algorithm to approximate R_B' and R_A
        F_AP[curr_layer+1] = model.get_deconvoluted_feat(target_BP, curr_layer, F_AP[next_layer], lr=lr[curr_layer],
                                                         blob_layers=params['layers'])
        F_B[curr_layer+1] = model.get_deconvoluted_feat(target_A, curr_layer, F_B[next_layer], lr=lr[curr_layer],
                                                         blob_layers=params['layers'])

        if type(F_B[curr_layer + 1]) == torch.DoubleTensor:
            F_B[curr_layer + 1] = F_B[curr_layer + 1].type(torch.FloatTensor)
            F_AP[curr_layer + 1] = F_AP[curr_layer + 1].type(torch.FloatTensor)
        elif type(F_B[curr_layer + 1]) == torch.cuda.DoubleTensor:
            F_B[curr_layer + 1] = F_B[curr_layer + 1].type(torch.cuda.FloatTensor)
            F_AP[curr_layer + 1] = F_AP[curr_layer + 1].type(torch.cuda.FloatTensor)
    
    # Obtain the output according to 4.5
    img_AP = reconstruct_avg(ann_AB, BP, nnf_patch_size[curr_layer], F_A_size[curr_layer][2:], F_B_size[curr_layer][2:])
    img_B = reconstruct_avg(ann_BA, A, nnf_patch_size[curr_layer], F_A_size[curr_layer][2:], F_B_size[curr_layer][2:])

    img_AP = np.clip(img_AP/255.0, 0, 1)[:,:,::-1]
    img_B = np.clip(img_B/255.0, 0, 1)[:,:,::-1]
    return img_AP, img_B







