from __future__ import print_function, division
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import argparse
import time
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('-A','--A_PATH',type=str,default='A.png',help='Path to image A')
parser.add_argument('-BP','--BP_PATH',type=str,default='BP.png',help='Path to image BP')
parser.add_argument('-c','--config',type=str,default='config.json',help='Path to config file')
parser.add_argument("-g", "--gpu", action="store",default='0',type=str, help="Choose gpu")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
import json
from utils import load_image
from deep_image_analogy import deep_image_analogy
 
def main():
    config = json.load(open("config.json"))
    """
    C: conv
    R: relu
    M: maxpool
    0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35...
    C R C R M C R C R M C  R  C  R  C  R  C  R  M  C  R  C  R  C  R  C  R  M  C  R  C  R  C  R  C  R....
    """
    
    print('='*20+"CONFIG"+'='*20)
    print(config)
    print('='*20+"CONFIG"+'='*20)
    
    img_A = load_image(args.A_PATH,config['resize_ratio'])
    img_BP = load_image(args.BP_PATH,config['resize_ratio'])
    
    t_begin = time.time()
    print("="*20+"Deep Image Analogy Alogrithm Start"+"="*20)
    img_AP,img_B = deep_image_analogy(A=img_A,BP=img_BP,config=config)
    elapse_time = time.time() - t_begin
    print("Deep Image Analogy Algorithm Finished, Elapsed: {:.2f}s".format(elapse_time))
    
    plt.imsave('AP-{}.png'.format(datetime.now().strftime('%Y-%m-%d-%H-%M-%S')),img_AP)
    plt.imsave('B-{}.png'.format(datetime.now().strftime('%Y-%m-%d-%H-%M-%S')),img_B)
 
if __name__ == '__main__':
    assert torch.cuda.is_available()
    main()




