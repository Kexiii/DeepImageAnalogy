# DeepImageAnalogy
Pytorch implementation of [Visual Attribute Transfer through Deep Image Analogy](https://arxiv.org/pdf/1705.01088.pdf)

# Some Results

![results](https://github.com/Kexiii/DeepImageAnalogy/blob/master/results.png)

# Usage

```
python main.py -g 0 --A_PATH XX --BP_PATH XXX
```

# Requirements
* Pytorch 0.3+
* GPU

# Performance
Nearly 30 minutes for one image pair in my test environment. The speed is mainly based on the number of maximum LBFGS iterations. The performance can be improved if we can use a GPU version PatchMatch algorithm 


# Acknowledgements
* This repo is mainly based on [harveyslash/Deep-Image-Analogy-PyTorch](https://github.com/harveyslash/Deep-Image-Analogy-PyTorch) and [Ben-Louis/Deep-Image-Analogy-PyTorch](https://github.com/Ben-Louis/Deep-Image-Analogy-PyTorch). Thanks for their great work.
* The test images come from official repo [msracver/Deep-Image-Analogy](https://github.com/msracver/Deep-Image-Analogy)
