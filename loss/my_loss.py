from math import exp

import torch
import torch.nn.functional as F

from torch.autograd import Variable
import cv2
import numpy as np
import torch
import torch.nn as nn






def distancepunish_loss(pred, truth):
#Open the full code after the article is received

if __name__ == '__main__':
    # Generate example input tensors
    img = torch.randn(3, 2, 640, 640)  # Tensor with shape (batch_size, channels, height, width)
    target = torch.randint(low=0, high=1, size=(3, 640, 640), dtype=torch.int32)
    # Calculate the structural similarity loss for the batch
    loss = structural_similarity_loss(img, target, window_size=7)
    print(loss)
    # Calculate the average loss across the batch
