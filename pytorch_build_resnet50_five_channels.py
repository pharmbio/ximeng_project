import matplotlib.pyplot as plt
import torch
from torchvision.datasets import ImageFolder
import torchvision.models as models



def main():
dataset = ImageFolder('/home/jovyan/mnt/external-images-pvc/ximeng/five_channel_images/')
