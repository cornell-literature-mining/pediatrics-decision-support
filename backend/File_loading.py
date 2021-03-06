import os
import torch

USE_CPU = True # Temp flag as my computer doesn't have a GPU

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

def Open(file_name, mode):
    return open(os.path.join(__location__, file_name), mode)

def torch_load(file_name):
    return torch.load(os.path.join(__location__, file_name))
