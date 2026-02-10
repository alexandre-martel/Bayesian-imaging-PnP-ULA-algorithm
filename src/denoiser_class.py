import numpy as np
import torch
import torch.nn as nn

class Denoiser():
 def __init__(self, denoiser):
  self.denoiser = denoiser

 def prox(self, x, denoiser_arg):
  return self.denoiser(x, denoiser_arg)