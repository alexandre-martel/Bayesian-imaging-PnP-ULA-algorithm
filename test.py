import deepinv as dinv
from scipy import linalg
from src.ULA_class import ULAIterator
import matplotlib.pyplot as plt
import PIL.Image as Image
import numpy as np
import torch


physics = ULAIterator.get_physics(sigma_noise=1/255, device='cpu')
L=1
denoiser_param = (5/255)**2
Ly = 255**2 * ULAIterator.power_iteration(physics, num_iterations=10)**2
delta = 0.5/(L/denoiser_param + Ly)

algo_params_default = {
 "alpha": 1.0,
 "denoiser_param": denoiser_param,
 "sigma_destruction": 1.5,
 "delta": delta,
 "physics": physics
}

ula = ULAIterator(algo_params_default)

img = np.array(Image.open('data/camera_man.jpg').convert('L')).astype(np.float32)
img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
img_blurred = physics(img)

# img_for_plot = img_blurred.squeeze().cpu().numpy()
# plt.imsave('camera_man_blurred.png', img_for_plot, cmap='gray')

img_temp = img_blurred.clone()
for i in range(200):
 print(f"Iteration {i+1}")
 img_temp = ula.step(img_temp, img_blurred)


img_for_plot = img_temp.squeeze().cpu().numpy()
plt.imsave('camera_man_unblurred.png', img_for_plot, cmap='gray')



 