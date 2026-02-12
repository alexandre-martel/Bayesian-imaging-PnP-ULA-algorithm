import deepinv as dinv
from src.ULA_class import ULAIterator
import matplotlib.pyplot as plt
import PIL.Image as Image
import numpy as np
import torch


denoiser_param = 1/255**2
sigma_destruction = 3/255
physics = ULAIterator.get_physics(sigma_noise=sigma_destruction, device='cpu')
L=1
Ly = ULAIterator.power_iteration(physics, num_iterations=100)/sigma_destruction**2
delta = 0.1/(L/denoiser_param + Ly)


algo_params_default = {
 "alpha": 1.0,
 "denoiser_param": denoiser_param,
 "sigma_destruction": sigma_destruction,
 "delta": delta,
 "physics": physics
}

ula = ULAIterator(algo_params_default)

img = np.array(Image.open('data/camera_man.jpg').convert('L')).astype(np.float32) / 255.0
img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
img_blurred = physics(img)

img_for_plot2 = img_blurred.squeeze().cpu().numpy()
plt.imsave('camera_man_blurred.png', img_for_plot2, cmap='gray')

burn_in = 300
n_iter = 2000

img_temp = img_blurred.clone()

mean_img = torch.zeros_like(img_temp)
count = 0

for i in range(n_iter):

    if i % 100 == 0:
        print(f"Iteration {i}")
        
    if i % 100 == 0:
        plt.imsave(f"debug/debug_{i}.png", img_temp.squeeze().cpu().numpy(), cmap="gray")

    img_temp = ula.step(img_temp, img_blurred)
    if i >= burn_in:
        mean_img += img_temp
        count += 1

mean_img /= count

img_for_plot = mean_img.squeeze().cpu().numpy()
plt.imsave('camera_man_unblurred.png', img_for_plot, cmap='gray')



 