import deepinv as dinv
from src.ULA_class import ULAIterator
import matplotlib.pyplot as plt
import PIL.Image as Image
import numpy as np
import torch
import deepinv as dinv
# PARAMS
denoiser_param = 25/255  # For DRUNET, non calibré
# denoiser_param = 4/255**2 # For DnCNN
sigma_destruction = 1/255
physics = ULAIterator.get_physics(sigma_noise=sigma_destruction, device='cpu')

L=1
Ly = ULAIterator.power_iteration(physics, num_iterations=100)/sigma_destruction**2
delta = 0.5/(L/denoiser_param + Ly)

# ULA PARAMS
algo_params_default = {
 "alpha": 1.0,
 "denoiser_param": denoiser_param,
 "sigma_destruction": sigma_destruction,
 "delta": delta,
 "physics": physics,
 "denoiser": "DRUNet"
}

# CREATE ULA INSTANCE
ula = ULAIterator(algo_params_default)

# LOAD IMAGE
img = np.array(Image.open('data/camera_man.jpg').convert('L')).astype(np.float32) / 255.0
img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
img_blurred = physics(img)
 
img_for_plot2 = img_blurred.squeeze().cpu().numpy()
plt.imsave('camera_man_blurred.png', img_for_plot2, cmap='gray')

# INSTANCIATE METRICS 
psnr_metric = dinv.metric.PSNR()
ssim_metric = dinv.metric.SSIM()

# RUN ULA 
burn_in = 50 #(FOR DRUNET)
n_iter = 300 #(FOR DRUNET)
# burn_in = 500 #(FOR DNCNN)
# n_iter = 4000 #(FOR DNCNN)

img_temp = img_blurred.clone()

mean_img = torch.zeros_like(img_temp)
count = 0
psrn_values = []
ssim_values = []

# We can save intermediate results to see the convergence of the algorithm
for i in range(n_iter):

    if i % 10 == 0:
        print(f"Iteration {i}")
        plt.imsave(f"debug/debug_{i}.png", img_temp.squeeze().cpu().numpy(), cmap="gray")        

    img_temp = ula.step(img_temp, img_blurred)
    if i >= burn_in:
        mean_img += img_temp
        count += 1

    psnr = psnr_metric(img_temp, img).item()
    ssim = ssim_metric(img_temp, img).item()
    # print(f"PSNR: {psnr}, SSIM: {ssim}")
    psrn_values.append(psnr)
    ssim_values.append(ssim)

mean_img /= count

img_for_plot = mean_img.squeeze().cpu().numpy()
plt.imsave('camera_man_unblurred_drunet.png', img_for_plot, cmap='gray')

# PLOT & SAVE METRICS
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(psrn_values)
plt.title('PSNR over Iterations')
plt.xlabel('Iteration')
plt.ylabel('PSNR (dB)')
plt.subplot(1, 2, 2)
plt.plot(ssim_values)
plt.title('SSIM over Iterations')
plt.xlabel('Iteration')
plt.ylabel('SSIM')
plt.tight_layout()
plt.savefig('metrics.png')
plt.show()



 