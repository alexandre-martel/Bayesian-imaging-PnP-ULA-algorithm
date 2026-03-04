import torch
from torch import Tensor
import time as time
import numpy as np
import deepinv as dinv

 

class ULAIterator():
 
    def __init__(self, algo_params: dict):
        
        if algo_params.get("denoiser") == "DnCNN":
            self.dncnn = dinv.models.DnCNN(in_channels=1,
            out_channels=1,
            pretrained="download")
        elif algo_params.get("denoiser") == "DRUNet":
            self.dncnn = dinv.models.DRUNet(in_channels=1,
            out_channels=1,
            pretrained="download")
        
        missing_params = []

        if "alpha" not in algo_params:
            missing_params.append("alpha")
        if "denoiser_param" not in algo_params:
            missing_params.append("denoiser_param")         
        if "delta" not in algo_params:
            missing_params.append("delta")
        if "physics" not in algo_params:
            missing_params.append("physics")

        if missing_params:
            raise ValueError(
                f"Missing required parameters for ULA: {', '.join(missing_params)}"
            )
            
        self.C = [0,1]
        self.denoiser_param = algo_params["denoiser_param"]
        self.alpha = algo_params["alpha"]
        self.sigma_destruction = algo_params["sigma_destruction"]
        self.delta = algo_params["delta"] # step size
        self.physics = algo_params["physics"]
    
    
    
    @staticmethod
    def get_physics(
        sigma_noise = 1/255,
        kernel_type = "box3",   # "identity" | "box3" 
        kernel_size = 3,            # utilisé si kernel_type=="box3" 
        sigma_blur = 0.6,         # utilisé si kernel_type=="gaussian"
        device = 'cpu'
    ):
        if kernel_type == "identity":
            # Noyau delta (pas de flou)
            kernel = torch.zeros((1, 1, 1, 1), device=device)
            kernel[0, 0, 0, 0] = 1.0

        elif kernel_type == "box3":
            ks = 3
            kernel = torch.ones((1, 1, ks, ks), device=device) / (ks * ks)

        # elif kernel_type == "gaussian":
        #     ks = int(kernel_size)
        #     if ks % 2 == 0:
        #         ks += 1  # on force impair
        #     kernel = _gaussian_kernel(ks, float(sigma_blur), device=device)

        else:
            raise ValueError(f"kernel_type inconnu: {kernel_type}")

        physics = dinv.physics.Blur(
            filter=kernel,
            padding="circular",
            device=device
        )
        physics.noise_model = dinv.physics.GaussianNoise(sigma=sigma_noise)
        return physics

 
    @staticmethod
    def power_iteration(physic, num_iterations: int) -> np.ndarray:
        # Ideally choose a random vector
        # To decrease the chance that our vector
        # Is orthogonal to the eigenvector
        b_k_np = np.random.rand(225, 225).astype(np.float32)
        b_k = torch.from_numpy(b_k_np).unsqueeze(0).unsqueeze(0)

        for _ in range(num_iterations):
            # calculate the matrix-by-vector product Ab
            b_k1 = physic.A_adjoint(physic.A(b_k))

            # calculate the norm
            b_k1_norm = torch.norm(b_k1)

            # re normalize the vector
            b_k = b_k1 / b_k1_norm

        spectral_norm_sq = torch.norm(physic.A_adjoint(physic.A(b_k))).item()
        return spectral_norm_sq

    def likelihood_grad(self, X, y):
        grad = -(self.sigma_destruction**2) *  self.physics.A_adjoint(self.physics.A(X) - y)
        return grad
    
    def clip(self, X):
        return torch.clamp(X, self.C[0], self.C[1])
            

    def step(self, X, y):
        # Bruit 
        Z = torch.randn_like(X)
        
        # Denoiser (DnCNN)
        D = self.dncnn(X, np.sqrt(self.denoiser_param))
        
        # Gradient de la vraisemblance
        grad_L = self.likelihood_grad(X, y)
        
        step_vraisemblance =  self.delta * grad_L
        step_prior = self.alpha * (self.delta / self.denoiser_param) * (D - X)
        bruit_langevin = np.sqrt(2 * self.delta) * Z
        
        x_next = X + step_vraisemblance + step_prior + bruit_langevin
        
        return self.clip(x_next)