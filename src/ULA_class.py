import torch
from torch import Tensor
import time as time
import numpy as np
import deepinv as dinv
from deepinv.physics.forward import adjoint_function

 

class ULAIterator():
 
    def __init__(self, algo_params: dict):
        
        self.dncnn = dinv.models.DnCNN(in_channels=1)
        
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
            
        self.C = [-1, 2]
        self.denoiser_param = algo_params["denoiser_param"]
        self.alpha = algo_params["alpha"]
        self.sigma_destruction = algo_params["sigma_destruction"]
        self.delta = algo_params["delta"] # step size
        self.physics = algo_params["physics"]
    
    
    @staticmethod
    def get_physics(sigma_noise=1/255, device='cpu'):
        kernel = torch.ones((1, 1, 9, 9)) / 81.0
        physics = dinv.physics.Blur(filter=kernel, device=device)
        physics.noise_model = dinv.physics.GaussianNoise(sigma=sigma_noise)
        return physics
 
    @staticmethod
    def power_iteration(physic, num_iterations: int) -> np.ndarray:
        # Ideally choose a random vector
        # To decrease the chance that our vector
        # Is orthogonal to the eigenvector
        b_k_np = np.random.rand(256, 256).astype(np.float32)
        b_k = torch.from_numpy(b_k_np).unsqueeze(0).unsqueeze(0)

        for _ in range(num_iterations):
            # calculate the matrix-by-vector product Ab
            b_k1 = physic.A_adjoint(physic.A(b_k))

            # calculate the norm
            b_k1_norm = torch.norm(b_k1)

            # re normalize the vector
            b_k = b_k1 / b_k1_norm

        return b_k

    def likelihood_grad(self, X, y):
        grad = (1/self.sigma_destruction**2) *  self.physics.A_adjoint(self.physics.A(X) - y)
        return grad
    
    def clip(self, X):
        return torch.clamp(X, self.C[0], self.C[1])
        
    def step(self, X, y):
        
        # Noise 
        Z = torch.randn_like(X)
        
        # Denoiser 
        D = self.dncnn(X, self.denoiser_param)
        
        # Likelihood gradient
        grad_likelihood = self.likelihood_grad(X, y)
        
        x_t1 = self.delta * grad_likelihood + self.alpha * self.delta/self.epsilon * (D - X) + np.sqrt(2* self.delta) * Z 
        
        final = self.clip(X + x_t1)
        
        return final
            

        # metrics PSRN et SIM