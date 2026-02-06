import torch
from torch import Tensor
import time as time
import numpy as np
from prox_class import PnP
from deep

A = ULAIterator.gaussian_kernel(size=9, sigma=1.5)

algo_params_default = {
    "delta": 0.1,
    "alpha": 0.1,
    "sigma": 0.1,
    "epsilon": 0.1,
    "destruction_operator": A
}

class ULAIterator():
 
    def __init__(self, algo_params: dict[str, float]):
        
        missing_params = []
        if "step_size" not in algo_params:
            missing_params.append("step_size")
        if "alpha" not in algo_params:
            missing_params.append("alpha")
        if "epsilon" not in algo_params:
            missing_params.append("epsilon")
        if "sigma" not in algo_params:
            missing_params.append("sigma")        
        if "delta" not in algo_params:
            missing_params.append("delta")
        if "destruction_operator" not in algo_params:
            missing_params.append("destruction_operator")

        if missing_params:
            raise ValueError(
                f"Missing required parameters for ULA: {', '.join(missing_params)}"
            )
            
        
        
        self.C = [-1, 2]
        self.epsilon = algo_params["epsilon"]
        self.alpha = algo_params["alpha"]
        self.sigma = algo_params["sigma"]
        self.delta = algo_params["delta"] # step size
        self.destruction_operator = algo_params["destruction_operator"]

        
    def likelihood_grad(self, X, y):
        grad = (1/self.sigma**2) *  self.destruction_operator.T @ (self.destruction_operator @ X - y)
        return grad

    @staticmethod
    def gaussian_kernel(size=9, sigma=1.5):

        ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
        gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
        kernel = np.outer(gauss, gauss)

        return kernel / kernel.sum()
    
    def clip(self, X):
        pass
        
    def step(self, X, y):
        
        # Noise 
        Z = torch.randn_like(X)
        
        # Denoiser 
        D = PnP()
        
        # Likelihood gradient
        grad_likelihood = self.likelihood_grad(X, y)
        
        
        
        x_t1 = self.delta * grad_likelihood + self.alpha * self.delta/self.epsilon (D(X) - X) + np.sqrt(2* self.delta) * Z 
        
        
        
            

