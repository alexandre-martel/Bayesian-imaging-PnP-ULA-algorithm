import torch
from torch import Tensor
import time as time
import numpy as np

A = ULAIterator.gaussian_kernel(size=9, sigma=1.5)

algo_params_default = {
    "step_size": 0.1,
    "alpha": 0.1,
    "sigma": 0.1,
    "destruction_operator": 
    
}

class ULAIterator():
 
    def __init__(self, algo_params: dict[str, float]):
        
        missing_params = []
        if "step_size" not in algo_params:
            missing_params.append("step_size")
        if "alpha" not in algo_params:
            missing_params.append("alpha")

        if "sigma" not in algo_params:
            missing_params.append("sigma")
            
        if "destruction_operator" not in algo_params:
            missing_params.append("destruction_operator")

        if missing_params:
            raise ValueError(
                f"Missing required parameters for ULA: {', '.join(missing_params)}"
            )
            
        
        
        # self.C = [-1, 2]^d
        self.step_size = algo_params["step_size"]
        self.alpha = algo_params["alpha"]
        self.sigma = algo_params["sigma"]
        self.destruction_operator = algo_params["destruction_operator"]

        
    def likelihood_grad(self, X, y):
        A_X = self.destruction_operator.dot(X)
        grad = -(A_X - y) / (2*self.sigma ** 2)
        grad = -2*
        return grad

    @staticmethod
    def gaussian_kernel(size=9, sigma=1.5):

        ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
        gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
        

        kernel = np.outer(gauss, gauss)

        return kernel / kernel.sum()
        
    def step(self, X, y):
        
        # Noise 
        Z = torch.randn_like(X)
        
        # Likelihood gradient
        grad_likelihood = self.likelihood_grad(X, y)
        
        
        
            

