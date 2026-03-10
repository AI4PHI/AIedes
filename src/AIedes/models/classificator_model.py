# src/models/fc_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class AIedesClassificator(nn.Module):
    """
    AIedesClassificator is a fully connected neural network model for classifying
    Aedes mosquito presence based on input features. It allows for dynamic layer
    configuration and can be used on different devices (CPU or GPU).
    This model is designed to be flexible, allowing users to specify the number of
    hidden layers and their sizes, making it suitable for various classification tasks.
    Inherits from torch.nn.Module to leverage PyTorch's neural network capabilities."""
    
    def __init__(self, input_size, hidden_layers=None, bias=True, device="cpu"):
        """
        MosquitoNet constructor.

        Args:
            input_size (int): Number of input features.
            hidden_layers (list of int, optional): Sizes of hidden layers. Defaults to None.
            bias (bool, optional): Whether to include bias in Linear layers. Defaults to True.
            device (torch.device or str, optional): The device on which the model will run.
                Defaults to None, in which case the model will use the default torch.device.
        """
        super(AIedesClassificator, self).__init__()
        if hidden_layers is None:
            hidden_layers = []
        self.device = device
        
        # Combine input size, hidden layers, and output size into one list
        layer_sizes = [input_size] + hidden_layers + [1]  # Output size is 1
        
        # Create a ModuleList to hold the layers
        self.layers = nn.ModuleList()
        
        # Create the layers dynamically based on the layer_sizes list and bias parameter
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=bias))
        
        # Move layers to the specified device
        self.to(self.device)
    
    def forward(self, x):
        """
        Forward pass for the MosquitoNet.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Ensure input tensor is on the same device as the model
        #x = x.to(self.device)

        # Apply ReLU activation after each layer except the last one
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        
        # For the last layer, don't apply activation (since it's the output layer)
        x = self.layers[-1](x)
        return x
    
    
    def get_model_summary(self):
        """
        Return a summary of the model's architecture.
        Can be used for logging and reporting.
        """
        return str(self)
