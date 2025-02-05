import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, env_size):
        super(DQN, self).__init__()
        
        # Define layer sizes based on environment size
        layer_sizes = self._get_layer_sizes(input_dim, output_dim, env_size)
        
        # Create layers dynamically
        self.layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i+1]) 
            for i in range(len(layer_sizes)-2)
        ])
        self.final_layer = nn.Linear(layer_sizes[-2], layer_sizes[-1])
    
    def _get_layer_sizes(self, input_dim, output_dim, env_size):
        # Define layer sizes based on environment size
        sizes = {
            4: [input_dim, 256, 512, 512, 256, output_dim],
            5: [input_dim, 512, 1024, 512, 256, output_dim],
            6: [input_dim, 512, 1024, 2048, 1024, 512, 256, output_dim],
            7: [input_dim, 512, 1024, 2048, 1024, 512, 256, output_dim],
            8: [input_dim, 512, 1024, 1024, 1024, 1024, 512, output_dim]
        }
        return sizes.get(env_size, sizes[4])  # Default to 4x4 if size not found
    
    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.final_layer(x)
        