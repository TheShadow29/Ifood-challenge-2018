import torch


class BaseModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    # Define common dependencies in the model here
    # Use this as a parent class to define new Models
