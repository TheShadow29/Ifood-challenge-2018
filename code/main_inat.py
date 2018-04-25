import torch
from pathlib import Path
from cfg import process_config
from data_loader import INat_simple
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from nn_models import BaseModel
from nn_trainer import BaseTrainer


if __name__ == "__main__":
    cfg_file = "./config.json"
    config = process_config(cfg_file)

    # define Train / Valid / Test Dataset and DataLoader
    # May need to define the Transforms as well

    # define model to be used (imported from nn_model)
    model = BaseModel(config)
    # define trainer to be used (imported from nn_trainer)
    optimizer = torch.optim.Adam(model.params(), config['lr'])
    trainer = BaseTrainer(config, model, optimizer)
    # train the model
