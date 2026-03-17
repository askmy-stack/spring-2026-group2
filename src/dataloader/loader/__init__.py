"""
loader — Dataset loaders, pipeline orchestration, tensor generation.

Usage:
    from src.loader.loader import generate, get_dataloaders
    from src.loader.chbmit_loader import CHBMITLoader
    from src.loader.siena_loader import SienaLoader
    from src.loader.base import BaseDatasetLoader

Note: imports are lazy to avoid requiring torch at package load time.
"""


def generate(*args, **kwargs):
    from dataloader.loader.loader import generate as _generate
    return _generate(*args, **kwargs)


def get_dataloaders(*args, **kwargs):
    from dataloader.loader.loader import get_dataloaders as _get_dataloaders
    return _get_dataloaders(*args, **kwargs)