"""
EEG Dataloaders — CHB-MIT and Siena Scalp EEG

Usage:
    from dataloaders import get_dataloaders, generate

    # Generate CSVs + tensors (run once)
    generate("chbmit", subjects=["chb01", "chb02", "chb03", "chb04", "chb05"])

    # Get PyTorch DataLoaders (instant if tensors exist)
    train_dl, val_dl, test_dl = get_dataloaders("chbmit")
"""
from dataloaders.common.loader import get_dataloaders, generate