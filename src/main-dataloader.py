#%%
from dataloader.chbmit import CHBMITBIDSLoader

loader = CHBMITBIDSLoader(bids_root="data/bids_chbmit")
loader.run(subjects=["chb01", "chb02"], val_size=0.15, test_size=0.15, seed=42)

train_df = loader.splits["train"]