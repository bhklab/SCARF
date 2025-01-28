
import os 
from utils import SegmentationModule, config
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

SEED = 234
pl.seed_everything(SEED, workers=True)

def train_segmentation_model(args):
    """
    Trains the segmentation model with the specified arguments.

    Args:
        args (Namespace): Parsed command-line arguments containing 
                          training configurations and hyperparameters.
    """
    model = SegmentationModule(args, update_lr=0.001) 
    checkpoint_callback = ModelCheckpoint( monitor="val_loss",
                                           filename=str(args.model + '-epoch{epoch:02d}-val_loss{val/loss:.2f}'),
                                           auto_insert_metric_name=False,
                                           mode="min",
                                           save_last=True,
                                           save_top_k=3,)
    trainer = pl.Trainer(
            accelerator="gpu",
            devices=-1, 
            reload_dataloaders_every_n_epochs=1,
            default_root_dir=model.hparams.root,
            max_epochs=model.hparams.n_epochs,
            sync_batchnorm=True,
            callbacks=[checkpoint_callback])

    trainer.fit(model)

if __name__ == '__main__':

    root_dir = os.path.dirname(os.path.realpath(__file__))
    args = config.add_args(return_='args')

    train_segmentation_model(args)
