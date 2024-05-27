"""
Runs a model on a single node across N-gpus.

See
https://williamfalcon.github.io/pytorch-lightning/

"""
import os # , torch, warnings
import numpy as np
from utils import SegmentationModule, config
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

"""
.. warning:: `logging` package has been renamed to `loggers` since v0.7.0.
 The deprecated package name will be removed in v0.9.0.
"""

SEED = 234
# torch.manual_seed(SEED)
# np.random.seed(SEED)
pl.seed_everything(SEED, workers=True)
# print('__CUDNN VERSION:', torch.backends.cudnn.version())
# print('__Number CUDA Devices:', torch.cuda.device_count())

def main(args):
    """
    Main training routine specific for this project
    :param hparams:
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
            devices=-1, # set to -1 to use all avaliable gpus...
            reload_dataloaders_every_n_epochs=1,
            # limit_train_batches=0.6,
            default_root_dir=model.hparams.root,
            max_epochs=model.hparams.n_epochs,
            # log_gpu_memory='min_max',
            sync_batchnorm=True,
            # precision=16,
            # accumulate_grad_batches={150:2, 400:4},#2, # changing this parameter affects outputs
            callbacks=[checkpoint_callback])
            # checkpoint_callback=checkpoint_callback)# < 1.4.0
            #args.weights_path) # this is how you resume training from lightning checkpoint...

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)

if __name__ == '__main__':

    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    root_dir = os.path.dirname(os.path.realpath(__file__))
    # import args to add to basic lightning module args
    args = config.add_args(return_='args')

    main(args)
