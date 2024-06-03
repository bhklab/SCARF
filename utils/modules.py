import os, torch, warnings, json
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import lightning as pl
import numpy as np
import einops as E
from .metrics import SoftDiceLoss
from .loss import *
from .optimizers import *
from .models import *
from .transform import *

def getJson(path):
    with open(path, 'r') as myfile:
        data=myfile.read()
    obj = json.loads(data)
    return obj

def read_image(path):
    img = sitk.ReadImage(path)
    img = sitk.GetArrayFromImage(img)

    if img.shape[0] == img.shape[1]:
        img = E.rearrange(img, 'H W D -> D H W')

    img = img.astype(np.float32)
    # img = torch.from_numpy(img)

    return img

class Dataset(Dataset):
    def __init__(self, paths, data_path, transform=None):
        super().__init__()
        self.paths = paths
        self.transform = transform
        self.data_path = data_path

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image_path = self.paths[index]
        image = read_image(os.path.join(self.data_path, image_path['image']))
        label = read_image(os.path.join(self.data_path, image_path['label']))    

        if self.transform is not None:
            image, label = self.transform(image, label)

        return image, label 

class SegmentationModule(pl.LightningModule):
    def __init__(self, hparams, update_lr=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        # reload pre-trained model...
        self.config = getJson(self.hparams.config_path)

        if update_lr is not None:
            self.hparams.lr = update_lr

    def setup(self, stage=None):
        train_data = self.config["train"]
        self.train_data = []
        for _ in range(5): # 10x elements in training fold
            self.train_data += train_data
        self.valid_data = self.config['val']

        self.get_model()
        self.get_data_info()
        self.get_loss()


    def forward(self, x):
        x = x.unsqueeze(1)
        return self.model(x)   

    def common_step(self, batch, batch_idx):

        inputs, targets = batch

        if inputs.shape != targets.shape:
            warnings.warn("Input Shape Not Same size as label...")
        if batch_idx == 0:
            print(inputs.min(), inputs.max(), inputs.size())
            print(targets.min(), targets.max(), targets.size())

        outputs = self(inputs)
        print(outputs.size())
        test = torch.softmax(outputs, dim=1)
        test = torch.argmax(test , dim=1) 
        print(test.min(), test.max(), test.size())

        # targets = targets.unsqueeze(1)
        loss = self.criterion(outputs, targets)

        nan_val = 10
        loss = torch.nan_to_num(loss, nan=nan_val, posinf=nan_val)

        print(f'LOSS: {loss}')

        return loss, outputs

    def training_step(self, batch, batch_idx):
        warnings.warn("AT TRAIN START")

        loss, _ = self.common_step(batch, batch_idx)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        warnings.warn("AT VALIDATION START")

        loss, outputs = self.common_step(batch, batch_idx)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        _, targets = batch

        outputs = torch.softmax(outputs, dim=1)
        outputs = torch.argmax(outputs , dim=1) 
        max_ = self.hparams.n_classes+1
        print(f'OUTPUT:{outputs.max()}\n TARGET:{targets.max()}\n {outputs.size()}')

        outputs = outputs.float().cpu().detach().numpy()
        targets = targets.float().cpu().detach().numpy()

        avg_dice = 0

        for batch_idx in range(len(outputs)):
            for idx in range(max_):
                pred = np.copy(outputs[batch_idx])
                gt = np.copy(targets[batch_idx])
                pred[pred!=idx] = 0
                gt[gt!=idx] = 0
                dice = (np.sum(pred[gt==idx])*2.0) / (np.sum(pred) + np.sum(gt))
                print(f'dice_{idx}: {dice}')
                dice = 0 if np.isnan(dice) else dice
                avg_dice+=dice
        avg_dice /= len(outputs)*max_
        self.log("dice", avg_dice, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def configure_optimizers(self):
        init_optimizer = RAdam(self.model.parameters(), lr=self.hparams.lr,
                                   weight_decay=self.hparams.decay)

        if self.hparams.scheduler is True:
            if self.hparams.scheduler_type == 'plateau':
                scheduler = ReduceLROnPlateau(init_optimizer, factor=self.hparams.gamma,
                                              patience=self.hparams.decay_after,
                                              threshold=0.0001)
            else:
                scheduler = StepLR( init_optimizer, step_size=self.hparams.decay_after,
                                    gamma=self.hparams.gamma,)

            return [init_optimizer], [scheduler]
        else:
            return [init_optimizer]
        
    # -------------------
    #  GET STUFF
    # -------------------
            
    def get_weights(self):

        base = np.array([1e-40])
        values = [int(v) for v in self.counts]
        weights = np.array(values)/(np.sum(values)+1e-4)
        weights = np.append(base, weights)
        weights = np.abs(np.log(weights))
        weights[0] = 0.1
        weights = (weights-np.min(weights))/(np.max(weights)-np.min(weights))
        weights[0] = 0.0001
        weights[np.argmax(weights)] = 0.9999
        self.config["weights"] = weights
        
        print(weights, "\n(Weights used to mitigate class imbalance...)\n") 

        return torch.from_numpy(weights.astype(np.float32))

    def get_loss(self):
        self.class_weights = self.get_weights()

        self.criterion = None

        if self.hparams.loss == "FOCAL":
            loss = FocalLoss(weight=self.class_weights)
            self.criterion = loss

        elif self.hparams.loss == "CATEGORICAL":
            loss = CrossEntropyLoss(weight=self.class_weights)
            self.criterion = loss

        elif self.hparams.loss == "WDCTOPK":
            ce_kwargs = {'weight':self.class_weights}
            soft_dice_kwargs = {'batch_dice':False, 'do_bg':True, 'smooth':1., 'square':False, 'weight':self.class_weights}
            # loss = WeightedCrossEntropyLoss(weight=self.class_weights)
            loss = DC_and_topk_loss(soft_dice_kwargs, ce_kwargs)
            self.criterion = loss

        elif self.hparams.loss == "WFTTOPK":
            ce_kwargs = {'weight':self.class_weights}
            tversky_kwargs = {'batch_dice':False, 'do_bg':True, 'smooth':1., 'square':False}
            # can add weight class if necessary ...
            loss = FocalTversky_and_topk_loss(tversky_kwargs, ce_kwargs)
            self.criterion = loss

        else:
            warnings.warn("Using Standard DICE loss. One Hot encoded target required.")
            loss = SoftDiceLoss(weight=self.class_weights)
            self.criterion = loss


    def get_model(self):

        """
        Layout model
        """

        if self.hparams.model == "VNET":
            self.model = VNet3D(num_classes=self.hparams.n_classes)

        elif self.hparams.model == "WOLNET":
            self.model = WolnyUNet3D(num_classes=self.hparams.n_classes, f_maps=self.hparams.f_maps)

        elif self.hparams.model == "RESUNET":
            self.model = ResUNet3D( num_classes=self.hparams.n_classes, f_maps=self.hparams.f_maps)

        elif self.hparams.model == "TIRAMISU":
            # 3D version of tiramisu_network...
            self.model = FCDenseNet(
                in_channels=1,
                down_blocks=(2, 2, 2, 2, 3),
                up_blocks=(3, 2, 2, 2, 2),
                bottleneck_layers=2,
                growth_rate=12,
                out_chans_first_conv=16,
                n_classes=self.hparams.n_classes,
            )

        elif self.hparams.model == "ANATOMY":
            # AnatomyNet discussed in https://github.com/wentaozhu/AnatomyNet-for-anatomical-segmentation
            self.model = AnatomyNet3D(num_classes=self.hparams.n_classes)

        elif self.hparams.model == "PIPOFAN":
            self.model = PIPOFAN3D(num_classes=self.hparams.n_classes, factor=3)

        elif self.hparams.model == "HIGHRESNET":
            self.model = HighResNet3D(classes=self.hparams.n_classes)

        elif self.hparams.model == "UNET++":
            self.model = NestedUNet( num_classes=self.hparams.n_classes, factor=4, deep_supervision=True)

        elif self.hparams.model == "UNET3+":
            self.model = UNet_3Plus(n_classes=self.hparams.n_classes, factor=2)

        elif self.hparams.model == "RSANET":
            self.model = RSANet(n_classes=self.hparams.n_classes)

        elif self.hparams.model == "DENSEVOX":
            self.model = DenseVoxelNet(in_channels=1, num_classes=self.hparams.n_classes)

    def get_data_info(self):

        dataset = Dataset(self.train_data, self.hparams.data_path, None)

        self.mean = 0.
        self.std = 0.

        self.counts = [0.] * (self.hparams.n_classes-1)

        for i in range(len(dataset)):
            image, label = dataset[i]

            self.mean += np.mean(image)
            self.std += np.std(image)

            for label_idx in range(1, self.hparams.n_classes):
                self.counts[label_idx-1] += np.sum(label == label_idx)

        self.mean /= len(dataset)
        self.std = len(dataset)

        print(f'Dataset has mean {self.mean} and std {self.std}')

        self.counts = [c/len(dataset) for c in self.counts]

    # -------------------
    #  DATA LOADERS
    # -------------------

    def get_dataloader( self, df, mode="valid", transform=None, resample=None,
                        shuffle=False, transform2=None, batch_size=None):

        dataset = Dataset(df, self.hparams.data_path, transform)

        batch_size = self.hparams.batch_size if batch_size is None else batch_size
        # best practices to turn shuffling off during validation...
        validate = ['test']
        shuffle = False if mode=='valid' else True

        return DataLoader( dataset=dataset, num_workers=self.hparams.workers,
                           batch_size=batch_size, pin_memory=True, shuffle=shuffle,
                           drop_last=True,)

    def train_dataloader(self):

        transform = Compose(
            [
                HistogramClipping(
                    min_hu=self.hparams.clip_min, max_hu=self.hparams.clip_max
                ),
                RandomFlip3D(), # left and right should be distinguished...
                RandomRotation3D(p=self.hparams.aug_prob/1.5),
                ElasticTransform3D(p=self.hparams.aug_prob/1.5),
                RandomZoom3D(p=self.hparams.aug_prob/1.5),
                RandomCrop3D(window=self.hparams.window,
                             mode="train",
                             factor=self.hparams.crop_factor,
                             crop_as=self.hparams.crop_as,
                            ),
                NormBabe(mean=self.mean, std=self.std, type=self.hparams.norm),
            ]
        )

        # add transform to dataloader
        return self.get_dataloader(df=self.train_data, mode="train", transform=transform,
                                   resample=False, batch_size=self.hparams.batch_size,)

    def val_dataloader(self):
        # imported from transform.py
        transform = Compose(
            [
                HistogramClipping(
                    min_hu=self.hparams.clip_min, max_hu=self.hparams.clip_max
                ),
                RandomCrop3D(
                    window=self.hparams.window,
                    mode="valid",
                    factor=self.hparams.crop_factor,
                    crop_as=self.hparams.crop_as,
                ),
                NormBabe(mean=self.mean, std=self.std, type=self.hparams.norm),
            ]
        )

        return self.get_dataloader( df=self.valid_data, mode="valid",
                                    transform=transform,  # should be default
                                    resample=False, batch_size=self.hparams.batch_size,)
    


    
    