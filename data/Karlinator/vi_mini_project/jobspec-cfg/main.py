import monai
import torch
import argparse

from visualize import visualize

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

def train(epochs, data):
    model = monai.networks.nets.SegResNet(
		blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=4,
        out_channels=3,
        dropout_prob=0.2
    ).to(device=device) # We're only segmenting "tumour/not tumour" (I think).

    loss = monai.losses.DiceLoss(
        smooth_nr=0,
        smooth_dr=1e-5,
        squared_pred=True,
        to_onehot_y=False,
        sigmoid=True,
    )

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=1e-4,
        weight_decay=1e-5,
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=epochs,
    )


    preprocessing_transforms = [
        monai.transforms.LoadImaged(keys=["image", "label"], image_only=False, ensure_channel_first=True),
        monai.transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        monai.transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]

    random_transforms = [
        monai.transforms.RandSpatialCropd(keys=["image", "label"], roi_size=[224, 224, 144], random_size=False),
        monai.transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        monai.transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        monai.transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        monai.transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        monai.transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ]

    preprocessing = monai.transforms.Compose(preprocessing_transforms + random_transforms)


    
    datalist = monai.data.load_decathlon_datalist(data or 'data/Task01_BrainTumour/dataset.json', data_list_key='training')

    dataset = monai.data.Dataset(
        data=datalist,
        transform=preprocessing,
    )

    dataloader = monai.data.DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=16)

    postprocessing = monai.transforms.Compose([
		monai.transforms.Activationsd(keys="pred", sigmoid=True),
        monai.transforms.AsDiscreted(keys="pred", threshold=0.5),
    ])

    inferer = monai.inferers.SimpleInferer()

    key_metric = {
        "train_mean_dice": monai.handlers.MeanDice(
            include_background=True,
            output_transform=monai.handlers.from_engine(["pred", "label"]))
    }

    handlers = [
        monai.handlers.LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=True),
        # monai.handlers.ValidationHandler(validator=evaluator, epoch_level=True, interval=1),
        monai.handlers.StatsHandler(tag_name="train_loss", output_transform=monai.handlers.from_engine(["loss"], first=True)),
        monai.handlers.CheckpointSaver(
            'checkpoints/',
            {'network': model, 'optimizer': optimizer, 'lr_scheduler': lr_scheduler},
            save_final=True,
            save_interval=1
        ),
    ]
        

    trainer = monai.engines.SupervisedTrainer(
        max_epochs=epochs,
        device=device,
        train_data_loader=dataloader,
        network=model,
        loss_function=loss,
        optimizer=optimizer,
        inferer=inferer,
        postprocessing=postprocessing,
        key_train_metric=key_metric,
        train_handlers=handlers,
        amp=True,
    )

    trainer.run()

    torch.save(model, 'models/model.pt')

def infer(data):
    model = monai.networks.nets.SegResNet(
		blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=4,
        out_channels=3,
        dropout_prob=0.2
    ).to(device=device)

    preprocessing_transforms = [
        monai.transforms.LoadImaged(keys="image", image_only=False, ensure_channel_first=True),
        monai.transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]

    preprocessing = monai.transforms.Compose(preprocessing_transforms)


    
    datalist = monai.data.load_decathlon_datalist(data or 'data/Task01_BrainTumour/dataset.json', data_list_key='test')

    dataset = monai.data.Dataset(
        data=datalist,
        transform=preprocessing,
    )

    dataloader = monai.data.DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=16)

    postprocessing = monai.transforms.Compose([
		monai.transforms.Activationsd(keys="pred", sigmoid=True),
        monai.transforms.Invertd(keys="pred", transform=preprocessing, orig_keys="image", meta_keys="pred_meta_dict", nearest_interp=False, to_tensor=True),
        monai.transforms.AsDiscreted(keys="pred", threshold=0.5),
        monai.transforms.Lambdad(keys="pred", func=lambda x: torch.where(x[[0]] > 0, 4, torch.where(x[[2]] > 0, 1, torch.where(x[[1]] > 0, 2, 0)))),
        monai.transforms.SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir="output/", output_postfix="seg", output_dtype="uint8", resample=False, squeeze_end_dims=True),
    ])

    inferer = monai.inferers.SlidingWindowInferer(roi_size=[240, 240, 160], sw_batch_size=1, overlap=0.5)

    handlers = [
        monai.handlers.CheckpointLoader(
            load_path='checkpoints/checkpoint_epoch=100.pt',
            load_dict={'network': model}
        ),
    ]

    evaluator = monai.engines.SupervisedEvaluator(
        device=device,
        val_data_loader=dataloader,
        network=model,
        inferer=inferer,
        postprocessing=postprocessing,
        val_handlers=handlers,
        amp=True,
    )
    
    handlers[0](evaluator)

    evaluator.run()

def main():

    parser = argparse.ArgumentParser(
            prog="Monai BRATS MRI Segmentation",
            description="Trains a thing",
    )

    parser.add_argument('mode', choices=['train', 'infer', 'visualize'])
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-d', '--data', help='Path to the dataset.json file in Task01_BrainTumour')

    args = parser.parse_args()

    if args.mode == 'train':
        train(args.epochs, args.data)
    elif args.mode == 'infer':
        infer(args.data)
    elif args.mode == 'visualize':
        visualize()

if __name__ == '__main__':
    main()

