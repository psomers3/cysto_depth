import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
import argparse
from data.depth_datamodule import EndoDepthDataModule
from models.depth_model import DepthEstimationModel
from models.gan_model import GAN
import glob
import os
import shutil


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=["gan", "synth"])
    parser.add_argument("stage", type=str, choices=["train", "test"])
    parser.add_argument('--gen-gan-data', dest='gen_gan_data', action='store_true')
    parser.add_argument('--adapt-synth', dest='adapt_synth', action='store_true')
    parser.add_argument('--enable-res-transfer', dest="res_transfer", action='store_true')
    parser.add_argument('--plot-graph', dest="plot_graph", action='store_true')
    parser.add_argument('--disable-plot-captions', dest="plot_captions", action='store_false')
    parser.add_argument('--lr-d', dest="lr_d", default=5e-5, type=float)
    parser.add_argument('--lr-g', dest="lr_g", default=5e-6, type=float)
    parser.add_argument('--img-disc-factor', dest="img_disc_factor", type=float, default=0)
    parser.add_argument('--p', dest="p", type=float, default=0)
    parser.add_argument('--pickup_ckpt', type=str)
    parser.add_argument('--adaptive_gate', dest='adaptive_gate', action='store_true')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--imageGAN', action='store_true', help='disable patches for discriminator')
    parser.set_defaults(gen_gan_data=False, adapt_synth=False, plot_graph=False, adaptive_gate=False,
                        imageGAN=False)
    return parser.parse_args()


def save_python_files(target_dir):
    source_dir = "."
    source_path = source_dir+"/**/*.py"
    py_files = glob.glob(source_path, recursive=True)
    if len(py_files) > 0:
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        os.makedirs(target_dir)
        for file in py_files:
            filename = os.path.basename(file)
            shutil.copyfile(file, target_dir+"/"+filename)
    else: 
        raise Exception("No files to save for training snapshot")
    
  
if __name__ == "__main__":
    args = parse_args()
    annotations_path = "../annotations"
    mode = args.mode
    generate_gan_data = args.gen_gan_data
    gan_data_dir = "../datasets/gan_data"
    dataset_dir = "../datasets"
    log_plot = False  # plot depth maps in log representation or not

    # the following values are redefined lower on. This should be done cleaner.
    batch_size = 0
    logging_dir = ''
    accumulate_grad_batches = 0

    trainer_dict = {
        'limit_val_batches': 200,
        'accelerator': "gpu",
        'devices': 1,  # so that the dataset validation is only checked from one device
        'gpus': [args.gpu] if args.gpu != -1 else -1,
        'strategy': 'ddp',  # use this strategy even for 1 node because matplotlib is used during training
        'resume_from_checkpoint': args.pickup_ckpt
    }
    
    if mode == "synth":  
        logging_dir = "../lightning_logs/depth"
        trainer_dict['max_epochs'] = 500
        batch_size = 32
        accumulate_grad_batches = 4
        if args.stage == "train":
            trainer_dict['val_check_interval'] = 1
            monitor = "val_rmse"
            del trainer_dict["limit_val_batches"]
            reduce_lr_patience = 5
            early_stop_patience = 15
            trainer_dict['max_epochs'] = 50
            checkpoint_callback = ModelCheckpoint(monitor=monitor, save_top_k=5)
            early_stop_callback = EarlyStopping(monitor=monitor, patience=early_stop_patience)
            trainer_dict['callbacks'] = [checkpoint_callback, early_stop_callback]     
    elif mode == "gan":
        warmup_steps = 0
        logging_dir = f"../lightning_logs/gan/{'gated' if args.res_transfer else 'vanilla'}" \
                      f"{'_adaptive' if args.adaptive_gate else ''}"
        batch_size = 25
        accumulate_grad_batches = 4
        
        if args.stage == "train":
            trainer_dict['max_steps'] = 200000
            trainer_dict['val_check_interval'] = 100
            unadapted_model = "../lightning_logs/depth/cyst/1/lightning_logs/version_17/checkpoints/" \
                              "epoch=22-step=52923.ckpt"
            checkpoint_callback = ModelCheckpoint(every_n_epochs=2)
            trainer_dict['callbacks'] = [checkpoint_callback]
            batch_size = 40
            accumulate_grad_batches = 3
        elif args.stage == "test":
            batch_size = 1
    ckpt = None

    dm = DepthDataModule(batch_size, annotations_path, mode,
                         dataset_dir=dataset_dir,
                         generate_data=generate_gan_data)
    logging_dir = logging_dir + "/cyst/1"

    logger = pl_loggers.TensorBoardLogger(logging_dir)
    logger.experiment.add_scalar("e_batch_size", accumulate_grad_batches*batch_size)
    # save_python_files(target_dir = "../code-snapshots-new/"+ mode + "-" + str(logger.version))
    trainer = pl.Trainer(logger=logger, accumulate_grad_batches=accumulate_grad_batches, **trainer_dict)
    
    if mode == "synth":
        if args.stage == "train":
            model = DepthEstimationModel(ckpt,
                                         lr_scheduler_patience=reduce_lr_patience,
                                         lr_scheduler_monitor=monitor,
                                         accumulate_grad_batches=accumulate_grad_batches,
                                         adaptive_gating=args.adaptive_gate)
            if args.plot_graph is True:
                logger.experiment.add_graph(model())
            trainer.validate(model, dm)
            trainer.fit(model, dm)
            trainer.test(model, dm)
        elif args.stage == "test":
            synth_model = DepthEstimationModel.load_from_checkpoint("../lightning_logs/depth/cyst/default/version_5/"
                                                                    "checkpoints/epoch=44-step=103319.ckpt")
            trainer.test(synth_model, dm)
    elif mode == "gan":
        if args.stage == "train":
            unadapted_model = DepthEstimationModel.load_from_checkpoint(unadapted_model)
            gan_model = GAN(depth_model=unadapted_model,
                            res_transfer=args.res_transfer,
                            image_gan=args.imageGAN,
                            adaptive_gating=args.adaptive_gate,
                            warmup_steps=warmup_steps,
                            lr_d=args.lr_d,
                            lr_g=args.lr_g,
                            img_discriminator_factor=args.img_disc_factor,
                            accum_grad_batches=accumulate_grad_batches,
                            res_loss_factor=args.p)
            trainer.validate(gan_model, dm)
            trainer.fit(gan_model, dm)
            trainer.test(gan_model, dm)
        elif args.stage == "test":
            # model_1 = "../lightning_logs/gan/cyst/default/version_14/checkpoints/epoch=19-step=6519.ckpt"
            model_2 = "../lightning_logs/gan/gated_adaptive/cyst/1/lightning_logs/version_18/checkpoints/" \
                      "epoch=51-step=30335.ckpt"
            model = GAN.load_from_checkpoint(model_2,
                                             res_transfer=args.res_transfer,
                                             adaptive_gating=args.adaptive_gate)
            trainer.test(model, dm)

    elif mode == "test-gan":
        # test adapted model
        print("Testing Adapted Model")
        adapted_model = GAN.load_from_checkpoint("../lightning_logs/gan/gated_adaptive/cyst/1/lightning_logs/"
                                                 "version_18/checkpoints/epoch=51-step=30335.ckpt")
        depth_dm = DepthDataModule(batch_size, annotations_path, mode,
                                   dataset_dir=dataset_dir,
                                   generate_data=generate_gan_data)
        trainer.test(adapted_model, depth_dm)