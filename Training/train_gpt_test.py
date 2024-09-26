import argparse
#import logging
import os
import sys
import pytorch_lightning as pl
from src.videogpt.gpt import VideoGPT
from src.datasets import MNISTDataModule, VideoDataModule
#from src.utils import set_logger
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm
from pytorch_lightning.loggers import TensorBoardLogger  

from pytorch_lightning.callbacks import Callback
import numpy as np 


'''

class MetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.train_loss = []
        self.train_accuracy = []

    def on_epoch_end(self, trainer, pl_module):
        # Save training loss and accuracy at the end of each epoch
        self.train_loss.append(trainer.logged_metrics.get('train_loss'))
        self.train_accuracy.append(trainer.logged_metrics.get('train_accuracy'))

    def on_train_end(self, trainer, pl_module):
        # Save the metrics to files using numpy at the end of training
        # np.save("metrics/train_loss.npy", np.array(self.train_loss))
        # np.save("metrics/train_accuracy.npy", np.array(self.train_accuracy))
        np.save("metrics/train_loss.npy", np.array(self.train_loss, dtype=np.float32))
        np.save("metrics/train_accuracy.npy", np.array(self.train_accuracy, dtype=np.float32))

'''

val_loss_values = []

class SaveValLossCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val/loss")
        val_loss_values.append(val_loss)




def main(args):
    sequence_length = 2 * args.sequence_length if args.emb_cond else args.sequence_length

    if args.dataset == "mnist":
        desc = "moving mnist"
        TNG_SIZE = 10000
        VAL_SIZE = 1000
        datamodule = MNISTDataModule(
            sequence_length, TNG_SIZE, VAL_SIZE, args.batch_size, args.num_workers
        )

    elif args.dataset == "Shinobi":
        desc = "_".join(("Shinobi"))
        #levels = args.levels.split(",") if args.levels != "all" else "all"
        #subjects = args.subjects.split(",") if args.subjects != "all" else "all"
        #test_tag = "ses-shinobi_005"
        #val_tag = "ses-shinobi_004"
        
        tng_list=[f.strip() for f  in args.tng_list.split(",")]
        val_list=[f.strip() for f in args.val_list.split(",")]


        lazy_load_dir = os.path.join(os.path.split(args.data_path)[0], "lazy_loading")
        os.makedirs(lazy_load_dir, exist_ok=True)
        datamodule = VideoDataModule(
            args.data_path,
            lazy_load_dir,
            n_frames=sequence_length,
            #levels=levels,
            tng_list=tng_list,
            val_list=val_list,
            #test_tag=test_tag,
            #val_tag=val_tag,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            stride=16,
            time_downsample=4,
            #subjects=subjects,
        )

    args.class_cond_dim = None
    args.max_steps = 200000
    gpt = VideoGPT(args, desc=desc)

    
    


    #callbacks = [ModelCheckpoint(monitor="val/loss", mode="min", save_last=True), MetricsCallback()]
    callbacks = ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_last=False,
            filename="best_recon_{epoch}-{step}"
        )
    #callbacks = [ModelCheckpoint(monitor="val/loss", mode="min", save_last=True)]
    #callbacks = [ModelCheckpoint(monitor="val_loss", mode="min", save_last=True), HistoryCallback()]
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
    #from pytorch_lightning.profiler import AdvancedProfiler
    #from pytorch_lightning.profiler import PyTorchProfiler

    #profiler = PyTorchProfiler(filename="perf-logs", profile_memory=True)
    
   
    kwargs = {}
    if args.gpus > 1:
        # find_unused_parameters = False to support gradient checkpointing
        kwargs = dict(
            accelerator="ddp",
            gpus=args.gpus,
            plugins=[pl.plugins.DDPPlugin(find_unused_parameters=False)],
            #strategy=pl.strategies.DDPStrategy(find_unused_parameters=False)
        )

    
    # Create an instance of the custom callback
    save_val_loss_instance = SaveValLossCallback()
    #tensorboard_logger = TensorBoardLogger(save_dir="./logs", name="sana_experiment")
    #trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, max_steps=200000, **kwargs)
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[callbacks, save_val_loss_instance], max_steps=200000, **kwargs)

    #trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=tensorboard_logger, max_steps=200000, **kwargs)

    trainer.fit(gpt, datamodule)
    '''   
    print(
        f"Best model saved at {callbacks[0].best_model_path}, \
           with a val loss of {callbacks[0].best_model_score}"
    )
    '''
    #val_loss_array = np.array(val_loss_values)
    #np.save("metrics/val_loss_values.npy", val_loss_array)
    val_loss_array = np.array([item.cpu().numpy() for item in val_loss_values])
    np.save("metrics/16bit_batchsize64_val_loss_values.npy", val_loss_array)

    '''
    val_loss = trainer.logged_metrics['val_loss']
    val_accuracy = trainer.logged_metrics['val_accuracy']
    train_accuracy = trainer.logged_metrics['train_accuracy']
    train_loss = trainer.logged_metrics['train_loss']

    # Create a directory to save the metrics
    os.makedirs("metrics", exist_ok=True)

    # Save metrics to files using numpy
    np.save("metrics/val_loss.npy", np.array(val_loss))
    np.save("metrics/val_accuracy.npy", np.array(val_accuracy))
    np.save("metrics/train_accuracy.npy", np.array(train_accuracy))
    np.save("metrics/train_loss.npy", np.array(train_loss))
 
    '''
if __name__ == "__main__":
    pl.seed_everything(1234)
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = VideoGPT.add_model_specific_args(parser)
    parser.add_argument("--dataset", type=str, default="Shinobi", choices=["mnist", "Shinobi"])
    
    parser.add_argument("--tng_list", type=str)
    parser.add_argument("--val_list", type=str)

    parser.add_argument("--sequence_length", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    print("\nArguments :\n", sys.argv[1:], "\n\n")
    main(args)
