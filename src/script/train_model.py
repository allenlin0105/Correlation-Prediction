from lightning.pytorch import Trainer

from src.data import ImageDataModule
from src.model import CNN
from src.utils.args import set_up_args
from src.utils.lightning import set_up_loggers, set_up_ckpt_callbacks


def main():
    args = set_up_args()

    # datamodule
    datamodule = ImageDataModule(args.data_dir)

    # model
    model = CNN(args.image_size)

    # trainer
    trainer = Trainer(accelerator='gpu', 
                      devices=[args.cuda], 
                      max_epochs=args.n_epoch, 
                      logger=set_up_loggers(),
                      callbacks=set_up_ckpt_callbacks(),
                      deterministic=True,)
    
    # start training
    trainer.fit(model, datamodule)


if __name__=="__main__":
    main()