from lightning.pytorch import Trainer

from src.data import ImageDataModule
from src.model import CNN
from src.utils.args import set_up_args
from src.utils.lightning import get_ckpt_file


def test():
    args = set_up_args()

    # datamodule
    datamodule = ImageDataModule(args.data_dir)

    # model
    ckpt_file = get_ckpt_file(args.test_ckpt)
    model = CNN.load_from_checkpoint(ckpt_file)

    # trainer
    trainer = Trainer(accelerator='gpu', 
                      devices=[args.cuda],
                      logger=False,
                      deterministic=True,)
    
    # start testing
    trainer.test(model, datamodule)


if __name__=="__main__":
    test()