from pathlib import Path

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint


def set_up_loggers():
    csv_logger = CSVLogger(save_dir="./")
    return [csv_logger]


def set_up_ckpt_callbacks():
    ckpt_callback = ModelCheckpoint(
        filename='{epoch:02d}-{valid_loss:.4f}',
        monitor="valid_loss",
        mode='min',
        save_top_k=1,
    )

    return [ckpt_callback]


def get_ckpt_file(ckpt_index):
    ckpt_folder = Path("lightning_logs", f"version_{ckpt_index}", "checkpoints")
    ckpt_file = [file for file in ckpt_folder.iterdir()][0]
    return ckpt_file