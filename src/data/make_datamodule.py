import csv
from pathlib import Path

import torch
import lightning as L
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


class ImageDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = Path(data_dir)
        
        self.fit_ratio = 0.8
        self.batch_size = 256

    def _load_images(self, image_files):
        transform = transforms.Compose([transforms.ToTensor()])
        images = [transform(Image.open(image_file)) for image_file in tqdm(image_files)]
        return images

    def setup(self, stage: str):
        image_files, labels = [], []
        with open(self.data_dir.joinpath("responses.csv"), "r") as fp:
            reader = csv.reader(fp)
            next(reader)
            for row in reader:
                image_id, corr = row
                image_files.append(self.data_dir.joinpath("images", image_id + ".png"))
                labels.append(float(corr))

        n = len(image_files)
        split_index = int(n * self.fit_ratio)

        # train & validation
        if stage == "fit":
            images = self._load_images(image_files[:split_index])
            n_train = int(split_index * 0.8)
            
            self.train_dataset, self.valid_dataset = random_split(
                TensorDataset(
                    torch.cat(images).unsqueeze(1),
                    torch.tensor(labels[:split_index]).unsqueeze(1),
                ),
                lengths=[n_train, split_index - n_train],
                generator=torch.Generator().manual_seed(42),
            )

        # test
        if stage == "test":
            images = self._load_images(image_files[split_index:])
            
            self.test_dataset = TensorDataset(
                torch.cat(images).unsqueeze(1),
                torch.tensor(labels[split_index:]).unsqueeze(1),
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


if __name__=="__main__":
    data_dir = "data/"
    datamodule = ImageDataModule(data_dir=data_dir)
    print("setup fit")
    datamodule.setup("fit")
    print("setup test (should not load data twice)")
    datamodule.setup("test")

    train_dataloader = datamodule.train_dataloader()
    for batch in train_dataloader:
        print(batch[0].size())
        print(batch[1].size())
        break