import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import pytorch_lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import mlflow

class SiameseDataset():
    def __init__(self,training_csv=None,training_dir=None,transform=None):
        self.train_df=pd.read_csv(training_csv)
        self.train_df.columns =["image1","image2","label"]
        self.train_dir = training_dir   
        self.transform = transform

    def __getitem__(self,index):
        image1_path=os.path.join(self.train_dir,self.train_df.iat[index,0])
        image2_path=os.path.join(self.train_dir,self.train_df.iat[index,1])
        img0 = Image.open(image1_path)
        img1 = Image.open(image2_path)
        img0 = img0.convert("L")
        img1 = img1.convert("L")
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            label = torch.tensor([self.train_df.iat[index, 2]], dtype=torch.float32)
        return img0, img1 , label
    
    def __len__(self):
        return len(self.train_df)
    

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)
        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        return torch.sum(loss) / 2.0 / x0.size()[0]


class SiameseNeuralNet(L.LightningModule):
    def __init__(self, lr=1e-3, margin=1.0):
        super().__init__()
        self.save_hyperparameters()
        
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(30976, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)
        )

        self.criterion = ContrastiveLoss(margin=self.hparams.margin)

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        return self.fc1(output)

    def forward(self, input1, input2):
        return self.forward_once(input1), self.forward_once(input2)

    def training_step(self, batch, batch_idx):
        img0, img1, label = batch
        out1, out2 = self(img0, img1)
        loss = self.criterion(out1, out2, label)
        
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=0.0005)
    
    def validation_step(self, batch, batch_idx):
        test_img0, test_img1, test_label = batch
        test_out1, test_out2 = self(test_img0, test_img1)
        val_loss = self.criterion(test_out1, test_out2, test_label)
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True)

    
if __name__ == "__main__":

    mlf_logger = MLFlowLogger(
            experiment_name="siamese_experiment", 
            tracking_uri="./ml-runs",
            log_model="all" 
        )


    print(torch.cuda.is_available())
    transform = transforms.Compose([
        transforms.Resize((105, 105)),
        transforms.ToTensor()
    ])

    dataset = SiameseDataset(
        training_csv=r"C:\Users\Gurpreet\siamese_network\src\data\train\train_small.csv", 
        training_dir=r"C:\Users\Gurpreet\siamese_network\src\data\train\train", 
        transform=transform
    )

    validation_dataset = SiameseDataset(
        training_csv=r"C:\Users\Gurpreet\siamese_network\src\data\test\test_data.csv",
        training_dir=r"C:\Users\Gurpreet\siamese_network\src\data\test\test",
        transform=transform

    )
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    valiidation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=4)


    model = SiameseNeuralNet(lr=1e-3, margin=1.0)


    checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints",         
            filename="siamese-{epoch:02d}", 
            every_n_epochs=2,            
            save_top_k=-1,                
            save_on_train_epoch_end=True  
        )

    trainer = L.Trainer(
        max_epochs=20,
        accelerator="auto", 
        devices=1,
        logger=mlf_logger
    )

    trainer.fit(model, train_dataloader, valiidation_dataloader)
    torch.save(model.state_dict(), "siamese_model.pth")
    mlflow.pytorch.log_model(model, artifact_path="model")