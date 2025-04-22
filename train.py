import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from models import Z_DCE_Network, SR_CNN
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import cv2 as cv

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Trainer:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.writer = SummaryWriter()
        self.z_dce_network = Z_DCE_Network().to(self.device)
        self.sr_cnn = SR_CNN().to(self.device)
        #self.z_criterion = torch.nn.MSELoss()
        self.sr_criterion = torch.nn.MSELoss()
        self.z_dce_optimizer = torch.optim.Adadelta(self.z_dce_network.parameters(), lr=.003, rho=0.9)
        self.sr_optimizer = torch.optim.Adadelta(self.sr_cnn.parameters(), lr=.0001, rho=0.9)
    
    def self_supervised_loss(self, enhanced_image):
        exposure_loss = torch.mean((enhanced_image - 0.6) ** 2)
        color_loss = torch.mean((enhanced_image[:, :, :, :-1] - enhanced_image[:, :, :, 1:]) ** 2)
        smoothness_loss = torch.mean((enhanced_image[:, :, :-1, :] - enhanced_image[:, :, 1:, :]) ** 2)
        
        total_loss = exposure_loss + 0.5 * color_loss + 0.5 * smoothness_loss
        return total_loss

    def train(self, epochs: int, batch_size: int, data_path: str, z_dce_model_path: str = None, sr_cnn_model_path: str = None):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        if not z_dce_model_path is None:
            self.z_dce_network.load_state_dict(torch.load(z_dce_model_path, map_location=self.device))
        if not sr_cnn_model_path is None:
            self.sr_cnn.load_state_dict(torch.load(sr_cnn_model_path, map_location=self.device))
        dataset = ImageFolder(data_path, transform=transform)
        train_size = int(0.6 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.z_dce_network.train()
            self.sr_cnn.train()
            for i, (images, _) in enumerate(train_loader):
                images = images.to(self.device)
                img_gray = torch.mean(images, dim=1, keepdim=True)
                z = self.z_dce_network(img_gray)
                z = z.repeat(1, 3, 1, 1)
                #z = z #* 0.25 + images * 0.75
                """img_original = images[0].permute(1, 2, 0).cpu().detach().numpy()
                img_original = cv.cvtColor(img_original, cv.COLOR_BGR2RGB)
                cv.imshow("Original Image", img_original)
                img = z[0].permute(1, 2, 0).cpu().detach().numpy()
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                cv.imshow("ZERO Image", img)"""
                sr = self.sr_cnn(z)
                intensity = torch.mean(images, dim=1, keepdim=True)
                sr_temp = sr * 0.15 + z * (2 - intensity) + images * 0.8
                
                if np.mean(sr_temp.cpu().detach().numpy()) < 0.5:
                    sr = sr_temp
                else:
                    sr = sr * 0.15 + images * 0.8
                    
                """img_sr = sr[0].permute(1, 2, 0).cpu().detach().numpy()
                img_sr = cv.cvtColor(img_sr, cv.COLOR_BGR2RGB)
                cv.imshow("SR Image", img_sr)
                cv.waitKey(0)"""
                sr = torch.clamp(sr, 0, 1)
                sr = sr.permute(0, 2, 3, 1)
                images = images.permute(0, 2, 3, 1)
                sr_loss = self.sr_criterion(images, sr)
                z = torch.clamp(z, 0, 1) 
                z_loss = self.self_supervised_loss(z)
                self.z_dce_optimizer.zero_grad()
                self.sr_optimizer.zero_grad()
                sr_loss.backward(retain_graph=True)
                z_loss.backward()
                self.z_dce_optimizer.step()
                self.sr_optimizer.step()
                print(f"Epoch: {epoch+1}/{epochs}, Batch: {i+1}/{len(train_loader)}, Z Loss: {z_loss.item()}, SR Loss: {sr_loss.item()}")
                self.writer.add_scalar("Z Loss", z_loss.item(), epoch*len(train_loader)+i)
                self.writer.add_scalar("SR Loss", sr_loss.item(), epoch*len(train_loader)+i)

            self.validate(val_loader, epoch)
            #Saving model aftyer every epoch
            torch.save(self.z_dce_network.state_dict(), f"z_dce_network_ZESR.pth")
            torch.save(self.sr_cnn.state_dict(), f"sr_cnn_ZESR.pth")

    def validate(self, val_loader, epoch):
        self.z_dce_network.eval()
        self.sr_cnn.eval()
        for i, (images, _) in enumerate(val_loader):
            images = images.to(self.device)
            img_gray = torch.mean(images, dim=1, keepdim=True)
            z = self.z_dce_network(img_gray)
            z = z.repeat(1, 3, 1, 1)
                #z = z #* 0.25 + images * 0.75
            """img_original = images[0].permute(1, 2, 0).cpu().detach().numpy()
            img_original = cv.cvtColor(img_original, cv.COLOR_BGR2RGB)
            cv.imshow("Original Image", img_original)
            img = z[0].permute(1, 2, 0).cpu().detach().numpy()
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            cv.imshow("ZERO Image", img)"""
            sr = self.sr_cnn(z)
            intensity = torch.mean(images, dim=1, keepdim=True)
            sr_temp = sr * 0.15 + z * (2 - intensity) + images * 0.8
                
            if np.mean(sr_temp.cpu().detach().numpy()) < 0.5:
                sr = sr_temp
            else:
                sr = sr * 0.15 + images * 0.8
                    
            """img_sr = sr[0].permute(1, 2, 0).cpu().detach().numpy()
            img_sr = cv.cvtColor(img_sr, cv.COLOR_BGR2RGB)
            cv.imshow("SR Image", img_sr)
            cv.waitKey(0)"""
            sr = torch.clamp(sr, 0, 1)
            sr = sr.permute(0, 2, 3, 1)
            images = images.permute(0, 2, 3, 1)
            sr_loss = self.sr_criterion(images, sr)
            z = torch.clamp(z, 0, 1) 
            z_loss = self.self_supervised_loss(z)
            print(f"Validation Epoch: {epoch+1}, Batch: {i+1}/{len(val_loader)}, Z Loss: {z_loss.item()}, SR Loss: {sr_loss.item()}")
            self.writer.add_scalar("Validation Z Loss", z_loss.item(), epoch*len(val_loader)+i)
            self.writer.add_scalar("Validation SR Loss", sr_loss.item(), epoch*len(val_loader)+i)
            

    cv.waitKey(0)


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train(epochs=10, batch_size=32, data_path=r"C:\Users\virad\ReX\CVIP\Project\vimeo_test_clean\sequences")