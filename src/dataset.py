import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class NIHChestXrayDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, labels=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['Image Index'])
        
        # Carrega a imagem e converte para RGB (ResNet espera 3 canais)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Extrai o vetor binário de 14 posições
        target = torch.tensor(row[self.labels].values.astype('float32'), dtype=torch.float32)
        
        return image, target


def get_transforms():
    """Retorna as transformações para Treino e Validação/Teste."""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=7),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_test_transform