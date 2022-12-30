import torch.utils.data as data
import os
from PIL import Image
from torch import randn

class CustomDataset(data.Dataset):
    def __init__(self, root_dir, transform=None, n_latent=100) -> None:
        super().__init__()

        self.images = os.listdir(root_dir)
        self.root_dir = root_dir
        self.transform = transform
        self.n_latent = n_latent
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_loc = os.path.join(self.root_dir, self.images[index])
        image = Image.open(img_loc).convert('RGB')
        z = randn(size=(self.n_latent, 1, 1))

        return z, self.transform(image)
