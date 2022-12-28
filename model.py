import torch
import torch.nn as nn

class GeneratorBlock(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super(GeneratorBlock, self).__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.model(x)

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down_sampling = 2) -> None:
        super(DiscriminatorBlock, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding='same'),
            nn.MaxPool2d(kernel_size=down_sampling, stride=down_sampling),
            nn.LeakyReLU(0.2, inplace=True),
        )
    
    def forward(self, x):
        return self.model(x)

class Generator(nn.Module):
    def __init__(self, n_latent) -> None:
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=n_latent, out_features=1024 * 4 * 4),
            nn.BatchNorm1d(1024 * 4 * 4),
            nn.ReLU(inplace=True),
            nn.Unflatten(dim=1, unflattened_size=(1024, 4, 4)),
            GeneratorBlock(1024, 512),
            GeneratorBlock(512, 256),
            GeneratorBlock(256, 128),
            GeneratorBlock(128, 64),
            GeneratorBlock(64, 32),

            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            DiscriminatorBlock(3, 128, down_sampling=4),
            DiscriminatorBlock(128, 64),
            DiscriminatorBlock(64, 32),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class GAN():
    def __init__(self, n_latent=100, lr_gen=0.0002, lr_disc=0.001, beta_gen=(0.5, 0.999), beta_disc=(0.5, 0.999)) -> None:
        self.cuda_available = torch.cuda.is_available()

        self.G = Generator(n_latent=n_latent)
        self.D = Discriminator()

        self.criterion = nn.BCELoss()

        if self.cuda_available:
            self.G.cuda()
            self.D.cuda()
            self.criterion.cuda()

        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=lr_gen, betas=beta_gen)
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=lr_disc, betas=beta_disc)

    def generate_fakes(self, z):
        return self.G(z)
    
    def save(self, path='./saved/save'):
        torch.save(self.G.state_dict(), path)

    def load(self, path='./state'):
        self.G.load_state_dict(torch.load(path))
    
    def classify_images(self, images):
        return self.D(images)
    
    def compute_loss(self, predictions, labels):
        return self.criterion(predictions, labels)
