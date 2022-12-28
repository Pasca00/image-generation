import numpy as np
import torch.utils.data as data
from torch.autograd import Variable
from torchvision import datasets, transforms
import argparse
import model
from data_loader import CustomDataset
from matplotlib.pyplot import imshow, show
import os

if __name__ == '__main__':
    if not os.path.exists("./images/"):
        os.makedirs("./images/")

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=1, help="number of cpu threads")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    opt = parser.parse_args()

    transorm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    dataset = CustomDataset('./images/', transform=transorm)
    dataloader = data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    z, image = next(iter(dataloader))

    # imshow(image[0].permute(1, 2, 0))
    # show()

    gan = model.GAN()

    for _ in range(opt.n_epochs):
        for idx, (z, images) in enumerate(dataloader):
            # Torch not compiled with CUDA enabled
            z, images = Variable(z), Variable(images)

            if gan.cuda_available:
                z = z.cuda()
                images = images.cuda()

            fake_images = gan.generate_fakes(z)

            


