import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.utils import save_image
import argparse
import model
from data_loader import CustomDataset
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    if not os.path.exists("./images/"):
        os.makedirs("./images/")
    
    if not os.path.exists("./saved/"):
        os.makedirs("./saved/")

    if not os.path.exists("./output/"):
        os.makedirs("./output/")

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=1, help="number of cpu threads")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--k", type=int, default=1, help="number of times discrimator is trained before the generator is trained once")
    opt = parser.parse_args()

    denormalize = transforms.Normalize((-1, -1, -1), (2, 2, 2))
    transorm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.25),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CustomDataset('./images/', transform=transorm, n_latent=opt.latent_dim)
    dataloader = data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    z, image = next(iter(dataloader))

    gan = model.GAN()

    real_label = 1
    fake_label = 0

    save_out = './output/'

    g_losses = []
    d_losses = []

    for curr_epoch in range(opt.n_epochs):
        for idx, (z, real_images) in enumerate(dataloader):
            real_labels = torch.ones(real_images.size(0), 1)
            fake_labels = torch.zeros(real_images.size(0), 1)

            if gan.cuda_available:
                z = z.cuda()
                real_images = real_images.cuda()
                real_labels = real_labels.cuda()
                fake_labels = fake_labels.cuda()
            
            # Train Discriminator
            gan.D.zero_grad()
            fake_images = gan.generate_fakes(z)

            predictions_real = gan.classify_images(real_images)
            disc_real_loss = gan.compute_loss(predictions_real, real_labels)
            disc_real_loss.backward()

            predictions_fake = gan.classify_images(fake_images.detach())

            disc_fake_loss = gan.compute_loss(predictions_fake, fake_labels)
            disc_fake_loss.backward()

            disc_loss = (disc_real_loss + disc_fake_loss)

            # disc_loss.backward()
            gan.optimizer_D.step()

            # Train Generator
            gan.G.zero_grad()

            # fake_images = gan.generate_fakes(z)
            predictions_fake = gan.classify_images(fake_images)

            gen_loss = gan.compute_loss(predictions_fake, real_labels)
            gen_loss.backward()
            gan.optimizer_G.step()

            if (idx % 100 == 0 and idx != 0):
                print('Epoch: %d/%d, batch: %d/%d - loss D: %.3f, loss G: %.3f'
                    % (curr_epoch + 1, opt.n_epochs, idx, len(dataloader), 
                    disc_loss.item(), gen_loss.item()))

                image_name = os.path.join(save_out, 'e_{}_b_{}.jpg'.format(curr_epoch, idx))
                save_image(denormalize(fake_images.detach().cpu()), image_name)

            g_losses.append(gen_loss.detach().cpu())
            d_losses.append(disc_loss.detach().cpu())

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses,label="G")
    plt.plot(d_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    gan.save()
            


