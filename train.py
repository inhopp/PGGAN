import os
import torch
import torch.nn as nn
import torch.optim as optim
from data import generate_loader
from option import get_option
from model import Generator, Discriminator
from tqdm import tqdm
from math import log2


class Solver():
    def __init__(self, opt):
        self.opt = opt

        self.dev = torch.device("cuda:{}".format(
            opt.gpu) if torch.cuda.is_available() else "cpu")
        print("device: ", self.dev)

        self.generator = Generator(
            z_dim=opt.z_dim, in_channels=opt.in_channels, img_channels=3).to(self.dev)
        self.discriminator = Discriminator(
            in_channels=opt.in_channels, img_channels=3).to(self.dev)

        if opt.pretrained:
            load_path = os.path.join(opt.chpt_root, "Gen.pt")
            self.generator.load_state_dict(torch.load(load_path))

            load_path = os.path.join(opt.chpt_root, "Disc.pt")
            self.discriminator.load_state_dict(torch.load(load_path))

        if opt.multigpu:
            self.generator = nn.DataParallel(
                self.generator, device_ids=opt.device_ids).to(self.dev)
            self.discriminator == nn.DataParallel(
                self.discriminator, device_ids=opt.device_ids).to(self.dev)

        print("# Generator params:", sum(
            map(lambda x: x.numel(), self.generator.parameters())))
        print("# Discriminator params:", sum(
            map(lambda x: x.numel(), self.discriminator.parameters())))

        # Define Loss

        self.optimizer_G = optim.Adam(
            self.generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    def fit(self):
        opt = self.opt
        print("start training")

        step = int(log2(opt.start_img_size / 4))
        progressive_epochs = [30] * len(opt.batch_size)

        for num_epochs in progressive_epochs[step:]:
            alpha = 1e-5
            loader, dataset = generate_loader(opt, 4 * 2 ** step)

            for epoch in range(num_epochs):
                loop = tqdm(loader)

                for i, img in enumerate(loop):
                    real = img.to(self.dev)
                    cur_batch_size = real.shape[0]

                    noise = torch.rand(
                        cur_batch_size, opt.z_dim, 1, 1).to(self.dev)
                    fake = self.generator(noise, alpha, step)

                    # Train Discriminator
                    D_real = self.discriminator(real, alpha, step)
                    D_fake = self.discriminator(fake.detach(), alpha, step)
                    gp = self.gradient_penalty(real, fake, alpha, step)

                    D_loss = (-(torch.mean(D_real) - torch.mean(D_fake)) +
                              (opt.gp_lambda * gp) + (0.001 * torch.mean(D_real ** 2)))

                    self.optimizer_D.zero_grad()
                    D_loss.backward()
                    self.optimizer_D.step()

                    # Train Generator
                    G_fake = self.discriminator(fake, alpha, step)
                    G_loss = -torch.mean(G_fake)

                    self.optimizer_G.zero_grad()
                    G_loss.backward()
                    self.optimizer_G.step()

                    # update alpha and ensure less than 1
                    alpha += cur_batch_size / \
                        ((progressive_epochs[step] * 0.5) * len(dataset))
                    alpha = min(alpha, 1)

            step += 1
            self.save()

    def gradient_penalty(self, real, fake, alpha, train_step):
        B, C, H, W = real.shape
        beta = torch.rand((B, 1, 1, 1)).repeat(1, C, H, W).to(self.dev)
        interporated_images = real * beta + fake.detach() * (1 - beta)
        interporated_images.requires_grad_(True)

        mixed_scores = self.discriminator(
            interporated_images, alpha, train_step)

        gradient = torch.autograd.grad(inputs=interporated_images, outputs=mixed_scores, grad_outputs=torch.ones_like(
            mixed_scores), create_graph=True, retain_graph=True)[0]
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

        return gradient_penalty

    def save(self):
        os.makedirs(os.path.join(self.opt.ckpt_root), exist_ok=True)
        G_save_path = os.path.join(self.opt.ckpt_root, "Gen.pt")
        D_save_path = os.path.join(self.opt.ckpt_root, "Disc.pt")
        torch.save(self.generator.state_dict(), G_save_path)
        torch.save(self.discriminator.state_dict(), D_save_path)


def main():
    opt = get_option()
    torch.manual_seed(opt.seed)
    solver = Solver(opt)
    solver.fit()


if __name__ == "__main__":
    main()
