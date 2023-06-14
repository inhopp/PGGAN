import os
import torch
from option import get_option
from model import Generator
from torchvision.utils import save_image


@torch.no_grad()
def main(opt):
    dev = dev = torch.device("cuda:{}".format(opt.gpu)
                             if torch.cuda.is_available() else "cpu")
    ft_path = os.path.join(opt.ckpt_root, "Gen.pt")

    model = Generator(
        z_dim=opt.z_dim, in_channels=opt.in_channels, img_channels=3).to(dev)
    model.load_state_dict(torch.load(ft_path))

    alpha = 1.0
    step = 8

    for i in range(10):
        noise = torch.rand(1, opt.z_dim, 1, 1).to(dev)
        img = model(noise, alpha, step)
        save_image(img*0.5+0.5, f"saved_examples/img_{i}.png")

    print("########## inference Finished ###########")


if __name__ == '__main__':
    opt = get_option()
    main(opt)
