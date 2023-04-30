from pathlib import Path
from loguru import logger

import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Subset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data import Data, compare_images, init_save_dir
from srgan import UNet, VGG16Discriminator


def PSNRLoss(batch_1, batch_2):
    """peak signal-to-noise ratio loss"""
    mse = torch.nn.MSELoss()
    mse_loss = mse(batch_1, batch_2)
    psnr = 10 * torch.log10(1 / mse_loss)  # dB maybe use numpy
    return psnr


LOSSES = dict(
    mse=torch.nn.MSELoss(),
    bce=torch.nn.BCELoss(),
    psnr=PSNRLoss,
    # maybe add DiceLoss
)


def main():
    full_set = 20  # running oom for 1000
    data_set = Data(data_dir="./data", n_images=full_set)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info('Using cuda')
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info('Using mps')
    else:
        device = torch.device('cpu')
        logger.info('Using cpu')

    device = torch.device('cpu')

    validation_split = .2  # percent we want to use for validation
    shuffle_dataset = True
    random_seed = 42
    batch_size = 10

    indices = list(range(full_set))
    split = int(np.floor(validation_split * full_set))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_loader = torch.utils.data.DataLoader(
        Subset(data_set, train_indices),
        batch_size=batch_size,
        drop_last=True,
    )
    validation_loader = torch.utils.data.DataLoader(
        Subset(data_set, val_indices),
        batch_size=batch_size * 2,
    )

    lr = 0.0001
    nr_epochs = 15
    n_filters = 8

    batch_losses = []
    epoch_losses = []
    val_losses = []

    save_dir = init_save_dir()
    logger.info(
        f"\nRunning training for {nr_epochs} epochs using"
        f"\nSaving results to:"
        f"\n{save_dir}"
    )

    #  ---- init model settings and models ----

    # to tune the complexity of the generator
    # and thus the complexity of the images
    latent_space = 100

    discriminator = VGG16Discriminator().to(device)
    generator = UNet(n_filters=n_filters).to(device)

    # loss_function = LOSSES[choose_loss]

    content_loss = torch.nn.L1Loss()  # or torch.nn.MSELoss()
    adversarial_loss = torch.nn.BCELoss()

    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    # generator_optimizer = torch.optim.SGD(model.parameters(), lr=lr * 10)

    image = data_set.input_images[3]  # for sample output during training
    losses_df = pd.DataFrame(
        index=range(nr_epochs),
        columns=["d_real", "d_fake", "discr", "gen", "g_val"]
    )
    for epoch in range(nr_epochs):
        # epoch_loss = 0.0
        for n_batch, (input_batch, target_batch) in enumerate(train_loader):
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            discriminator_optimizer.zero_grad()

            real_outputs = discriminator(target_batch)
            real_loss = adversarial_loss(real_outputs, real_labels)
            losses_df.loc[epoch, "d_real"] = real_loss.item()

            fake_images = generator(input_batch)
            fake_outputs = discriminator(fake_images.detach())
            fake_loss = adversarial_loss(fake_outputs, fake_labels)
            losses_df.loc[epoch, "d_fake"] = fake_loss.item()

            discriminator_loss = (real_loss + fake_loss) / 2
            discriminator_loss.backward()
            discriminator_optimizer.step()
            losses_df.loc[epoch, "discr"] = discriminator_loss.item()

            # Train the generator
            generator_optimizer.zero_grad()
            content_outputs = content_loss(generator(input_batch), target_batch)
            adversarial_outputs = adversarial_loss(discriminator(generator(input_batch)), real_labels)
            generator_loss = content_outputs + adversarial_outputs
            generator_loss.backward()
            generator_optimizer.step()

            # logging
            losses_df.loc[epoch, "gen"] = generator_loss.item()
            '\t'.join([f"{k}={v:.3f}" for k, v in losses_df.loc[epoch].to_dict().items()])
            # epoch_loss += loss.item()
            batch_loss = losses_df.loc[epoch].mean()
            batch_losses.append(batch_loss)
        epoch_loss = losses_df.loc[epoch].mean()
        epoch_losses.append(epoch_loss / len(train_loader))
        # logger.info(f'Epoch {epoch}/{nr_epochs}, loss {epoch_losses[-1]}')

        with torch.no_grad():
            logits = generator(image.unsqueeze(0).to(device))
            val_loss = 0
            for val_image_batch, val_target_batch in validation_loader:
                val_image_batch = val_image_batch.to(device)
                val_target_batch = val_target_batch.to(device)

                logits_batch = generator(val_image_batch)
                loss = content_loss(logits_batch, val_target_batch)
                val_loss += loss.item()
            val_losses.append(val_loss / len(validation_loader))
            losses_df.loc[epoch, "g_val"] = val_loss
        print(
            f"Epoch {epoch}/{nr_epochs}\t Losses: \t" + " | ".join(
                [f"{k}={v:06.3f}" for k, v in losses_df.loc[epoch].to_dict().items()])
        )

        plt.show()
        if epoch % 5 == 4:
            logits = generator.relu(logits)
            prob = torch.nn.functional.tanh(logits)

            fig, ax = plt.subplots(1, 2, figsize=(10, 5))

            ax[0].imshow(prob[0, 1].cpu().detach())
            ax[0].imshow(prob.cpu().numpy().squeeze().transpose(1, 2, 0))
            ax[0].set_title(f'Prediction, epoch:{len(epoch_losses) - 1}')

            sns.relplot(losses_df, ax=ax[1], kind="line")
            # ax[1].plot(np.linspace(0, len(epoch_losses), len(batch_losses)),
            #            batch_losses, lw=0.5)  # blue
            # ax[1].plot(np.arange(len(epoch_losses)) + 0.5, epoch_losses, lw=2)  # orange
            # ax[1].plot(np.linspace(0, len(epoch_losses) - 0.5, len(val_losses)),
            #            val_losses, lw=1)  # green
            # ax[1].set_title('Batch loss, epoch loss (training) and test loss')
            # ax[1].set_ylim(0, 1.1 * max(epoch_losses + val_losses))
            plt.savefig(save_dir / f"loss_epoch{epoch}.jpg", dpi=300)

    torch.save(generator.state_dict(), save_dir / "model.pt")

    # export epoch and validation loss as a csv
    pd.DataFrame(
        [epoch_losses, val_losses],
        columns=range(nr_epochs), index=["train", "test"]
    ).T.to_csv(save_dir / "losses.csv")

    # plotting an inference example
    peek_index = 9

    input_image = input_batch[peek_index]
    target_image = target_batch[peek_index]

    logits = generator(input_image.unsqueeze(0).to(device))
    reconstruction = torch.nn.functional.tanh(logits).cpu().detach().numpy().squeeze().transpose(1, 2, 0)

    input_display = input_image.cpu().numpy().squeeze().transpose(1, 2, 0)
    target_display = target_image.cpu().numpy().squeeze().transpose(1, 2, 0)

    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    ax[0, 0].imshow(reconstruction[100:200, 100:200, :])
    ax[0, 1].imshow(input_display[100:200, 100:200, :])
    ax[0, 2].imshow(target_display[100:200, 100:200, :])

    ax[1, 0].imshow(reconstruction[:, :, :])
    ax[1, 1].imshow(input_display[:, :, :])
    ax[1, 2].imshow(target_display[:, :, :])
    plt.savefig(save_dir / "reconstruction_sample.jpg", dpi=300)

    # inferenece_dir = save_dir / "inference"
    # inferenece_dir.mkdir()

    # fig = compare_images(reconstruction, target_image)
    # plt.show()


if __name__ == '__main__':
    main()
