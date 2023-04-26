import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from data import Data


class UNet(torch.nn.Module):
    """"""

    def __init__(self, input_channels=3, out_channels=3, n_filters=64):
        super().__init__()
        # Learnable
        self.conv1A = torch.nn.Conv2d(input_channels, n_filters, 3)
        self.conv1B = torch.nn.Conv2d(n_filters, n_filters, 3)
        self.conv2A = torch.nn.Conv2d(n_filters, 2 * n_filters, 3)
        self.conv2B = torch.nn.Conv2d(2 * n_filters, 2 * n_filters, 3)
        self.conv3A = torch.nn.Conv2d(2 * n_filters, 4 * n_filters, 3)
        self.conv3B = torch.nn.Conv2d(4 * n_filters, 4 * n_filters, 3)
        self.conv4A = torch.nn.Conv2d(4 * n_filters, 2 * n_filters, 3)
        self.conv4B = torch.nn.Conv2d(2 * n_filters, 2 * n_filters, 3)
        self.conv5A = torch.nn.Conv2d(2 * n_filters, n_filters, 3)
        self.conv5B = torch.nn.Conv2d(n_filters, n_filters, 3)
        self.convtrans34 = torch.nn.ConvTranspose2d(4 * n_filters, 2 * n_filters, 2, stride=2)
        self.convtrans45 = torch.nn.ConvTranspose2d(2 * n_filters, n_filters, 2, stride=2)

        self.convfinal = torch.nn.Conv2d(n_filters, out_channels, 1)

        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        # contraction, keeping layer outputs we'll need later. todo make use of torch.sequential instead
        l1 = self.relu(self.conv1B(self.relu(self.conv1A(x))))
        l2 = self.relu(self.conv2B(self.relu(self.conv2A(self.pool(l1)))))
        out = self.relu(self.conv3B(self.relu(self.conv3A(self.pool(l2)))))
        # expansion
        out = torch.cat([self.convtrans34(out), l2[:, :, 4:-4, 4:-4]], dim=1)  # copy & crop
        # out = torch.cat([self.convtrans34(out), l2], dim=1)  # copy & no crop needs convtrans to change
        out = self.relu(self.conv4B(self.relu(self.conv4A(out))))

        out = torch.cat([self.convtrans45(out), l1[:, :, 16:-16, 16:-16]], dim=1)
        # out = torch.cat([self.convtrans45(out), l1], dim=1)  # copy & no crop
        out = self.relu(self.conv5B(self.relu(self.conv5A(out))))

        out = self.convfinal(out)
        return out


def PSNRLoss(batch_1, batch_2):
    """peak signal-to-noise ratio loss"""
    mse = torch.nn.MSELoss()
    mse_loss = mse(batch_1, batch_2)
    psnr = 10 * torch.log10(1 / mse_loss)
    return psnr


full_set = 20  # running oom for 1000
data_set = Data(data_dir="./data", n_images=full_set)

losses = dict(
    mse=torch.nn.MSELoss(),
    bce=torch.nn.BCELoss(),
    psnr=PSNRLoss,
    # maybe add DiceLoss
)
data_loader_config = dict(
    validation_split=.2,
    shuffle_dataset=True,
    random_seed=42,
)
indices = list(range(full_set))
split = int(np.floor(data_loader_config["validation_split"] * full_set))
if data_loader_config["shuffle_dataset"]:
    np.random.seed(data_loader_config["random_seed"])
    np.random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(
    data_set,
    batch_size=10,
    drop_last=True,
    sampler=train_sampler
)
validation_loader = torch.utils.data.DataLoader(
    data_set,
    batch_size=20,
    sampler=valid_sampler
)

# Initiate the model, dataloaders and optimizer.
lr = 0.0001
nr_epochs = 10
device = "cpu"
loss = "psnr"
n_filters = 8

model = UNet(n_filters=n_filters).to(device)
loss_function = losses[loss]
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# losses = pd.DataFrame(columns=["per_batch", "per_epoch", "test"]) todo

batch_losses = []
epoch_losses = []
test_losses = []
image = data_set.input_images[3]


for epoch in range(nr_epochs):
    print(f'Epoch {epoch}/{nr_epochs}', end='')

    epoch_loss = 0.0
    for batch in train_loader:
        image_batch, target_batch = batch  # unpack the data
        image_batch = image_batch.to(device)
        target_batch = target_batch.to(device)

        logits_batch = model(image_batch)
        optimizer.zero_grad()
        loss = loss_function(logits_batch, target_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        batch_losses.append(loss.item())

    epoch_losses.append(epoch_loss / len(train_loader))
    print(f', loss {epoch_losses[-1]}')

    if epoch % 10 == 9:
        #  Bookkeeping and visualizing every tenth iterations
        with torch.no_grad():
            logits = model(image.unsqueeze(0).to(device))
            test_loss = 0
            for validation_batch in validation_loader:
                val_image_batch, val_target_batch = validation_batch  # unpack the data
                val_image_batch = val_image_batch.to(device)
                val_target_batch = val_target_batch.to(device)
                logits_batch = model(val_image_batch)
                val_loss = loss_function(logits_batch, val_target_batch)
                val_loss += val_loss.item()
            test_losses.append(val_loss)  # / len(validation_loader))

        # prob = torch.nn.functional.softmax(logits, dim=1)
        prob = torch.nn.functional.tanh(logits)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(prob[0, 1].cpu().detach())
        ax[0].set_title(f'Prediction, epoch:{len(epoch_losses) - 1}')

        ax[1].plot(np.linspace(0, len(epoch_losses), len(batch_losses)),
                   batch_losses, lw=0.5)
        ax[1].plot(np.arange(len(epoch_losses)) + 0.5, epoch_losses, lw=2)
        ax[1].plot(np.linspace(9.5, len(epoch_losses) - 0.5, len(test_losses)),
                   test_losses, lw=1)
        ax[1].set_title('Batch loss, epoch loss (training) and test loss')
        ax[1].set_ylim(0, 1.1 * max(epoch_losses + test_losses))
        plt.show()

# check
fig, ax = plt.subplots(2, image_batch.shape[0], figsize=(20, 5))
image_batch, target_batch = batch
for i, (im, tar) in enumerate(zip(image_batch, target_batch)):
    input_display = im.cpu().numpy().squeeze().transpose(1, 2, 0)
    target_display = tar.cpu().numpy().squeeze().transpose(1, 2, 0)
    ax[0, i].imshow(input_display[:, :, :])
    ax[1, i].imshow(target_display[:, :, :])
plt.show()

# %%
peek_index = 0
input_image = image_batch[peek_index]
target_image = target_batch[peek_index]
logits = model(input_image.unsqueeze(0).to(device))

reconstruction = torch.nn.functional.tanh(logits).cpu().detach().numpy().squeeze().transpose(1, 2, 0)
input_display = input_image.cpu().numpy().squeeze().transpose(1, 2, 0)
target_display = target_image.cpu().numpy().squeeze().transpose(1, 2, 0)

fig, ax = plt.subplots(2, 3, figsize=(15, 10))
ax[0, 0].imshow(reconstruction[100:200, 100:200, 1])
ax[0, 1].imshow(input_display[100:200, 100:200, 1])
ax[0, 2].imshow(target_display[100:200, 100:200, 1])

ax[1, 0].imshow(reconstruction[:, :, 1])
ax[1, 1].imshow(input_display[:, :, 1])
ax[1, 2].imshow(target_display[:, :, 1])
plt.show()
