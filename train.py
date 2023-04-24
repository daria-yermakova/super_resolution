import torch
import matplotlib.pyplot as plt
import numpy as np

from image_preprocessing import get_cropped_path

from config import PATCH_SIZE, STRIDE
from data import Data


class UNet(torch.nn.Module):
    """Takes in patches of 128/512^2 RGB, returns 88^2"""

    def __init__(self, input_channels=3, out_channels=3, n_filters=64):
        super().__init__()
        # Learnable
        self.conv1A = torch.nn.Conv2d(input_channels, n_filters, 3)
        self.conv1B = torch.nn.Conv2d(n_filters, n_filters, 3)
        self.conv2A = torch.nn.Conv2d(n_filters, 2 * n_filters, 3)
        self.conv2B = torch.nn.Conv2d(2 * n_filters, 2 * n_filters, 3)
        self.conv3A = torch.nn.Conv2d(2 * n_filters, 3 * n_filters, 3)
        self.conv3B = torch.nn.Conv2d(3 * n_filters, 3 * n_filters, 3)
        self.conv4A = torch.nn.Conv2d(3 * n_filters, 2 * n_filters, 3)
        self.conv4B = torch.nn.Conv2d(2 * n_filters, 2 * n_filters, 3)
        self.conv5A = torch.nn.Conv2d(2 * n_filters, n_filters, 3)
        self.conv5B = torch.nn.Conv2d(n_filters, n_filters, 3)
        self.convtrans34 = torch.nn.ConvTranspose2d(3 * n_filters, 2 * n_filters, 2, stride=2)
        self.convtrans45 = torch.nn.ConvTranspose2d(2 * n_filters, n_filters, 2, stride=2)

        self.convfinal = torch.nn.Conv2d(n_filters, out_channels, 1)

        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        # Down, keeping layer outputs we'll need later. todo make use of torch.sequential instead
        l1 = self.relu(self.conv1B(self.relu(self.conv1A(x))))
        l2 = self.relu(self.conv2B(self.relu(self.conv2A(self.pool(l1)))))
        out = self.relu(self.conv3B(self.relu(self.conv3A(self.pool(l2)))))

        # Up, now we overwritte out in each step.
        out = torch.cat([self.convtrans34(out), l2[:, :, 4:-4, 4:-4]], dim=1)  # copy & crop
        # out = torch.cat([self.convtrans34(out), l2], dim=1)  # copy & no crop needs convtrans to change
        out = self.relu(self.conv4B(self.relu(self.conv4A(out))))

        # out = torch.cat([self.convtrans45(out), l1[:, :, 16:-16, 16:-16]], dim=1)
        out = torch.cat([self.convtrans45(out), l1], dim=1)  # copy & no crop
        out = self.relu(self.conv5B(self.relu(self.conv5A(out))))

        # Finishing
        out = self.convfinal(out)

        return out


images = Data(data_dir="./data", n_images=100)
x_train, y_train, x_test, y_test = images.data_train_test_split()

# Initiate the model, dataloaders and optimizer.
lr = 0.0001
nr_epochs = 50
device = "cpu"

model = UNet().to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# losses = pd.DataFrame(columns=["per_batch", "per_epoch", "test"]) todo
batch_losses = []
epoch_losses = []
test_losses = []
#  Loaders for training and testing set
trainloader = torch.utils.data.DataLoader(
    # [x_train, y_train],
    [x_train.dataset, y_train.dataset],
    batch_size=10,
    shuffle=True,
    drop_last=True
)
testloader = torch.utils.data.DataLoader(
    x_test.dataset,
    batch_size=20
)
for batch in trainloader:
    image_batch, label_batch = batch  # unpack the data
    image_batch = image_batch.to(device)
    label_batch = label_batch.to(device)

image = x_train[0]
image.to("cpu")
logits_batch = model(image.unsqueeze(0).to(device))
optimizer.zero_grad()
loss = loss_function(logits_batch, label_batch)
loss.backward()
optimizer.step()


def train():
    for epoch in range(nr_epochs):
        print(f'Epoch {epoch}/{nr_epochs}', end='')

        epoch_loss = 0.0
        for batch in trainloader:
            image_batch, label_batch = batch  # unpack the data
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)

            logits_batch = model(image_batch)
            optimizer.zero_grad()
            loss = loss_function(logits_batch, label_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_losses.append(loss.item())

        epoch_losses.append(epoch_loss / len(trainloader))
        print(f', loss {epoch_losses[-1]}')

        if epoch % 10 == 9:
            #  Book-keeping and visualizing every tenth iterations
            with torch.no_grad():
                logits = model(image.unsqueeze(0).to(device))
                test_loss = 0
                for batch in testloader:
                    image_batch, label_batch = batch  # unpack the data
                    image_batch = image_batch.to(device)
                    label_batch = label_batch.to(device)
                    logits_batch = model(image_batch)
                    loss = loss_function(logits_batch, label_batch)
                    test_loss += loss.item()
                test_losses.append(test_loss / len(testloader))

            prob = torch.nn.functional.softmax(logits, dim=1)

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
