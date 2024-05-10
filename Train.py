print("Loading Libraries ...")

from Neural_Network import Net
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import GTSRB
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import max as tmax
from torch import save as tsave

NUM_CATEGORIES = 43
IMG_SIZE = 32
NUM_CHANNELS = 3

# Loading Dataset
print("Loading GTSRB Dataset ...")
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize(size=(IMG_SIZE, IMG_SIZE)),
                                transforms.Lambda(lambda x: transforms.Normalize(x.mean([1, 2]), x.std([1, 2]))(x))])

train_dataset = GTSRB('./data', "train", transform=transform, download=True)
validation_dataset = GTSRB('./data', "test", transform=transform, download=True)


input_dim = (IMG_SIZE ** 2) * NUM_CHANNELS
output_dim = NUM_CATEGORIES


# Defining Neural Network
print("Building Neural Network ...")
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(dataset=train_dataset, batch_size=20, shuffle=True)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=30, shuffle=False)

layers = [input_dim, 4000, 2000, output_dim]
cnn_layers = [NUM_CHANNELS, 100, 150, 250]
cnn_kernels = [5, 3, 3]
model = Net(layers, cnn_layers, cnn_kernels, IMG_SIZE)

optimizer = optim.Adam(model.parameters(), lr=0.0001)

scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=10)


# Start training
print("Training ...")

epochs = 100

for epoch in range(epochs):
    i = 0
    model.train()
    for x_train, y_train in train_loader:
        optimizer.zero_grad()
        z = model(x_train)
        loss = criterion(z, y_train)
        loss.backward()
        optimizer.step()
        i += 1
        print("  Training Epoch {} ----- {:.2f} %".format(epoch + 1, float(i / len(train_loader) * 100)), end="\r")

    correct = 0
    validation_loss = 0
    i = 0
    model.eval()
    for x_test, y_test in validation_loader:
        z = model(x_test)
        _, label = tmax(z, 1)
        validation_loss += F.nll_loss(z, y_test, size_average=False).data.item()
        correct += (label == y_test).sum().item()
        i += 1
        print("  Validation Epoch {} ----- {:.2f} %".format(epoch + 1, float(i / len(validation_loader) * 100)), end="\r")

    validation_loss /= len(validation_dataset)
    accuracy = 100 * (correct / len(validation_dataset))

    if epoch > 20:
        scheduler.step()

    print("Saving Model ...", end="\r")
    tsave(model.state_dict(), f"model.pt")
    print(f"Epoch {epoch + 1}/{epochs} -------------- Accuracy = {round(accuracy, 2)} %  Loss = {round(validation_loss, 4)}")


# Saving results
print("Saving Model ...")
