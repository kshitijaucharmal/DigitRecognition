import math
# Normal torch libraries
import torch
import torch.nn as nn # Torch neural network module
import torch.optim as optim # Torch optimizer module

# For using MNIST Database (Set Download to true if not downloaded already)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# For plotting images
import matplotlib.pyplot as plt

# For making a screen and for drawing
import pygame, sys

# For Image compression and stuff
from PIL import Image

class NeuralNet(nn.Module):
    def __init__(self, inodes, hnodes1, hnodes2, onodes, lr):
        super(NeuralNet, self).__init__()
        self.inodes = inodes
        self.hnodes1 = hnodes1
        self.hnodes2 = hnodes2
        self.onodes = onodes
        self.lr = lr

        self.l1 = nn.Linear(self.inodes, self.hnodes1)
        self.l2 = nn.Linear(self.hnodes1, self.hnodes2)
        self.l3 = nn.Linear(self.hnodes2, self.onodes)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

        self.criterion = nn.NLLLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=self.lr)
        pass

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        x = self.softmax(x)
        return x

    def backward(self, x, y):
        self.optimizer.zero_grad()
        output = self(x)
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()
        return loss

    def train(self, trainloader):
        running_loss = 0
        for images, labels in trainloader:
            images = images.view(images.shape[0], -1)
            
            running_loss += self.backward(images, labels).item()
        else:
            print(f"Training loss: {running_loss/len(trainloader)}")
        pass

# model = NeuralNet(784, 128, 64, 10, 0.003)
model = torch.load('.pytorch/model1.pt')

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# Download and load the training data
trainset = datasets.MNIST('.pytorch/MNIST_data/', download=False, train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.MNIST('.pytorch/MNIST_data/', download=False, train=False, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=True)

# Training Done --------------------
# for i in range(30):
    # model.train(trainloader)

# torch.save(model, '.pytorch/model1.pt')
# ----------------------------------

# Image Compression by cv2 ---------------------
def compress_image(path, show=False):
    image = Image.open(path)
    image = image.resize((28, 28))
    image = image.convert('L')
    image.save(path)
    if show:
        plt.imshow(image)
        plt.show()
    return image
#-----------------------------------------------

# images, labels = next(iter(testloader))
# img = images[0].view(1, 784)

# ps = torch.exp(model(img))
# print(torch.argmax(ps))

# plt.imshow(img.view(28, 28))
# plt.show()

def window():
    pygame.init()
    ds = pygame.display.set_mode((600, 600))

    drawing = False
    last_pos = (0, 0)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    color = WHITE

    ds.fill(BLACK)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
                last_pos = event.pos
            if event.type == pygame.MOUSEBUTTONUP:
                drawing = False
            if event.type == pygame.MOUSEMOTION:
                if drawing:
                    pygame.draw.circle(ds, color, last_pos, 40)
                    last_pos = event.pos
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    ds.fill(BLACK)
                if event.key == pygame.K_s:
                    # take screenshot
                    img_name = 'number.jpg'
                    pygame.image.save(ds, img_name)
                    image = compress_image(img_name, False)
                    t = transforms.ToTensor()
                    image = t(image)
                    image = image.view(1, 784)
                    with torch.no_grad():
                        ps = torch.exp(model(image))[0]
                        for i in range(len(ps)):
                            print(f'{i} : {round(ps[i].item(), 4) * 100}%')
                        print(f'Final Prediction: {torch.argmax(ps)}')
        pygame.display.update()

window()
