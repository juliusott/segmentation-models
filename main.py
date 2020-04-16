import torch
import torch.nn as nn
from torchvision import transforms
from segmenation_nn import SegmentationNN
from data_utils import ValDataSegmentation, TrainDataSegmentation, Normalize
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
#GLOBALS
NUM_EPOCHS = 10
BATCH_SIZE = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transforms = transforms.Compose([Normalize()])
dataset = TrainDataSegmentation("./images", transform=transforms)
dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

model = SegmentationNN()
print([module for module in model.modules()])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
test_scores = []
for eopch in range(NUM_EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(dataloader,0):
        image, label = data
        optimizer.zero_grad()

        output = model(image)["out"]
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        _, preds = torch.max(outputs, 1)
        targets_mask = targets >= 0
        test_scores.append(np.mean((preds == targets)[targets_mask].data.cpu().numpy()))
        running_loss += loss.item()

        if i %100 == 0:
            print("loss {} and accuracy {} in epoch {}".format(running_loss/i, np.mean(test_scores), i))
            running_loss = 0.0
            test_scores = []
