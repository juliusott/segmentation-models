import os

import torch
import torch.nn as nn
import torch.optim as optim

from helper import *


def train(num_epochs, train_dataloader, model, input_path="./trained_models",
          output_path=None, autoencoder=False):
    # setup gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # loading model
    if os.path.isfile(input_path):
        model.load_state_dict(torch.load("segresnet_nn.pt"))
        print("loaded model checkpoint")
    model.to(device)

    # print([module for module in model.modules()])
    if autoencoder:
        criterion = nn.MSELoss(reduction='mean')
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    test_scores = []

    # train loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            model.train()
            inputs, mask = data
            inputs = inputs.to(device)
            mask = mask.to(device)
            # print( "max target: {}, min target: {}".format(torch.max(mask), torch.min(mask)))
            optimizer.zero_grad()

            output = model(inputs)
            loss = criterion(output, mask)
            loss.backward()
            optimizer.step()

            if autoencoder:
                running_loss += loss.item()

                if i % 5 == 0 and i > 0:
                    print("loss {} and accuracy {} in iteration {} and epoch {}".format(running_loss / i,
                                                                                        np.mean(test_scores), i, epoch))
                    running_loss = 0.0
            else:
                _, preds = torch.max(output, 1)
                targets_mask = mask >= 0
                test_scores.append(np.mean((preds == mask)[targets_mask].data.cpu().numpy()))
                running_loss += loss.item()

                if i % 5 == 0 and i > 0:
                    print("loss {} and accuracy {} in iteration {} and epoch {}".format(running_loss / i,
                                                                                        np.mean(test_scores), i, epoch))
                    running_loss = 0.0
                    test_scores = []
    if output_path:
        torch.save(model.state_dict(), output_path)
    else:  # overwrite the input model
        torch.save(model.state_dict(), input_path)
