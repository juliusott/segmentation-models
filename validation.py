import torch
import torch.nn as nn
import torch.optim as optim

from helper import *


def validation(val_dataset, val_dataloader, model, BATCH_SIZE, path_to_model, print_image = True):
    model.load_state_dict(torch.load(path_to_trained_model))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # print([module for module in model.modules()])
    criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    test_scores = []

    # validation loop
    running_loss = 0.0
    test_scores_val = []
    for i, data in enumerate(val_dataloader, 0):
        model.eval()
        inputs, mask = data
        inputs = inputs.to(device)
        mask = mask.to(device)
        output = model(inputs)
        loss = criterion(output, mask)
        running_loss += loss.item()
        _, preds = torch.max(output, 1)
        targets_mask = mask >= 0
        test_scores_val.append(np.mean((preds == mask)[targets_mask].data.cpu().numpy()))
        if i % 5 == 0 and i > 0:
            print("val_loss {} and accuracy {} [{}/{}]".format(running_loss / i, np.mean(test_scores_val), i,
                                                               len(val_dataset) / BATCH_SIZE))
            test_scores_val = []
            running_loss = 0.0

    if print_image:
        rgb = val_dataset[3][0]
        target = val_dataset[3][1]
        output = model(rgb.reshape((1, 3, 360, 640)).to(device))
        _, preds = torch.max(output, 1)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(rgb.numpy().transpose(1, 2, 0))
        ax1.set_title("input image")
        ax2.imshow(label_img_to_rgb(preds.cpu().numpy()))
        ax2.set_title("predicted image")
        ax3.imshow(label_img_to_rgb(target.numpy()))
        ax3.set_title("target image")
        plt.show()
