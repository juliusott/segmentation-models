from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from data_utils import ValDataSegmentation, TrainDataSegmentation, Normalize
from models.SegNet import SegNet
from models.SegResNet import SegResNet
from models.autoencoder import Autoencoder
from models.deeplabv3 import DeepLab
from training import *
from validation import *

# GLOBALS
NUM_EPOCHS = 10
BATCH_SIZE = 1  # large Batch size leads to a memory issue

# choose model
models = [DeepLab(), SegResNet(), SegNet(), Autoencoder()]
model = models[0]  # 0=DeepLab, 1=SegResNet, 2= SegNet, 3=Autoencoder

# import dataset
transform = transforms.Compose([Normalize()])
train_dataset = TrainDataSegmentation("./images", transform=transform)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_dataset = ValDataSegmentation("./images", transform=transforms)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# training
# set autoencoder flag if necessary!
train(num_epochs=NUM_EPOCHS, train_dataloader=train_dataloader, model=model, input_path="./trained_models/segresnet.pt")
validation(val_dataset, val_dataloader, model, BATCH_SIZE, path_to_model="./trained_models/segresnet.pt")
