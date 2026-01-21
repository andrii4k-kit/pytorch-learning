"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import data_set_load, engine, model_builder, utils
from torchvision import transforms

import argparse
# Initialize ArgumentParser
parser = argparse.ArgumentParser(description="Get some hyperparameters.")

parser.add_argument("--train_dir", type=str, help="Directory for training data.", default="data/pizza_steak_sushi/train")
parser.add_argument("--test_dir", type=str, help="Directory for test data.", default="data/pizza_steak_sushi/test")
parser.add_argument("--NUM_EPOCHS", type=int, help="The amount of epochs.", default=5)
parser.add_argument("--BATCH_SIZE", type=int, help="Amount of img in batch.", default=32)
parser.add_argument("--HIDDEN_UNITS", type=int, help="Amount of parameters/neurons per layer.", default=10)
parser.add_argument("--LEARNING_RATE", type=float, help="Learning Rate for the optimizer.", default=0.001)

args = parser.parse_args()


# Setup hyperparameters
NUM_EPOCHS = args.NUM_EPOCHS
BATCH_SIZE = args.BATCH_SIZE
HIDDEN_UNITS = args.HIDDEN_UNITS
LEARNING_RATE = args.LEARNING_RATE

# Setup directories
train_dir = args.train_dir
test_dir = args.test_dir



# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
  transforms.Resize((64, 64)),
  transforms.ToTensor()
])

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_set_load.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# Create model with help from model_builder.py
model = model_builder.TinyVGG(
    input_shape=3,
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names)
).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start training with help from engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir="models",
                 model_name="05_going_modular_script_mode_tinyvgg_model.pth")
