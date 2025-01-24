##-- Import libraries --##
import torch
import torchvision
from torch import nn
import matplotlib.pyplot as plt
from torchvision import transforms
from helper_modules import data_setup, engine, utils
from timeit import default_timer as timer

##-- Setup device agnostic code --##
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PyTorch Version: {torch.__version__}")
print(f"Device: {device}\n")

##-- Define hyperparamters --##
EPOCHS = 20
LEARNING_RATE = 0.001
BATCH_SIZE = 64
RANDOM_STATE = 42


##-- Setup directory paths --##
from pathlib import Path
image_data_path = Path(r"C:\Users\nextt\Desktop\Hawking\eyeDx-research\aug_train_val")
train_dir = image_data_path / "train"
val_dir = image_data_path / "val"


##-- Get a set of pretrained model weights --##
weights_b7 =torchvision.models.EfficientNet_B7_Weights.DEFAULT # 'DEFAULT' = best availabel weight


##-- Get the transforms used to create the pretrained weights
auto_transforms = weights_b7.transforms()

##-- Create DataLoaders using auto_transforms --##
train_dataloader, val_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                              val_dir=val_dir,
                                                                              transform=auto_transforms,
                                                                              batch_size=BATCH_SIZE)

##-- Setting Up Pretrained Model --##
weights_b7 = torchvision.models.EfficientNet_B7_Weights.DEFAULT # "DEFAULT" = get the best available weights
model_b7 = torchvision.models.efficientnet_b7(weights=weights_b7).to(device)

##-- Freeze all the base layers in EffNetB7 --##
for params in model_b7.features.parameters():
    params.requires_grad = False # won't updated the weights
    
    
##-- Update the classifier head of our model to suite our problem --##
torch.manual_seed(RANDOM_STATE)
torch.cuda.manual_seed(RANDOM_STATE)

model_b7.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=2560, # features vector coming in from the forzen layers
              out_features=len(class_names)).to(device))


torch.manual_seed(RANDOM_STATE)
torch.cuda.manual_seed(RANDOM_STATE)

##-- Define Loss & Optimizer --##
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_b7.parameters(), lr=LEARNING_RATE)

##-- Start timer --##
start_time = timer()

##-- Setup training and save the results --##
model_b7_results = engine.train(model=model_b7,
                               train_dataloader=train_dataloader,
                               val_dataloader=val_dataloader,
                               optimizer=optimizer,
                               loss_fn=loss_fn,
                               epochs=EPOCHS,
                               device=device)

##-- End timer --##
end_time = timer()

##- Save the model --##
utils.save_model(model=model_b7,
                 target_dir="./saved_models",
                 model_name=f"efficientnetb7_{LEARNING_RATE}_{EPOCHS}_{BATCH_SIZE}.pth",
                 optimizer=optimizer,
                 epoch=EPOCHS)

print(f"[INFO]: Total training time: {(end_time-start_time)/60:.3f} minutes.")


##-- Plot loss curves --##
epochs = [i for i in range(1, EPOCHS+1)]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, model_b7["train_loss"], color="blue", label="train_loss")
plt.plot(epochs, model_b7["val_loss"], color="red", label="val_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(epochs, model_b7["train_acc"], color="blue", label="train_acc")
plt.plot(epochs, model_b7["val_acc"], color="green", label="val_acc")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.suptitle("EfficientnetV2_M Results")
plt.savefig(f"plot_v2_m_{EPOCHS}_{LEARNING_RATE}_{BATCH_SIZE}.png")
plt.legend()
plt.show()


