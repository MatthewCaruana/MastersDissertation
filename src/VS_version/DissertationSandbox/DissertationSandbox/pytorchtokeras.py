import torch
from torchsummary import summary

model = torch.load("EntityDetection//Mohammed//nn//saved_checkpoints//lstm//id1_best_model.pt")

summary(model)