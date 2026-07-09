import torch
from utils.load_pretrain_gpt import load_model

#selecting device for the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#loading the required model and its configurations
gpt, NEW_CONFIG = load_model(model_size="124M", model_name="gpt2-small (124M)")

# sending model to GPU
gpt.to(device);

# performing freezing of the model parameters to avoid unnecessary gradient computations
for param in gpt.parameters():
    param.requires_grad = False

gpt.out_head = torch.nn.Linear(NEW_CONFIG["emb_dim"], 2, bias=True).to(device)  # Adjust output layer for binary classification

# making transformer model trainable for fine-tuning
for param in gpt.trf_blocks[-1].parameters():
    param.requires_grad = True

for param in gpt.final_norm.parameters():
    param.requires_grad = True
