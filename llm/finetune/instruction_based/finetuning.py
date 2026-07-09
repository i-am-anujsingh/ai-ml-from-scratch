import torch
import time
import tiktoken
from custom_llm.train import train_model
from utils.load_pretrain_gpt import load_model
from finetune.instruction_based.dataloader import (train_loader,
val_loader, val_data, format_input)

tokenizer = tiktoken.get_encoding("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(123)

gpt, NEW_CONFIG = load_model(model_size="355M", model_name="gpt2-medium (355M)")

# sending model to GPU
gpt.to(device);

#time when training started
start_time = time.time()

optimizer = torch.optim.AdamW(gpt.parameters(), lr = 0.00005, weight_decay = 0.1)

epoch = 1

gpt.train()

train_losses, val_losses, tokens_seen = train_model(model=gpt,
                                                    train_loader=train_loader,
                                                    val_loader=val_loader,
                                                    optimizer=optimizer,
                                                    device=device,
                                                    num_epochs=epoch,
                                                    eval_freq=5, eval_iter=5,
                                                    start_context=format_input(val_data[0]),
                                                    tokenizer=tokenizer
                                                    )


#time when training ended
end_time = time.time()
print(f"Training Completed in {(end_time-start_time)/60 :.2f} minutes.")

# saving te model
torch.save(gpt.state_dict(), r"C:\Users\HP\Desktop\AI-ML\machineLearning\llm\models\finetuned_model\instruction_model.pth")
print("Instruction Model is saved successfully!")
