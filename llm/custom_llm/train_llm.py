import torch
import tiktoken
from utils.raw_data import verdict
from core.dataloader import create_dataloader
from gpt_architecture.architecture import GPTModel, GPT_CONFIG_124M
from custom_llm.train import train_model

device = ("cuda" if torch.cuda.is_available() else "cpu")

llm = GPTModel(GPT_CONFIG_124M)
llm.to(device)

tokenizer = tiktoken.get_encoding("gpt2")

train_ratio = 0.90
split_idx =int(train_ratio*len(verdict))
train_data = verdict[:split_idx]
val_data = verdict[split_idx:]

train_loader = create_dataloader(train_data,
                                 batch_size=2,
                                 max_length=GPT_CONFIG_124M["context_length"],
                                 stride=GPT_CONFIG_124M["context_length"],
                                 drop_last=True,
                                 shuffle=True,
                                 num_workers=0)

val_loader = create_dataloader(val_data,
                                 batch_size=2,
                                 max_length=GPT_CONFIG_124M["context_length"],
                                 stride=GPT_CONFIG_124M["context_length"],
                                 drop_last=False,
                                 shuffle=False,
                                 num_workers=0)

epoch = 10

llm.train()

optimizer = torch.optim.AdamW(llm.parameters(), lr = 0.0004, weight_decay = 0.1)

train_losses, val_losses, tokens_seen = train_model(model=llm,
                                                    train_loader=train_loader,
                                                    val_loader=val_loader,
                                                    optimizer=optimizer,
                                                    device=device,
                                                    num_epochs=epoch,
                                                    eval_freq=5, eval_iter=5,
                                                    start_context="Every effort moves you",
                                                    tokenizer=tokenizer
                                                    )

# saving te model
torch.save(llm.state_dict(), r"C:\Users\HP\Desktop\AI-ML\machineLearning\llm\models\base_llm_model\llm_model_re.pth")
print("LLM Model is saved successfully!")