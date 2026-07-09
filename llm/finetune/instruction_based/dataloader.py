import torch
import tiktoken
from utils.raw_data import instruction_data
from torch.utils.data import DataLoader, Dataset

torch.manual_seed(123)

tokenizer = tiktoken.get_encoding("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_portion = int(len(instruction_data)*0.85)
test_portion = int(len(instruction_data)*0.1)
val_portion = len(instruction_data) - train_portion - test_portion

train_data = instruction_data[:train_portion]
test_data = instruction_data[train_portion: train_portion + test_portion]
val_data = instruction_data[train_portion + test_portion:]

def format_input(entry):
    inst_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry['input'] else ""

    response_text = f"\n\n### Response:\n{entry['output']}"

    return inst_text + input_text + response_text

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_text = []
        for ent in data:
            instruction = format_input(ent)
            self.encoded_text.append(tokenizer.encode(instruction))

    def __getitem__(self, index):
        return self.encoded_text[index]
    
    def __len__(self):
        return len(self.data)
    

def custom_collate_fn(batch, pad_token_id=50256, ignore_idx=-100,
                   allowed_max_length=None, device="cuda"):

    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs
    inputs_lst, targets_list = [], []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to batch_max_length
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
 
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets==pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_idx

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_list.append(targets)

    # Convert list of inputs to tensor and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_list).to(device)
    return inputs_tensor, targets_tensor


from functools import partial

customized_collate_fn = partial(
    custom_collate_fn, device=device, allowed_max_length = 1024
)

num_workers = 0
batch_size = 8

train_dataset = InstructionDataset(train_data, tokenizer=tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True, drop_last=True,
    num_workers=num_workers
)

test_dataset = InstructionDataset(test_data, tokenizer=tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False, drop_last=False,
    num_workers=num_workers
)

val_dataset = InstructionDataset(val_data, tokenizer=tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False, drop_last=False,
    num_workers=num_workers
)

