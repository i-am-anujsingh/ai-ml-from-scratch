import torch
import tiktoken
from utils.raw_data import created_df
from torch.utils.data import DataLoader, Dataset

def random_split(df, train_frac, val_frac):
    train_df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    train_end = int(len(train_df) * train_frac)
    val_end = train_end + int(len(train_df) * val_frac)
    train_set = train_df[:train_end]
    val_set = train_df[train_end:val_end]
    test_set = train_df[val_end:]
    return train_set, val_set, test_set

class SampleDataset(Dataset):
    def __init__(self, df, max_length=None, pad_token_id=50256):
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.df = df
        self.encoded_text = [self.tokenizer.encode(text) for text in df['text']]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
        self.encoded_text = [enc[:self.max_length] for enc in self.encoded_text]
        self.encoded_text = [enc + [pad_token_id] * (self.max_length - len(enc)) for enc in self.encoded_text]

    def __getitem__(self, idx):
        return torch.tensor(self.encoded_text[idx]), torch.tensor(self.df['label'].iloc[idx])
    
    def __len__(self):
        return len(self.df)
    
    def _longest_encoded_length(self):
        return max(len(enc) for enc in self.encoded_text)

random_split_df = random_split(created_df, train_frac=0.7, val_frac=0.1)

train_dataset = SampleDataset(random_split_df[0], max_length=None)
val_dataset = SampleDataset(random_split_df[1], max_length=train_dataset.max_length)
test_dataset = SampleDataset(random_split_df[2], max_length=train_dataset.max_length)

num_workers = 0
batch_size = 8

torch.manual_seed(123)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False)