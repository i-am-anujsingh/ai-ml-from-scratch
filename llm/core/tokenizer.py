#Creating tokenizer

import string
import re

# loading the text data
with open("the-verdict.txt", "r") as f:
    raw_text = (f.read())

punctuations = "(--|[" + string.punctuation + "]|\s)"
preprocessed = re.split(rf'{punctuations}', raw_text)
preprocessed = [item for item in preprocessed if item.strip()]
all_words = sorted(set(preprocessed))

# added <|UNK|> token for the words that are not in the vocabulary
all_words.insert(0,"<|UNK|>")
vocab_size = (len(all_words))

vocabs = {token:intId for intId, token in enumerate(all_words)} # Dictionary which contains words and their ids

class SimpleTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab # regular vocab
        self.int_to_str= {i:s for s,i in vocab.items()} #inverted vocab
    
    def encode(self, text):
        punctuations = "(--|[" + string.punctuation + "]|\s)"
        preprocessed = re.split(rf'{punctuations}', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] if s in self.str_to_int else ( 0 ) for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        punctuations = "\s+([" + string.punctuation + "])"
        text = re.sub(rf'{punctuations}', r'\1', text)
        return text
    

# sample text encoding amd decoding with SimpleTokenizer class

sample_text = "Hello, Mrs. Stroud!" # sample text for encoding

s = SimpleTokenizer(vocab=vocabs)

token_ids = s.encode(text=sample_text) 

print(token_ids)

decoded_text = s.decode(ids=token_ids) #using the same token ids that we get above for decoding

print(decoded_text)