'''RAW DATA FOR TRAINING WILL BE LOADED HERE'''

import pandas as pd
import json

# Loading The Verdict
path = r"C:\Users\HP\Desktop\AI-ML\machineLearning\data_sets\the-verdict.txt"

with open(path, "r", encoding='utf-8') as f:
    verdict = f.read()


# Loading data for classification finetuning
path = r"C:\Users\HP\Desktop\AI-ML\machineLearning\data_sets\sms+spam+collection\SMSSpamCollection.tsv"
spam_df  = pd.read_csv(path, sep='\t', header=None, names=['label', 'text','extra'])
spam_df["text"] = spam_df["text"] + " " + spam_df["extra"].fillna("")
spam_df = spam_df.drop(columns=["extra"])

def create_balanced_dataset(df):
    num_spam = df[df['label'] == 'spam'].shape[0]
    ham_subset = df[df['label'] == 'ham'].sample(n=num_spam, random_state=42)
    balanced_df = pd.concat([ham_subset, df[df['label'] == 'spam']])
    balanced_df["label"] = balanced_df["label"].map({"ham": 0, "spam": 1})
    return balanced_df.sample(frac=1, random_state=123).reset_index(drop=True)

created_df = create_balanced_dataset(spam_df)

# Loading data for instruction finetuning
path = r"C:\Users\HP\Desktop\AI-ML\machineLearning\data_sets\instruction_dataset\instruction-data.json"
with open(path, "r", encoding="utf-8") as file:
    instruction_data = json.load(file)
