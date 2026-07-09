import torch
import tiktoken
from finetune.classifing_model.upgrade_model import gpt
from finetune.classifing_model.dataloader import train_dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def classify_text(text, model, tokenizer, device, max_length=None, pd_t_id=50256):
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]
    input_ids = input_ids[:min(max_length, supported_context_length)]
    input_ids += [pd_t_id]*(max_length - len(input_ids))
    input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_ids)[:, -1, :]
    predcited_label = torch.argmax(logits, dim=-1)
    return "Spam" if predcited_label.item()==1 else "Not spam"

print("Loading Finetuned Model...")

gpt.load_state_dict(
    torch.load(r"C:\Users\HP\Desktop\AI-ML\machineLearning\llm\models\finetuned_model\classifier_model.pth", map_location=device, weights_only=True)
)

sample = (
    "Check the below message and classify as 'sapm' or 'not spam'."
    "You ar a winner you have been selected to receive $1000 cash or a $2000 award. Click the link below to claim it."
    )

sample2 = "hey , there buddy!"

tokenizer = tiktoken.get_encoding('gpt2')
print(f"\nPrediction for:- \n`{sample}`")
print(":-",classify_text(text=sample, model=gpt, tokenizer=tokenizer, device=device, max_length=train_dataset.max_length))
print(f"\nPrediction for:- \n`{sample2}`")
print(":-",classify_text(text=sample2, model=gpt, tokenizer=tokenizer, device=device, max_length=train_dataset.max_length))
