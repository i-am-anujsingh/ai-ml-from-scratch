import torch
import tiktoken
from utils.load_pretrain_gpt import load_model
from gpt_architecture.architecture import modified_gts
from core.text_token_converter import text_to_tokens, tokens_to_text
from finetune.instruction_based.dataloader import (test_data, format_input)

tokenizer = tiktoken.get_encoding("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(123)

#loading the model and its configurations
gpt, NEW_CONFIG = load_model(model_size="355M", model_name="gpt2-medium (355M)")

# sending model to GPU
gpt.to(device);

print("Loading Finetuned Model...\n")
gpt.load_state_dict(
    torch.load(r"C:\Users\HP\Desktop\AI-ML\machineLearning\llm\models\finetuned_model\instruction_model_v1.pth", map_location=device, weights_only=True)
)

gpt.eval()

for ent in test_data[:3]:
    input_text = format_input(ent)
    token_ids = modified_gts(
        model=gpt,
        idx=text_to_tokens(input_text, tokenizer=tokenizer).to(device),
        max_new_tokens=256,
        context_size=NEW_CONFIG['context_length'],
        # eos_id=50256,
    )
    generated_text = tokens_to_text(token_ids, tokenizer)
    start = generated_text.find("### Response:")
    if start != -1:
        response = generated_text[start + len("### Response:"):]
        # Stop at EOS if present
        response = response.split("<|endoftext|>")[0].strip()

    print(input_text)
    print(f"\nCorrect response:-\n{ent['output']}")
    print(f"\nModel response:-\n{response.strip()}")
    print("\n-------------------------------------------------------------------")
    print("-------------------------------------------------------------------\n")