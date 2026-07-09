''' HOW TO LOAD THE SAVED MODEL '''

import torch
import tiktoken
from core.text_token_converter import text_to_tokens, tokens_to_text
from gpt_architecture.architecture import GPTModel, GPT_CONFIG_124M, modified_gts

tokenizer = tiktoken.get_encoding("gpt2")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

new_model = GPTModel(GPT_CONFIG_124M)
new_model.to(device)

# loading the model
new_model.load_state_dict(
    torch.load(r"C:\Users\HP\Desktop\AI-ML\machineLearning\llm\models\base_llm_model\llm_model_re.pth", map_location=device, weights_only=True)
)

new_model.eval()

text = "Hi, What is your"

encoded = text_to_tokens(text=text, tokenizer=tokenizer)

token_ids = modified_gts(
            model=new_model, idx=(encoded).to(device),
            max_new_tokens=20, context_size = GPT_CONFIG_124M["context_length"],
            top_k=25, temperature=1.4
        )

print(f"OUTPUT:-\n{tokens_to_text(token_ids, tokenizer)}")