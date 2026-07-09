import time
import torch
from finetune.classifing_model.upgrade_model import gpt
from finetune.classifing_model.train import train_classifier
from finetune.classifing_model.accuracy_calculation import calc_accuracy
from finetune.classifing_model.dataloader import (train_loader, val_loader, test_loader)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(123)

optimizer = torch.optim.AdamW(gpt.parameters(), lr=5e-5, weight_decay=0.1)

epochs = 5

start_time = time.time()

train_losses, val_losses, train_accs, val_accs, example_seen = train_classifier(
    model=gpt, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer,
    device=device, num_epochs=epochs, eval_freq=50, eval_iter=5 
)

ent_time = time.time()
print(f"Training Completed in {(ent_time-start_time)/60 :.2f} minutes.")

train_accuracy = calc_accuracy(train_loader, gpt, device)
val_accuracy = calc_accuracy(val_loader, gpt, device)
test_accuracy = calc_accuracy(test_loader, gpt, device)
print(f"Train Accuracy: {train_accuracy*100:.4f}")
print(f"Val Accuracy: {val_accuracy*100:.4f}")
print(f"Test Accuracy: {test_accuracy*100:.4f}")

# saving te model
torch.save(gpt.state_dict(), r"C:\Users\HP\Desktop\AI-ML\machineLearning\llm\models\finetuned_model\classifier_model.pth")
print("Classifier Model is saved successfully!")
