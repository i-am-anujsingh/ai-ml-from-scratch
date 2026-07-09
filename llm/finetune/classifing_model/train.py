import torch
from finetune.classifing_model.accuracy_calculation import calc_accuracy
from finetune.classifing_model.loss_claculation import calc_loss_loader, calc_loss

def train_classifier(model, train_loader, val_loader, optimizer, device,
                num_epochs, eval_freq, eval_iter):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    example_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            example_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
        
        train_accuracy = calc_accuracy(data_loader=train_loader, model=model, device=device, num_batches=eval_iter)
        val_accuracy = calc_accuracy(data_loader=val_loader, model=model, device=device, num_batches=eval_iter)
        print(f"Training Accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation Accuracy: {val_accuracy*100:.2f}% | ")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, example_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss
