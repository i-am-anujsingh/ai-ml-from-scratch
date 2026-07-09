import torch

def calc_accuracy(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions = 0
    num_examples = 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (inputs, targets) in enumerate(data_loader):
        if i < num_batches:
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                outputs = outputs[:, -1, :]
                predicted_classes = torch.argmax(outputs, dim=-1)
                num_examples += predicted_classes.shape[0]
                correct_predictions += (predicted_classes == targets).sum().item()
        else:
            break

    return correct_predictions / num_examples if num_examples > 0 else 0.0