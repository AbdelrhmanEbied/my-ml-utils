import torch 
def accuracy_fn(y_pred, y_true):
    predicted_classes = torch.argmax(y_pred, dim=1)
    correct = (predicted_classes == y_true).sum().item()
    return correct / len(y_true)