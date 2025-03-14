import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

def evaluate_model(model, dataloader, device, num_classes):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    
    one_hot_labels = np.eye(num_classes)[all_labels]
    try:
        roc_auc = roc_auc_score(one_hot_labels, all_probs, multi_class='ovr')
    except:
        roc_auc = None
    
    print(f"Validation Accuracy: {acc:.4f}, ROC-AUC: {roc_auc if roc_auc is not None else 0.0:.4f}")
    return acc, roc_auc
