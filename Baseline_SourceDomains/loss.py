import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

class DTILoss(nn.Module):
    """Custom loss function for DTI prediction"""
    
    def __init__(self, loss_type='bce', alpha=0.5, beta=0.5):
        super().__init__()
        self.loss_type = loss_type
        self.alpha = alpha
        self.beta = beta
        
        if loss_type == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_type == 'focal':
            self.criterion = FocalLoss(alpha=alpha, gamma=2.0)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, predictions, targets):
        return self.criterion(predictions, targets)

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class CombinedLoss(nn.Module):
    """Combined loss function with multiple components"""
    
    def __init__(self, bce_weight=1.0, focal_weight=0.0, consistency_weight=0.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.consistency_weight = consistency_weight
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        if focal_weight > 0:
            self.focal_loss = FocalLoss()
    
    def forward(self, predictions, targets, predictions_aux=None):
        total_loss = 0.0
        
        # BCE Loss
        if self.bce_weight > 0:
            bce = self.bce_loss(predictions, targets)
            total_loss += self.bce_weight * bce
        
        # Focal Loss
        if self.focal_weight > 0:
            focal = self.focal_loss(predictions, targets)
            total_loss += self.focal_weight * focal
        
        # Consistency Loss (if auxiliary predictions provided)
        if self.consistency_weight > 0 and predictions_aux is not None:
            consistency = F.mse_loss(predictions, predictions_aux)
            total_loss += self.consistency_weight * consistency
        
        return total_loss

def calculate_metrics(predictions, targets, threshold=0.5):
    """Calculate various evaluation metrics"""
    
    # Convert to numpy for sklearn metrics
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()
    
    # Apply sigmoid if needed
    if predictions.max() > 1.0 or predictions.min() < 0.0:
        predictions = 1 / (1 + np.exp(-predictions))
    
    # Binary predictions
    binary_preds = (predictions > threshold).astype(int)
    
    # Calculate metrics
    metrics = {}
    
    try:
        metrics['accuracy'] = accuracy_score(targets, binary_preds)
    except:
        metrics['accuracy'] = 0.0
    
    try:
        metrics['auc'] = roc_auc_score(targets, predictions)
    except:
        metrics['auc'] = 0.0
    
    try:
        metrics['f1'] = f1_score(targets, binary_preds)
    except:
        metrics['f1'] = 0.0
    
    try:
        metrics['precision'] = precision_score(targets, binary_preds)
    except:
        metrics['precision'] = 0.0
    
    try:
        metrics['recall'] = recall_score(targets, binary_preds)
    except:
        metrics['recall'] = 0.0
    
    return metrics

def evaluate_model(model, data_loader, device, criterion=None):
    """Evaluate model on a data loader"""
    
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in data_loader:
            # Move batch to device
            batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
            
            # Forward pass
            predictions = model(batch)
            targets = batch['labels'].float()
            
            # Calculate loss
            if criterion is not None:
                loss = criterion(predictions, targets)
                total_loss += loss.item()
            
            # Store predictions and targets
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate metrics
    metrics = calculate_metrics(all_predictions, all_targets)
    
    if criterion is not None:
        metrics['loss'] = total_loss / len(data_loader)
    
    return metrics, all_predictions, all_targets
