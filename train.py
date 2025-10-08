import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import json
import numpy as np
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============== Label Smoothing Loss ==============

class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy loss"""
    def __init__(self, num_classes, smoothing=0.1, weight=None):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.weight = weight
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        
        if self.weight is not None:
            pred = pred * self.weight.unsqueeze(0)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

# ============== Model Definition ==============

class CrossAttention(nn.Module):
    """Cross-attention mechanism for contrastive interaction"""
    
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len_q, _ = query.size()
        seq_len_k = key.size(1)
        
        # Project and reshape to multi-head format
        Q = self.query_proj(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_proj(key).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_proj(value).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        # Apply softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Get context vectors
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.hidden_size)
        output = self.out_proj(context)
        
        return output

class ContrastiveInteractionLayer(nn.Module):
    """Bidirectional cross-attention for claim-evidence interaction"""
    
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        # Bidirectional cross-attention
        self.claim_to_evidence = CrossAttention(hidden_size, num_heads, dropout)
        self.evidence_to_claim = CrossAttention(hidden_size, num_heads, dropout)
        
        # Layer normalization
        self.ln_claim = nn.LayerNorm(hidden_size)
        self.ln_evidence = nn.LayerNorm(hidden_size)
        
        # Feed-forward networks
        self.ffn_claim = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
        self.ffn_evidence = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
    def forward(self, claim_hidden: torch.Tensor, evidence_hidden: torch.Tensor,
                claim_mask: Optional[torch.Tensor] = None,
                evidence_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Claim attends to evidence
        verified_claim = self.claim_to_evidence(
            query=claim_hidden,
            key=evidence_hidden,
            value=evidence_hidden,
            attention_mask=evidence_mask
        )
        
        # Evidence attends to claim
        relevant_evidence = self.evidence_to_claim(
            query=evidence_hidden,
            key=claim_hidden,
            value=claim_hidden,
            attention_mask=claim_mask
        )
        
        # Add & Norm
        verified_claim = self.ln_claim(claim_hidden + verified_claim)
        relevant_evidence = self.ln_evidence(evidence_hidden + relevant_evidence)
        
        # Feed-forward
        verified_claim = verified_claim + self.ffn_claim(verified_claim)
        relevant_evidence = relevant_evidence + self.ffn_evidence(relevant_evidence)
        
        return verified_claim, relevant_evidence

class DifferenceReasoningModule(nn.Module):
    """Module to capture differences between original and interacted representations"""
    
    def __init__(self, hidden_size: int, reasoning_hidden_size: int = 512, dropout: float = 0.1):
        super().__init__()
        
        # Different methods to capture semantic differences
        self.difference_methods = nn.ModuleDict({
            'subtract': nn.Linear(hidden_size * 2, reasoning_hidden_size),
            'hadamard': nn.Linear(hidden_size * 2, reasoning_hidden_size),
            'concat': nn.Linear(hidden_size * 4, reasoning_hidden_size),
        })
        
        # Combine all difference representations
        # 3 methods + 2 cosine similarities = 5 components
        self.reasoning_layers = nn.Sequential(
            nn.Linear(reasoning_hidden_size * 5, reasoning_hidden_size * 2),
            nn.LayerNorm(reasoning_hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(reasoning_hidden_size * 2, reasoning_hidden_size),
            nn.LayerNorm(reasoning_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.hidden_size = hidden_size
        self.reasoning_hidden_size = reasoning_hidden_size
        
    def forward(self, original_claim: torch.Tensor, verified_claim: torch.Tensor,
                original_evidence: torch.Tensor, relevant_evidence: torch.Tensor) -> torch.Tensor:
        
        differences = []
        
        # Method 1: Subtraction (explicit difference)
        claim_diff = verified_claim - original_claim
        evidence_diff = relevant_evidence - original_evidence
        diff_subtract = torch.cat([claim_diff, evidence_diff], dim=-1)
        differences.append(self.difference_methods['subtract'](diff_subtract))
        
        # Method 2: Hadamard product (element-wise interaction)
        claim_hadamard = original_claim * verified_claim
        evidence_hadamard = original_evidence * relevant_evidence
        diff_hadamard = torch.cat([claim_hadamard, evidence_hadamard], dim=-1)
        differences.append(self.difference_methods['hadamard'](diff_hadamard))
        
        # Method 3: Full concatenation (complete information)
        full_concat = torch.cat([
            original_claim, verified_claim,
            original_evidence, relevant_evidence
        ], dim=-1)
        differences.append(self.difference_methods['concat'](full_concat))
        
        # Method 4: Cosine similarity (directional difference)
        claim_cosine = F.cosine_similarity(original_claim, verified_claim, dim=-1, eps=1e-8).unsqueeze(-1)
        evidence_cosine = F.cosine_similarity(original_evidence, relevant_evidence, dim=-1, eps=1e-8).unsqueeze(-1)
        
        # Expand cosine similarities to reasoning_hidden_size
        claim_cosine_expanded = claim_cosine.expand(-1, self.reasoning_hidden_size)
        evidence_cosine_expanded = evidence_cosine.expand(-1, self.reasoning_hidden_size)
        differences.append(claim_cosine_expanded)
        differences.append(evidence_cosine_expanded)
        
        # Combine all difference representations
        combined_differences = torch.cat(differences, dim=-1)
        
        # Apply reasoning layers
        reasoning_output = self.reasoning_layers(combined_differences)
        
        return reasoning_output

class HDCR(nn.Module):
    """Health Distortion Detector with Contrastive Reasoning"""
    
    def __init__(self, config):
        super().__init__()
        
        # Initialize encoders based on config
        self.claim_encoder = AutoModel.from_pretrained(config['claim_encoder_name'])
        self.evidence_encoder = AutoModel.from_pretrained(config['evidence_encoder_name'])
        self.num_labels = config['num_labels']
        
        # Get hidden sizes from encoder configurations
        claim_hidden_size = self.claim_encoder.config.hidden_size
        evidence_hidden_size = self.evidence_encoder.config.hidden_size
        hidden_size = config['hidden_size']
        
        # Projection layers to unify dimensions
        self.claim_projection = nn.Linear(claim_hidden_size, hidden_size) if claim_hidden_size != hidden_size else nn.Identity()
        self.evidence_projection = nn.Linear(evidence_hidden_size, hidden_size) if evidence_hidden_size != hidden_size else nn.Identity()
        
        # Core modules
        self.interaction_layer = ContrastiveInteractionLayer(
            hidden_size, 
            config['num_attention_heads'], 
            config['dropout']
        )
        
        self.reasoning_module = DifferenceReasoningModule(
            hidden_size, 
            config['reasoning_hidden_size'], 
            config['dropout']
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config['reasoning_hidden_size'], config['reasoning_hidden_size'] // 2),
            nn.LayerNorm(config['reasoning_hidden_size'] // 2),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['reasoning_hidden_size'] // 2, config['num_labels'])
        )
        
        # Pooling method: cls, mean, or max
        self.pooling_method = config.get('pooling_method', 'cls')
        
        # Layer normalization
        self.ln_claim = nn.LayerNorm(hidden_size)
        self.ln_evidence = nn.LayerNorm(hidden_size)
        
    def forward(self, claim_input_ids, claim_attention_mask, 
                evidence_input_ids, evidence_attention_mask, labels=None):
        
        # Encode claim and evidence
        claim_outputs = self.claim_encoder(
            input_ids=claim_input_ids,
            attention_mask=claim_attention_mask
        )
        evidence_outputs = self.evidence_encoder(
            input_ids=evidence_input_ids,
            attention_mask=evidence_attention_mask
        )
        
        # Project and normalize hidden states
        claim_hidden = self.claim_projection(claim_outputs.last_hidden_state)
        evidence_hidden = self.evidence_projection(evidence_outputs.last_hidden_state)
        
        claim_hidden = self.ln_claim(claim_hidden)
        evidence_hidden = self.ln_evidence(evidence_hidden)
        
        # Get pooled representations based on pooling method
        if self.pooling_method == 'cls':
            # Use [CLS] token representation
            original_claim = claim_hidden[:, 0, :]
            original_evidence = evidence_hidden[:, 0, :]
        elif self.pooling_method == 'mean':
            # Mean pooling with attention mask
            claim_mask_expanded = claim_attention_mask.unsqueeze(-1).expand(claim_hidden.size()).float()
            claim_sum = torch.sum(claim_hidden * claim_mask_expanded, 1)
            claim_mean = claim_sum / claim_mask_expanded.sum(1).clamp(min=1e-9)
            original_claim = claim_mean
            
            evidence_mask_expanded = evidence_attention_mask.unsqueeze(-1).expand(evidence_hidden.size()).float()
            evidence_sum = torch.sum(evidence_hidden * evidence_mask_expanded, 1)
            evidence_mean = evidence_sum / evidence_mask_expanded.sum(1).clamp(min=1e-9)
            original_evidence = evidence_mean
        else:  # max pooling
            # Max pooling over sequence
            claim_hidden.masked_fill_(~claim_attention_mask.unsqueeze(-1).bool(), -1e9)
            original_claim = torch.max(claim_hidden, dim=1)[0]
            
            evidence_hidden.masked_fill_(~evidence_attention_mask.unsqueeze(-1).bool(), -1e9)
            original_evidence = torch.max(evidence_hidden, dim=1)[0]
        
        # Contrastive interaction between claim and evidence
        verified_claim, relevant_evidence = self.interaction_layer(
            claim_hidden, evidence_hidden,
            claim_attention_mask, evidence_attention_mask
        )
        
        # Pool interacted representations
        verified_claim_pooled = verified_claim[:, 0, :]
        relevant_evidence_pooled = relevant_evidence[:, 0, :]
        
        # Difference reasoning
        reasoning_output = self.reasoning_module(
            original_claim, verified_claim_pooled,
            original_evidence, relevant_evidence_pooled
        )
        
        # Classification
        logits = self.classifier(reasoning_output)
        
        outputs = {"logits": logits}
        
        # Calculate loss if labels are provided
        if labels is not None:
            if hasattr(self, 'class_weights') and self.class_weights is not None:
                if hasattr(self, 'label_smoothing') and self.label_smoothing > 0:
                    loss_fct = LabelSmoothingCrossEntropy(
                        num_classes=self.num_labels,
                        smoothing=self.label_smoothing,
                        weight=self.class_weights
                    )
                else:
                    loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            else:
                if hasattr(self, 'label_smoothing') and self.label_smoothing > 0:
                    loss_fct = LabelSmoothingCrossEntropy(
                        num_classes=self.num_labels,
                        smoothing=self.label_smoothing
                    )
                else:
                    loss_fct = nn.CrossEntropyLoss()
            
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs["loss"] = loss
        
        return outputs

# ============== Dataset ==============

class MedicalMisinformationDataset(Dataset):
    """Dataset for medical misinformation detection with claim-evidence pairs"""
    
    def __init__(self, data_path, claim_tokenizer, evidence_tokenizer, config):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.claim_tokenizer = claim_tokenizer
        self.evidence_tokenizer = evidence_tokenizer
        self.max_claim_length = config['max_claim_length']
        self.max_evidence_length = config['max_evidence_length']
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize claim
        claim_encoding = self.claim_tokenizer(
            item['claim'],
            truncation=True,
            padding='max_length',
            max_length=self.max_claim_length,
            return_tensors='pt'
        )
        
        # Tokenize evidence
        evidence_encoding = self.evidence_tokenizer(
            item['document'],
            truncation=True,
            padding='max_length',
            max_length=self.max_evidence_length,
            return_tensors='pt'
        )
        
        return {
            'id': item['id'],
            'claim_input_ids': claim_encoding['input_ids'].squeeze(0),
            'claim_attention_mask': claim_encoding['attention_mask'].squeeze(0),
            'evidence_input_ids': evidence_encoding['input_ids'].squeeze(0),
            'evidence_attention_mask': evidence_encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(item['class_label'], dtype=torch.long),
            'language': item.get('language', 'en')
        }

# ============== Main Pipeline ==============

def run_experiment(experiment_name, claim_encoder_name, evidence_encoder_name):
    """Run a single experiment with specified encoders"""
    
    # Configuration
    config = {
        # Model paths
        'claim_encoder_name': claim_encoder_name,
        'evidence_encoder_name': evidence_encoder_name,
        'num_labels': 5,
        'hidden_size': 768,
        'reasoning_hidden_size': 768,
        'num_attention_heads': 12,
        'dropout': 0.2,
        
        # Data paths
        'train_path': './data_splits/train.json',
        'dev_path': './data_splits/dev.json',
        'test_path': './data_splits/test.json',
        'max_claim_length': 64,
        'max_evidence_length': 256,
        
        # Training hyperparameters
        'batch_size': 8,
        'gradient_accumulation_steps': 4,
        'num_epochs': 15,
        'learning_rate': 2e-5,
        'encoder_lr': 1e-6,
        'warmup_ratio': 0.1,
        'patience': 7,
        'weight_decay': 0.01,
        'max_grad_norm': 1.0,
        
        # Other settings
        'use_class_weights': True,
        'pooling_method': 'mean',
        'label_smoothing': 0.1,
        
        # Output directory
        'output_dir': f'./hdcr_{experiment_name}_output'
    }
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Claim encoder: {claim_encoder_name}")
    logger.info(f"Evidence encoder: {evidence_encoder_name}")
    logger.info(f"{'='*50}\n")
    
    os.makedirs(config['output_dir'], exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Label mapping
    label_names = [
        "none",
        "over_generalization", 
        "improper_restriction",
        "effect_exaggeration",
        "spurious_causation",
    ]
    
    # Initialize model and tokenizers
    logger.info("Initializing model and tokenizers...")
    model = HDCR(config).to(device)
    claim_tokenizer = AutoTokenizer.from_pretrained(config['claim_encoder_name'])
    evidence_tokenizer = AutoTokenizer.from_pretrained(config['evidence_encoder_name'])
    
    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = MedicalMisinformationDataset(
        config['train_path'], claim_tokenizer, evidence_tokenizer, config
    )
    dev_dataset = MedicalMisinformationDataset(
        config['dev_path'], claim_tokenizer, evidence_tokenizer, config
    )
    test_dataset = MedicalMisinformationDataset(
        config['test_path'], claim_tokenizer, evidence_tokenizer, config
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Calculate class weights for imbalanced dataset
    if config.get('use_class_weights', False):
        from sklearn.utils.class_weight import compute_class_weight
        train_labels = [item['class_label'] for item in train_dataset.data]
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        logger.info(f"Using class weights: {class_weights}")
    else:
        class_weights = None
    
    # Setup optimizer with different learning rates for different components
    encoder_params = []
    interaction_params = []
    reasoning_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'claim_encoder' in name or 'evidence_encoder' in name:
            encoder_params.append(param)
        elif 'interaction_layer' in name:
            interaction_params.append(param)
        elif 'reasoning_module' in name:
            reasoning_params.append(param)
        else:
            classifier_params.append(param)
    
    optimizer = AdamW([
        {'params': encoder_params, 'lr': config['encoder_lr'], 'weight_decay': config['weight_decay']},
        {'params': interaction_params, 'lr': config['learning_rate'], 'weight_decay': config['weight_decay']},
        {'params': reasoning_params, 'lr': config['learning_rate'], 'weight_decay': config['weight_decay']},
        {'params': classifier_params, 'lr': config['learning_rate'] * 2, 'weight_decay': config['weight_decay']}
    ], eps=1e-8)
    
    # Learning rate scheduler with warmup
    num_training_steps = len(train_loader) * config['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps * config['warmup_ratio']),
        num_training_steps=num_training_steps
    )
    
    # Pass class weights and label smoothing to model
    if class_weights is not None:
        model.class_weights = class_weights
    model.label_smoothing = config.get('label_smoothing', 0.0)
    
    # Training loop
    logger.info("Starting training...")
    best_dev_f1 = 0
    patience_counter = 0
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    
    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        
        optimizer.zero_grad()
        for step, batch in enumerate(train_progress):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            outputs = model(
                claim_input_ids=batch['claim_input_ids'],
                claim_attention_mask=batch['claim_attention_mask'],
                evidence_input_ids=batch['evidence_input_ids'],
                evidence_attention_mask=batch['evidence_attention_mask'],
                labels=batch['labels']
            )
            
            loss = outputs['loss']
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            train_loss += loss.item() * gradient_accumulation_steps
            
            # Update weights every gradient_accumulation_steps
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('max_grad_norm', 1.0))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            train_progress.set_postfix({'loss': f"{loss.item() * gradient_accumulation_steps:.4f}"})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Evaluation phase on dev set
        model.eval()
        dev_predictions = []
        dev_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc="Evaluating"):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                outputs = model(
                    claim_input_ids=batch['claim_input_ids'],
                    claim_attention_mask=batch['claim_attention_mask'],
                    evidence_input_ids=batch['evidence_input_ids'],
                    evidence_attention_mask=batch['evidence_attention_mask']
                )
                
                predictions = torch.argmax(outputs['logits'], dim=-1)
                dev_predictions.extend(predictions.cpu().numpy())
                dev_labels.extend(batch['labels'].cpu().numpy())
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(dev_labels, dev_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            dev_labels, dev_predictions, average='weighted'
        )
        
        logger.info(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}")
        logger.info(f"Dev - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        # Save best model based on F1 score
        if f1 > best_dev_f1:
            best_dev_f1 = f1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(config['output_dir'], 'best_model.pt'))
            logger.info(f"New best model saved with F1: {best_dev_f1:.4f}")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= config['patience']:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Generate predictions on test set
    logger.info("\nLoading best model for test predictions...")
    model.load_state_dict(torch.load(os.path.join(config['output_dir'], 'best_model.pt')))
    model.eval()
    
    test_predictions_data = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Generating test predictions")):
            batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                           for k, v in batch.items()}
            
            outputs = model(
                claim_input_ids=batch_device['claim_input_ids'],
                claim_attention_mask=batch_device['claim_attention_mask'],
                evidence_input_ids=batch_device['evidence_input_ids'],
                evidence_attention_mask=batch_device['evidence_attention_mask']
            )
            
            predictions = torch.argmax(outputs['logits'], dim=-1)
            
            # Collect predictions for each sample
            batch_size = len(predictions)
            for i in range(batch_size):
                actual_idx = batch_idx * config['batch_size'] + i
                if actual_idx < len(test_dataset.data):
                    original_item = test_dataset.data[actual_idx]
                    
                    test_predictions_data.append({
                        "id": original_item['id'],
                        "claim": original_item['claim'],
                        "document": original_item['document'],
                        "true_label": original_item['class_label'],
                        "true_label_name": label_names[original_item['class_label']],
                        "pred_label": predictions[i].item(),
                        "pred_label_name": label_names[predictions[i].item()],
                        "correct": original_item['class_label'] == predictions[i].item(),
                        "language": original_item.get('language', 'en'),
                        "experiment": experiment_name
                    })
    
    # Save predictions to JSON file
    predictions_path = os.path.join(config['output_dir'], 'test_predictions.json')
    with open(predictions_path, 'w', encoding='utf-8') as f:
        json.dump({"predictions": test_predictions_data}, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Test predictions saved to {predictions_path}")
    logger.info(f"Experiment {experiment_name} completed!\n")
    
    return test_predictions_data

def main():
    """Run all experiments with different encoder combinations"""
    
    # Local model paths configuration
    model_paths = {
        'xlm-roberta-base': './models/xlm-roberta-base',
        'mbert': './models/bert-base-multilingual-cased',
        'infoxlm': './models/infoxlm-base',
        'biobert': './models/biobert-v1.1',
        'pubmedbert': './models/pubmedbert-base',
        'bge-m3': './models/bge-m3',
        'mt5': './models/mt5-base'
    }
    
    # Define experiment configurations
    experiments = [
        {
            'name': 'xlmr_pubmedbert',
            'claim_encoder': model_paths['xlm-roberta-base'],
            'evidence_encoder': model_paths['pubmedbert']
        },
        {
            'name': 'bge_pubmedbert',
            'claim_encoder': model_paths['bge-m3'],
            'evidence_encoder': model_paths['pubmedbert']
        },
        {
            'name': 'mbert_biobert',
            'claim_encoder': model_paths['mbert'],
            'evidence_encoder': model_paths['biobert']
        }
    ]
    
    all_predictions = {}
    
    # Run each experiment
    for exp in experiments:
        predictions = run_experiment(
            experiment_name=exp['name'],
            claim_encoder_name=exp['claim_encoder'],
            evidence_encoder_name=exp['evidence_encoder']
        )
        all_predictions[exp['name']] = predictions
    
    # Save all predictions in one combined file
    combined_predictions_path = './all_experiments_predictions.json'
    with open(combined_predictions_path, 'w', encoding='utf-8') as f:
        json.dump(all_predictions, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\nAll experiments completed!")
    logger.info(f"Combined predictions saved to {combined_predictions_path}")

if __name__ == "__main__":
    main()
