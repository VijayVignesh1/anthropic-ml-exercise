from transformers import AutoModel
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional, Dict, List
import torch

class PretrainedSentimentClassifier(nn.Module):
    def __init__(self, config: Dict, num_classes: int = 2) -> None:
        super().__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained(config['model']['name'])
        self.hidden_size: int = self.encoder.config.hidden_size
        if self.config['model']['freeze_encoder']: 
            for param in self.encoder.parameters():
                param.requires_grad = False 
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.hidden_size, num_classes) 

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  
        x = self.dropout(pooled)
        return self.classifier(x)

class SentimentLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for sentiment classification using a pretrained model.
    """
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.config = config
        self.model = PretrainedSentimentClassifier(self.config, num_classes=2)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = float(self.config['model']['learning_rate'])  
        self.weight_decay = float(self.config['training']['weight_decay'])  
        self.save_hyperparameters()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""

        return self.model(input_ids, attention_mask)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step for a single batch."""

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step for a single batch."""

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        self.log('val_loss', loss, prog_bar=self.config['training']['enable_progress_bar']) 
        self.log('val_acc', acc, prog_bar=self.config['training']['enable_progress_bar']) 

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step for a single batch."""

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log('test_loss', loss, prog_bar=self.config['training']['enable_progress_bar']) 
        self.log('test_acc', acc, prog_bar=self.config['training']['enable_progress_bar']) 
        
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer for training."""

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer
