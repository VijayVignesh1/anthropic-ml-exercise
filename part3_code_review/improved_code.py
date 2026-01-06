import torch
import torch.nn as nn
from typing import Tuple, Optional
from torch.utils.data import DataLoader

class TransformerModel(nn.Module):
    """Transformer encoder model for sequence classification."""
    
    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_layers: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            batch_first=True 
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            mask: Padding mask of shape (batch_size, seq_len)
            
        Returns:
            Output logits of shape (batch_size, seq_len, vocab_size)
        """
        x = self.embedding(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.fc(x)
        return x

def train_model(
    model: nn.Module, 
    data: DataLoader, 
    epochs: int = 10
) -> None:
    """
    Train transformer model.
    
    Args:
        model: TransformerModel instance
        data: DataLoader with (inputs, targets) batches
        epochs: Number of training epochs
    """
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
   
    model.train() 
    
    for epoch in range(epochs):
        for batch in data:
            inputs, targets = batch
            optimizer.zero_grad()  
            
            outputs = model(inputs)
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    vocab_size: int = 10000
    model = TransformerModel(vocab_size, 512, 8, 6)
    train_data = DataLoader(sample_dataset, batch_size=32, shuffle=True) # define sample_dataset
    train_model(model, train_data)
