import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import pytorch_lightning as pl
from typing import Dict, List, Any

class IMDBDataset(torch.utils.data.Dataset):
    """IMDB dataset with BERT tokenization."""
    def __init__(self, data: Any, tokenizer, max_len: int = 256) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.data[idx]['text']
        label = self.data[idx]['label']
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'text': text, 
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class IMDBDataModule(pl.LightningDataModule):
    def __init__(self, config: Dict) -> None: 
        super().__init__()
        self.config = config
        dataset = load_dataset("imdb")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model']['name'])

        # Native splits: train=25K, test=25K
        train_data = dataset['train']
        test_data = dataset['test']  # Use 5K for testing to speed up

        train_val_split = train_data.train_test_split(test_size=self.config['data']['val_split'], seed=42) 
        train_data_final = train_val_split['train'] 
        val_data = train_val_split['test'] 

        train_dataset = IMDBDataset(train_data_final, self.tokenizer, self.config['model']['max_len']) 
        val_dataset = IMDBDataset(val_data, self.tokenizer, self.config['model']['max_len']) 
        test_dataset = IMDBDataset(test_data, self.tokenizer, self.config['model']['max_len'])

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset 
        self.batch_size = self.config['data']['batch_size'] 

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        texts = [item['text'] for item in batch]
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        return {'text': texts, 'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                         shuffle=True, collate_fn=self.collate_fn, num_workers=self.config['data']['num_workers']) 

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                         collate_fn=self.collate_fn, num_workers=self.config['data']['num_workers'])

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is not None:
            return DataLoader(self.test_dataset, batch_size=self.batch_size,
                            collate_fn=self.collate_fn, num_workers=self.config['data']['num_workers'])
        else:
            return None