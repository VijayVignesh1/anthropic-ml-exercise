import os
import yaml
import torch
import torchmetrics
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import seaborn as sns
import glob
from pytorch_lightning import Trainer, seed_everything

from model import SentimentLightningModule, PretrainedSentimentClassifier
from dataset_loader import IMDBDataModule

def validate_config_dict(config_dict: Dict) -> None:
    """
    Eval specific config validation for raw YAML dict.
    Raises ValueError with all errors.
    """
    errors = []

    required_sections = {
        "model": ["name"],
    }
    
    for section, required_fields in required_sections.items():
        if section not in config_dict:
            errors.append(f"Missing section: {section}")
            continue
            
        section_data = config_dict.get(section, {})
        for field in required_fields:
            if field not in section_data:
                errors.append(f"Missing required field: {section}.{field}")

    model_data = config_dict.get("model", {})

    # Model name validation
    valid_models = ["prajjwal1/bert-tiny", "huawei-noah/TinyBERT_General_4L_312D"]
    model_name = model_data.get("name", "")
    if model_name and model_name not in valid_models:
        errors.append(f"Unknown model '{model_name}'. Tested: {valid_models}")
    
    # raise errors if any found
    if errors:
        raise ValueError("\n".join(errors))
    
    print("Config validation Passed!")
    
def load_best_model(config: Dict) -> SentimentLightningModule:
    """Load best checkpoint model."""
    
    ckpt_dir = Path(config['paths']['models_dir'])
    ckpt_path = ckpt_dir / "sentiment-analyzer.ckpt"

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    model = SentimentLightningModule(config)
    
    # Load weights
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()

    return model

def error_analysis(model, dataloader, device, top_k: int = 20) -> pd.DataFrame:
    """Perform error analysis on test set."""
    model.eval()
    errors = []
    with torch.no_grad():
        for batch in dataloader:
            # inputs = {k: v.to(device) for k, v in batch.items() if k != 'text'}
            inference_inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)
            }
            texts = batch['text']
            preds = model(**inference_inputs)
            targets = batch['labels'].to(device)
            wrong = preds.argmax(-1) != targets
            for i in range(len(wrong)):
                if wrong[i]:
                    errors.append({
                        'text': texts[i],
                        'pred': preds[i].argmax().item(),
                        'true': targets[i].item(),
                        'length': len(texts[i].split())
                    })
    df = pd.DataFrame(errors[:top_k])
    print("Top Errors:\n", df.head())
    return df

def main(config_path: str = "config.yaml") -> None:
    """Comprehensive evaluation script."""

    seed_everything(42)
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML: {e}")
    
    validate_config_dict(config)
    
    data_module = IMDBDataModule(config)
    model = load_best_model(config)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    trainer = Trainer(
        accelerator='auto', devices=1,
        enable_progress_bar=True
    )

    print("Running test evaluation...")
    test_metrics = trainer.test(model, datamodule=data_module)[0]
    print(f"Test Metrics: {test_metrics}")
    
    # Error analysis
    print("Error Analysis...")
    error_df = error_analysis(model, data_module.test_dataloader(), device)
    
if __name__ == "__main__":
    main()
