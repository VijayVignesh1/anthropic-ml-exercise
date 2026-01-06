import pytorch_lightning as pl
import torch
import yaml
import torch.nn as nn
from model import SentimentLightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from typing import Dict
from dataset_loader import IMDBDataModule

def validate_config_dict(config_dict: Dict) -> None:
    """
    Complete config validation for raw YAML dict.
    Raises ValueError with all errors.
    """
    errors = []

    required_sections = {
        "model": ["name"],
        "data": ["batch_size"],
        "training": ["max_epochs"]
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
    data_data = config_dict.get("data", {})
    training_data = config_dict.get("training", {})
    
    try:
        if "learning_rate" in model_data:
            lr = float(model_data["learning_rate"])
            if lr <= 0 or lr > 1:
                errors.append(f"Invalid learning_rate: {lr} (must be 0 < lr <= 1)")
        
        if "max_len" in model_data:
            max_len = int(model_data["max_len"])
            if max_len < 32 or max_len > 512:
                errors.append(f"Invalid max_len: {max_len} (32-512)")
        
        # Data validation
        if "batch_size" in data_data:
            batch_size = int(data_data["batch_size"])
            if batch_size < 1 or batch_size > 128:
                errors.append(f"Invalid batch_size: {batch_size} (1-128)")
        
        train_split = float(data_data.get("train_split", 0.8))
        val_split = float(data_data.get("val_split", 0.2))
        if train_split + val_split > 1.0:
            errors.append(f"Invalid splits: {train_split} + {val_split} > 1.0")
        
        # Training validation
        if "max_epochs" in training_data:
            epochs = int(training_data["max_epochs"])
            if epochs < 1 or epochs > 1000:
                errors.append(f"Invalid max_epochs: {epochs} (1-1000)")
        
        if "weight_decay" in training_data:
            wd = float(training_data["weight_decay"])
            if wd < 0 or wd > 0.1:
                errors.append(f"Invalid weight_decay: {wd} (0-0.1)")
    
    except (ValueError, TypeError) as e:
        errors.append(f"Type conversion error: {e}")
    
    # Model name validation
    valid_models = ["prajjwal1/bert-tiny", "huawei-noah/TinyBERT_General_4L_312D"]
    model_name = model_data.get("name", "")
    if model_name and model_name not in valid_models:
        errors.append(f"Unknown model '{model_name}'. Tested: {valid_models}")
    
    # raise errors if any found
    if errors:
        raise ValueError("\n".join(errors))
    
    print("Config validation Passed!")


def main(config_path: str = "config.yaml") -> None:
    """Main function to set up data module, model, and trainer, then start training and testing."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML: {e}")
    
    validate_config_dict(config)

    data_module = IMDBDataModule(config)
    model = SentimentLightningModule(config)
    logger = TensorBoardLogger(config['paths']['logger_dir'], name="eval")

    # Lightning Trainer with early stopping & checkpointing
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=config['paths']['checkpoint_monitor'], 
        dirpath= config['paths']['models_dir'], 
        filename='sentiment-analyzer',
        mode=config['training']['mode'],  
        save_last=config['training']['save_last'] 
    )

    early_stopping = pl.callbacks.EarlyStopping(
        monitor=config['paths']['checkpoint_monitor'],  
        patience=config['training']['patience'], 
        mode=config['training']['mode'] 
    )

    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],  
        accelerator=config['training']['accelerator'],
        devices=config['training']['devices'],  
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=config['training']['log_every_n_steps'], 
        enable_progress_bar=config['training']['enable_progress_bar'], 
        logger=logger
    )

    print(f"Starting training..")
    trainer.fit(model, data_module)
    trainer.test(model, data_module)

    print(f"Logs: {trainer.logged_metrics}")

if __name__ == "__main__":
    main()