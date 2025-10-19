"""
Basic FLAN-T5 fine-tuning with minimal complexity for reliable training.
"""

import json
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer
)
import os

class BasicEntityDataset(Dataset):
    """Basic dataset for entity extraction training."""
    
    def __init__(self, data_path, tokenizer):
        self.tokenizer = tokenizer
        
        # Load data
        self.data = []
        with open(data_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        print(f"Loaded {len(self.data)} samples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Simple formatting with safety checks
        input_text = item['input']
        output_text = item['output']
        
        # Ensure both are strings
        if not isinstance(input_text, str):
            input_text = str(input_text) if input_text is not None else ""
        if not isinstance(output_text, str):
            output_text = str(output_text) if output_text is not None else "unknown"
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            truncation=True,
            max_length=200,
            padding='max_length',
            return_tensors='pt'
        )
        
        targets = self.tokenizer(
            output_text,
            truncation=True,
            max_length=20,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

def main():
    """Main training function."""
    
    print("🚀 Basic FLAN-T5 Fine-tuning")
    print("=" * 50)
    
    # Configuration
    model_name = "google/flan-t5-small"
    train_data_path = "instruction_data/train.jsonl"
    test_data_path = "instruction_data/test.jsonl"
    output_dir = "basic_checkpoints"
    
    print(f"Model: {model_name}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and tokenizer
    print("\n📦 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    print(f"Model loaded: {model.num_parameters():,} parameters")
    
    # Load datasets
    print("\n📊 Loading datasets...")
    train_dataset = BasicEntityDataset(train_data_path, tokenizer)
    eval_dataset = BasicEntityDataset(test_data_path, tokenizer)
    
    # Training arguments - very simple
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,  # Increased from 2 to 3 epochs
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-4,
        warmup_steps=50,
        weight_decay=0.01,
        
        # Evaluation
        eval_strategy="no",  # Disable evaluation during training
        logging_steps=10,
        save_strategy="epoch",
        
        # Checkpointing
        save_total_limit=2,
        
        # Performance
        dataloader_num_workers=0,
        report_to="none",
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    # Start training
    print("\n🏋️ Starting full dataset training...")
    print(f"Training samples: {len(train_dataset)} (full dataset)")
    print(f"Test samples: {len(eval_dataset)} (full test set)")
    print(f"Epochs: 3")
    
    try:
        # Train the model
        trainer.train()
        
        # Save final model
        final_model_path = f"{output_dir}/final_model"
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        
        print(f"\n✅ Training completed successfully!")
        print(f"Final model saved to: {final_model_path}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        print("Saving current checkpoint...")
        trainer.save_model(f"{output_dir}/interrupted_checkpoint")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise
    
    print("\n🎉 Basic fine-tuning process completed!")

if __name__ == "__main__":
    main()
