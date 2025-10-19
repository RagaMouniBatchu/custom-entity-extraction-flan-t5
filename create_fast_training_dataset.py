"""
Create a smaller, faster training dataset for quick fine-tuning.
"""

import json
import random

def create_fast_training_dataset():
    """Create a smaller dataset for faster training."""
    
    print("🚀 Creating Fast Training Dataset")
    print("=" * 50)
    
    # Load original training data
    with open("instruction_data/train.jsonl", 'r') as f:
        train_data = [json.loads(line) for line in f]
    
    with open("instruction_data/test.jsonl", 'r') as f:
        test_data = [json.loads(line) for line in f]
    
    print(f"Original train samples: {len(train_data)}")
    print(f"Original test samples: {len(test_data)}")
    
    # Sample smaller datasets for faster training
    # Use 10% of original data (500 train, 200 test) for CPU training
    fast_train_size = 500
    fast_test_size = 200
    
    # Stratified sampling to maintain entity distribution
    entity_types = {}
    for item in train_data:
        # Extract entity type from instruction
        instruction = item['input']
        if 'transaction_type' in instruction:
            entity_type = 'transaction_type'
        elif 'account_type_source' in instruction:
            entity_type = 'account_type_source'
        elif 'account_type_target' in instruction:
            entity_type = 'account_type_target'
        elif 'merchant' in instruction:
            entity_type = 'merchant'
        elif 'channel' in instruction:
            entity_type = 'channel'
        elif 'direction' in instruction:
            entity_type = 'direction'
        else:
            entity_type = 'other'
        
        if entity_type not in entity_types:
            entity_types[entity_type] = []
        entity_types[entity_type].append(item)
    
    # Sample from each entity type proportionally
    fast_train_data = []
    for entity_type, items in entity_types.items():
        sample_size = min(len(items), fast_train_size // len(entity_types))
        sampled = random.sample(items, sample_size)
        fast_train_data.extend(sampled)
    
    # If we need more samples, fill with random samples
    if len(fast_train_data) < fast_train_size:
        remaining_needed = fast_train_size - len(fast_train_data)
        remaining_items = [item for item in train_data if item not in fast_train_data]
        additional_samples = random.sample(remaining_items, min(remaining_needed, len(remaining_items)))
        fast_train_data.extend(additional_samples)
    
    # Sample test data
    fast_test_data = random.sample(test_data, min(fast_test_size, len(test_data)))
    
    # Shuffle the data
    random.shuffle(fast_train_data)
    random.shuffle(fast_test_data)
    
    print(f"Fast train samples: {len(fast_train_data)}")
    print(f"Fast test samples: {len(fast_test_data)}")
    
    # Save fast training data
    with open("instruction_data/train_fast.jsonl", 'w') as f:
        for item in fast_train_data:
            f.write(json.dumps(item) + '\n')
    
    with open("instruction_data/test_fast.jsonl", 'w') as f:
        for item in fast_test_data:
            f.write(json.dumps(item) + '\n')
    
    print("✅ Fast training dataset created!")
    print("Files:")
    print("  - instruction_data/train_fast.jsonl")
    print("  - instruction_data/test_fast.jsonl")

if __name__ == "__main__":
    create_fast_training_dataset()
