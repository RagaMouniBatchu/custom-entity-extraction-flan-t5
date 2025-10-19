"""
Create FLAN-T5 instruction fine-tuning dataset from extracted entities.
Expands dataset to one row per entity, creates FLAN-T5 style prompts,
and performs stratified sampling to reduce 'unknown' entities.
"""

import pandas as pd
import json
import random
from collections import Counter
from sklearn.model_selection import train_test_split
import numpy as np

def load_entities_data(file_path):
    """Load the extracted entities CSV file."""
    print(f"Loading entities data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    return df

def expand_to_entity_rows(df):
    """Expand dataset so each sample has 6 rows, one for each entity type."""
    print("\nExpanding dataset to entity-level rows...")
    
    entity_types = [
        'transaction_type',
        'account_type_source', 
        'account_type_target',
        'merchant',
        'channel',
        'direction'
    ]
    
    expanded_rows = []
    
    for idx, row in df.iterrows():
        complaint = row['complaint']
        
        for entity_type in entity_types:
            entity_value = row[entity_type]
            
            expanded_rows.append({
                'original_index': idx,
                'complaint': complaint,
                'entity_type': entity_type,
                'entity_value': entity_value
            })
    
    expanded_df = pd.DataFrame(expanded_rows)
    print(f"Expanded to {len(expanded_df)} rows ({len(df)} × {len(entity_types)} = {len(expanded_df)})")
    
    return expanded_df

def create_flan_t5_prompt(complaint, entity_type, entity_value):
    """Create FLAN-T5 style instruction prompt for entity extraction."""
    
    # Define entity descriptions for better context
    entity_descriptions = {
        'transaction_type': 'the type of financial transaction (debit, credit, transfer, purchase, cash_withdrawal, deposit, fee, interest, refund, chargeback, payment, other, unknown)',
        'account_type_source': 'the source account type where money/transaction originates (checking, savings, credit_card, brokerage, loan, cash, other, unknown)',
        'account_type_target': 'the target account type where money/transaction goes to (checking, savings, credit_card, brokerage, loan, cash, other, unknown, none)',
        'merchant': 'the merchant/vendor name if mentioned, or "unknown" if not specified',
        'channel': 'the payment/transaction channel used (ACH, wire, Zelle, Venmo, CashApp, card_present, card_not_present, ATM, mobile_check, branch, online, other, unknown)',
        'direction': 'whether the transaction is incoming (money coming to consumer) or outgoing (money going from consumer), or unknown'
    }
    
    # Create the instruction
    instruction = f"""Given the following consumer complaint, extract {entity_descriptions[entity_type]}.

Complaint: "{complaint}"

Extract the {entity_type}:"""

    # Create the response
    response = entity_value
    
    return instruction, response

def add_flan_t5_prompts(df):
    """Add FLAN-T5 instruction prompts to the expanded dataset."""
    print("\nCreating FLAN-T5 style instruction prompts...")
    
    instructions = []
    responses = []
    
    for idx, row in df.iterrows():
        instruction, response = create_flan_t5_prompt(
            row['complaint'], 
            row['entity_type'], 
            row['entity_value']
        )
        instructions.append(instruction)
        responses.append(response)
    
    df['instruction'] = instructions
    df['response'] = responses
    
    print(f"Added instruction prompts to {len(df)} samples")
    return df

def analyze_entity_distribution(df):
    """Analyze the distribution of entity values."""
    print("\nAnalyzing entity value distribution...")
    
    entity_distributions = {}
    for entity_type in df['entity_type'].unique():
        entity_values = df[df['entity_type'] == entity_type]['entity_value']
        distribution = entity_values.value_counts()
        entity_distributions[entity_type] = distribution
        
        print(f"\n{entity_type}:")
        print(f"  Total samples: {len(entity_values)}")
        print(f"  Unknown count: {distribution.get('unknown', 0)}")
        print(f"  Unknown percentage: {distribution.get('unknown', 0) / len(entity_values) * 100:.1f}%")
        print(f"  Top 5 values: {dict(distribution.head())}")
    
    return entity_distributions

def stratified_sampling(df, target_size=7000, max_unknown_ratio=0.25):
    """Perform stratified sampling to reduce 'unknown' entities while maintaining diversity."""
    print(f"\nPerforming stratified sampling to get {target_size} samples...")
    print(f"Target max 'unknown' ratio: {max_unknown_ratio * 100}%")
    
    # Separate known and unknown samples
    known_samples = df[df['entity_value'] != 'unknown'].copy()
    unknown_samples = df[df['entity_value'] == 'unknown'].copy()
    
    print(f"Known samples: {len(known_samples)}")
    print(f"Unknown samples: {len(unknown_samples)}")
    
    # Calculate how many unknown samples we can include
    max_unknown_count = int(target_size * max_unknown_ratio)
    
    # Sample from known samples (prioritize these)
    if len(known_samples) >= target_size - max_unknown_count:
        # We have enough known samples
        sampled_known = known_samples.sample(n=target_size - max_unknown_count, random_state=42)
        sampled_unknown = unknown_samples.sample(n=min(max_unknown_count, len(unknown_samples)), random_state=42)
    else:
        # Take all known samples and fill remaining with unknown
        sampled_known = known_samples
        remaining_slots = target_size - len(sampled_known)
        sampled_unknown = unknown_samples.sample(n=min(remaining_slots, len(unknown_samples)), random_state=42)
    
    # Combine and shuffle
    stratified_df = pd.concat([sampled_known, sampled_unknown]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Verify the final distribution
    unknown_ratio = len(stratified_df[stratified_df['entity_value'] == 'unknown']) / len(stratified_df)
    
    print(f"\nStratified sampling results:")
    print(f"Final dataset size: {len(stratified_df)}")
    print(f"Known samples: {len(stratified_df[stratified_df['entity_value'] != 'unknown'])}")
    print(f"Unknown samples: {len(stratified_df[stratified_df['entity_value'] == 'unknown'])}")
    print(f"Unknown ratio: {unknown_ratio * 100:.1f}%")
    
    return stratified_df

def create_train_test_split(df, test_size=2000, train_size=5000):
    """Create train and test splits."""
    print(f"\nCreating train/test split...")
    print(f"Target train size: {train_size}")
    print(f"Target test size: {test_size}")
    
    if len(df) < train_size + test_size:
        print(f"Warning: Dataset size ({len(df)}) is smaller than requested train+test size ({train_size + test_size})")
        # Adjust sizes proportionally
        total_requested = train_size + test_size
        train_size = int(len(df) * train_size / total_requested)
        test_size = len(df) - train_size
    
    # Perform stratified split to maintain entity type distribution
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        train_size=train_size,
        stratify=df['entity_type'],  # Stratify by entity type
        random_state=42
    )
    
    print(f"Train set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    return train_df, test_df

def save_flan_t5_dataset(train_df, test_df, output_dir='instruction_data'):
    """Save the dataset in FLAN-T5 format."""
    import os
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving FLAN-T5 dataset to {output_dir}/")
    
    # Save train set in FLAN-T5 format
    train_file = f"{output_dir}/train.jsonl"
    with open(train_file, 'w') as f:
        for idx, row in train_df.iterrows():
            data = {
                "input": row['instruction'],
                "output": row['response']
            }
            f.write(json.dumps(data) + '\n')
    
    # Save test set in FLAN-T5 format
    test_file = f"{output_dir}/test.jsonl"
    with open(test_file, 'w') as f:
        for idx, row in test_df.iterrows():
            data = {
                "input": row['instruction'],
                "output": row['response']
            }
            f.write(json.dumps(data) + '\n')
    
    # Save CSV versions for easy inspection
    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)
    
    # Save statistics
    stats = {
        "total_samples": len(train_df) + len(test_df),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "entity_types": list(train_df['entity_type'].unique()),
        "train_entity_distribution": train_df['entity_type'].value_counts().to_dict(),
        "test_entity_distribution": test_df['entity_type'].value_counts().to_dict(),
        "train_unknown_ratio": len(train_df[train_df['entity_value'] == 'unknown']) / len(train_df),
        "test_unknown_ratio": len(test_df[test_df['entity_value'] == 'unknown']) / len(test_df)
    }
    
    with open(f"{output_dir}/dataset_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Create README
    readme_content = """# FLAN-T5 Instruction Fine-tuning Dataset

## Dataset Overview
This dataset contains 7,000 instruction-response pairs for fine-tuning FLAN-T5 on financial entity extraction from consumer complaints.

- **Train set**: 5,000 samples
- **Test set**: 2,000 samples
- **Entity types**: transaction_type, account_type_source, account_type_target, merchant, channel, direction

## File Format (FLAN-T5)
- `train.jsonl` - Training data in FLAN-T5 format
- `test.jsonl` - Test data in FLAN-T5 format

**Format:**
```json
{"input": "Given the following consumer complaint, extract...", "output": "unknown"}
```

## Entity Types and Values

1. **transaction_type**: debit, credit, transfer, purchase, cash_withdrawal, deposit, fee, interest, refund, chargeback, payment, other, unknown
2. **account_type_source**: checking, savings, credit_card, brokerage, loan, cash, other, unknown
3. **account_type_target**: checking, savings, credit_card, brokerage, loan, cash, other, unknown, none
4. **merchant**: merchant name or "unknown"
5. **channel**: ACH, wire, Zelle, Venmo, CashApp, card_present, card_not_present, ATM, mobile_check, branch, online, other, unknown
6. **direction**: incoming, outgoing, unknown

## Usage Example

```python
import json

# Load training data
train_data = []
with open('instruction_data/train.jsonl', 'r') as f:
    for line in f:
        train_data.append(json.loads(line))
```

## Dataset Statistics
- Total samples: 7,000
- Train samples: 5,000
- Test samples: 2,000
- Unknown entity ratio: ~25%
- Balanced entity distribution across train/test splits
"""
    
    with open(f"{output_dir}/README.md", 'w') as f:
        f.write(readme_content)
    
    print(f"Saved files:")
    print(f"  - {train_file}")
    print(f"  - {test_file}")
    print(f"  - {output_dir}/train.csv")
    print(f"  - {output_dir}/test.csv")
    print(f"  - {output_dir}/dataset_stats.json")
    print(f"  - {output_dir}/README.md")
    
    return stats

def print_sample_prompts(df, num_samples=3):
    """Print sample prompts for inspection."""
    print(f"\nSample FLAN-T5 prompts:")
    print("=" * 80)
    
    for i in range(min(num_samples, len(df))):
        row = df.iloc[i]
        print(f"\nSample {i+1}:")
        print(f"Entity Type: {row['entity_type']}")
        print(f"Input: {row['instruction'][:200]}...")
        print(f"Output: {row['response']}")
        print("-" * 80)

def main():
    """Main function to create the FLAN-T5 instruction dataset."""
    print("🚀 Creating FLAN-T5 Instruction Fine-tuning Dataset")
    print("=" * 60)
    
    # Load the extracted entities data
    entities_file = 'extracted_entities_20251018_095556.csv'
    df = load_entities_data(entities_file)
    
    # Expand to entity-level rows (6500 → 39,000 rows)
    expanded_df = expand_to_entity_rows(df)
    
    # Add FLAN-T5 instruction prompts
    instruction_df = add_flan_t5_prompts(expanded_df)
    
    # Analyze entity distribution
    entity_distributions = analyze_entity_distribution(instruction_df)
    
    # Perform stratified sampling to reduce 'unknown' entities
    stratified_df = stratified_sampling(instruction_df, target_size=7000, max_unknown_ratio=0.25)
    
    # Create train/test split
    train_df, test_df = create_train_test_split(stratified_df, test_size=2000, train_size=5000)
    
    # Save the FLAN-T5 dataset
    stats = save_flan_t5_dataset(train_df, test_df)
    
    # Print sample prompts
    print_sample_prompts(train_df, num_samples=3)
    
    # Print final statistics
    print(f"\n📊 Final Dataset Statistics:")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Train samples: {stats['train_samples']}")
    print(f"Test samples: {stats['test_samples']}")
    print(f"Train unknown ratio: {stats['train_unknown_ratio'] * 100:.1f}%")
    print(f"Test unknown ratio: {stats['test_unknown_ratio'] * 100:.1f}%")
    
    print(f"\n✅ FLAN-T5 instruction dataset created successfully!")
    print(f"Ready for FLAN-T5 fine-tuning!")

if __name__ == "__main__":
    main()
