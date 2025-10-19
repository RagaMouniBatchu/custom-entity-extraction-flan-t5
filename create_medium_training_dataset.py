import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def create_medium_training_dataset(
    train_input_path='instruction_data/train.jsonl',
    test_input_path='instruction_data/test.jsonl',
    output_dir='instruction_data',
    medium_train_size=1000,  # Double the current fast training size
    medium_test_size=400     # Double the current fast test size
):
    """
    Creates a medium-sized subset of the instruction dataset for better training.
    This doubles the current training samples while still being manageable for CPU training.
    """
    print("🚀 Creating Medium Training Dataset")
    print("="*50)

    # Load original datasets
    with open(train_input_path, 'r') as f:
        train_data = [json.loads(line) for line in f]
    with open(test_input_path, 'r') as f:
        test_data = [json.loads(line) for line in f]

    print(f"Original train samples: {len(train_data)}")
    print(f"Original test samples: {len(test_data)}")

    # Convert to DataFrame for easier stratified sampling
    df_train = pd.DataFrame(train_data)
    df_test = pd.DataFrame(test_data)

    # Extract entity type from instruction for stratification
    def extract_entity_type_from_instruction(instruction):
        # Extract entity type from the instruction format
        # Example: "Extract the transaction_type:"
        lines = instruction.split('\n')
        for line in lines:
            if line.startswith("Extract the "):
                return line.replace("Extract the ", "").replace(":", "").strip()
        return "unknown_entity_type"

    df_train['entity_type'] = df_train['input'].apply(extract_entity_type_from_instruction)
    df_test['entity_type'] = df_test['input'].apply(extract_entity_type_from_instruction)

    # Perform stratified sampling
    if len(df_train) > medium_train_size:
        df_train_medium, _ = train_test_split(
            df_train, 
            train_size=medium_train_size, 
            stratify=df_train['entity_type'], 
            random_state=42
        )
    else:
        df_train_medium = df_train

    if len(df_test) > medium_test_size:
        df_test_medium, _ = train_test_split(
            df_test, 
            train_size=medium_test_size, 
            stratify=df_test['entity_type'], 
            random_state=42
        )
    else:
        df_test_medium = df_test

    # Remove the temporary 'entity_type' column
    df_train_medium = df_train_medium.drop(columns=['entity_type'])
    df_test_medium = df_test_medium.drop(columns=['entity_type'])

    print(f"Medium train samples: {len(df_train_medium)}")
    print(f"Medium test samples: {len(df_test_medium)}")

    # Save medium datasets
    train_output_path = os.path.join(output_dir, 'train_medium.jsonl')
    test_output_path = os.path.join(output_dir, 'test_medium.jsonl')

    with open(train_output_path, 'w') as f:
        for item in df_train_medium.to_dict(orient='records'):
            f.write(json.dumps(item) + '\n')
    
    with open(test_output_path, 'w') as f:
        for item in df_test_medium.to_dict(orient='records'):
            f.write(json.dumps(item) + '\n')

    print("✅ Medium training dataset created!")
    print("Files:")
    print(f"  - {train_output_path}")
    print(f"  - {test_output_path}")
    
    # Show entity distribution for verification
    print("\nEntity type distribution in medium training set:")
    train_entities = [extract_entity_type_from_instruction(item['input']) for item in df_train_medium.to_dict(orient='records')]
    entity_counts = pd.Series(train_entities).value_counts()
    print(entity_counts.to_string())

if __name__ == "__main__":
    create_medium_training_dataset()
