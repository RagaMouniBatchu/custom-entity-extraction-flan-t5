"""
Test entity extraction accuracy on the trained FLAN-T5 model.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import random

def load_trained_model(model_path="basic_checkpoints/final_model"):
    """Load the trained model and tokenizer."""
    print(f"Loading trained model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    print(f"Model loaded successfully!")
    return model, tokenizer

def test_entity_extraction(model, tokenizer, test_data_path="instruction_data/test.jsonl", num_samples=1500):
    """Test entity extraction on a subset of test data."""
    
    print(f"Testing entity extraction on {num_samples} samples...")
    
    # Load test data
    with open(test_data_path, 'r') as f:
        test_data = [json.loads(line) for line in f]
    
    # Sample random test cases
    if len(test_data) > num_samples:
        test_samples = random.sample(test_data, num_samples)
    else:
        test_samples = test_data
    
    print(f"Testing on {len(test_samples)} samples")
    
    predictions = []
    ground_truths = []
    
    for i, sample in enumerate(test_samples):
        if i % 10 == 0:
            print(f"Processing sample {i+1}/{len(test_samples)}")
        
        # Get prediction with safety checks
        input_text = sample['input']
        expected_output = sample['output']
        
        # Ensure both are strings
        if not isinstance(input_text, str):
            input_text = str(input_text) if input_text is not None else ""
        if not isinstance(expected_output, str):
            expected_output = str(expected_output) if expected_output is not None else "unknown"
        
        # Tokenize input
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=256)
        
        # Generate prediction
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=32,
                num_beams=2,
                early_stopping=True,
                do_sample=False
            )
        
        # Decode prediction
        predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        predictions.append(predicted_text.strip())
        ground_truths.append(expected_output.strip())
    
    return predictions, ground_truths, test_samples

def analyze_results(predictions, ground_truths, test_samples):
    """Analyze the results and provide detailed metrics."""
    
    print("\n" + "="*60)
    print("ENTITY EXTRACTION TEST RESULTS")
    print("="*60)
    
    # Calculate exact match accuracy
    exact_matches = sum(1 for pred, truth in zip(predictions, ground_truths) if pred.lower() == truth.lower())
    exact_accuracy = exact_matches / len(predictions)
    
    print(f"Exact Match Accuracy: {exact_accuracy:.4f} ({exact_matches}/{len(predictions)})")
    
    # Analyze by entity type
    entity_types = {}
    for i, sample in enumerate(test_samples):
        # Extract entity type from instruction
        instruction = sample['input']
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
            entity_types[entity_type] = {'predictions': [], 'ground_truths': []}
        
        entity_types[entity_type]['predictions'].append(predictions[i])
        entity_types[entity_type]['ground_truths'].append(ground_truths[i])
    
    print(f"\nEntity Type Performance:")
    print("-" * 40)
    for entity_type, data in entity_types.items():
        entity_matches = sum(1 for pred, truth in zip(data['predictions'], data['ground_truths']) 
                           if pred.lower() == truth.lower())
        entity_accuracy = entity_matches / len(data['predictions'])
        print(f"{entity_type:20s}: {entity_accuracy:.4f} ({entity_matches}/{len(data['predictions'])})")
    
    # Show some examples
    print(f"\nSample Predictions:")
    print("-" * 40)
    
    # Show correct predictions
    correct_examples = [(i, pred, truth) for i, (pred, truth) in enumerate(zip(predictions, ground_truths)) 
                       if pred.lower() == truth.lower()]
    
    print(f"✅ Correct Predictions ({len(correct_examples)} examples):")
    for i, (idx, pred, truth) in enumerate(correct_examples[:5]):
        instruction = test_samples[idx]['input'][:100] + "..."
        print(f"  {i+1}. {instruction}")
        print(f"     Predicted: {pred}")
        print(f"     Expected:  {truth}")
        print()
    
    # Show incorrect predictions
    incorrect_examples = [(i, pred, truth) for i, (pred, truth) in enumerate(zip(predictions, ground_truths)) 
                         if pred.lower() != truth.lower()]
    
    print(f"❌ Incorrect Predictions ({len(incorrect_examples)} examples):")
    for i, (idx, pred, truth) in enumerate(incorrect_examples[:5]):
        instruction = test_samples[idx]['input'][:100] + "..."
        print(f"  {i+1}. {instruction}")
        print(f"     Predicted: {pred}")
        print(f"     Expected:  {truth}")
        print()
    
    # Value distribution analysis
    print(f"Predicted Value Distribution:")
    print("-" * 30)
    pred_counts = pd.Series(predictions).value_counts()
    for value, count in pred_counts.head(10).items():
        print(f"  {value}: {count}")
    
    print(f"\nGround Truth Value Distribution:")
    print("-" * 30)
    truth_counts = pd.Series(ground_truths).value_counts()
    for value, count in truth_counts.head(10).items():
        print(f"  {value}: {count}")
    
    return exact_accuracy

def main():
    """Main function to test entity extraction."""
    
    print("🧪 Testing Entity Extraction Accuracy")
    print("=" * 50)
    
    try:
        # Load trained model
        model, tokenizer = load_trained_model()
        
        # Test entity extraction on all remaining test samples (1500)
        predictions, ground_truths, test_samples = test_entity_extraction(model, tokenizer, num_samples=1500)
        
        # Analyze results
        accuracy = analyze_results(predictions, ground_truths, test_samples)
        
        print(f"\n🎯 Final Test Accuracy: {accuracy:.4f}")
        
        # Save results
        results = {
            "accuracy": accuracy,
            "num_samples": len(predictions),
            "predictions": predictions,
            "ground_truths": ground_truths
        }
        
        with open("test_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Test results saved to test_results.json")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        raise

if __name__ == "__main__":
    main()
