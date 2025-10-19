"""
Entity extraction from CFPB complaints using Gemini 2.5 Flash with structured output.
Extracts financial entities from complaint narratives.
"""

import json
import pandas as pd
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import random
from typing import Dict, Any, List
import os
from datetime import datetime

# Configure Gemini API
def setup_gemini(api_key: str = None):
    """Setup Gemini API with the provided key or from environment variable."""
    if api_key:
        genai.configure(api_key=api_key)
    else:
        # Try to get from environment variable
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            # Try to load from .env file
            if os.path.exists('.env'):
                with open('.env', 'r') as f:
                    for line in f:
                        if line.startswith('GEMINI_API_KEY='):
                            api_key = line.strip().split('=', 1)[1]
                            # Remove quotes if present
                            api_key = api_key.strip('"').strip("'")
                            os.environ['GEMINI_API_KEY'] = api_key
                            break
            
            if not api_key:
                raise ValueError("Please provide GEMINI_API_KEY environment variable or run setup_api_key.py first")
        genai.configure(api_key=api_key)

# Define the structured output schema
ENTITY_SCHEMA = {
    "type": "object",
    "properties": {
        "transaction_type": {
            "type": "string",
            "enum": ["debit", "credit", "transfer", "purchase", "cash_withdrawal", "deposit", "fee", "interest", "refund", "chargeback", "payment", "other", "unknown"]
        },
        "account_type_source": {
            "type": "string", 
            "enum": ["checking", "savings", "credit_card", "brokerage", "loan", "cash", "other", "unknown"]
        },
        "account_type_target": {
            "type": "string",
            "enum": ["checking", "savings", "credit_card", "brokerage", "loan", "cash", "other", "unknown", "none"]
        },
        "merchant": {
            "type": "string",
            "description": "Normalize common merchant variants if easy, otherwise null"
        },
        "channel": {
            "type": "string",
            "enum": ["ACH", "wire", "Zelle", "Venmo", "CashApp", "card_present", "card_not_present", "ATM", "mobile_check", "branch", "online", "other", "unknown"]
        },
        "direction": {
            "type": "string",
            "enum": ["incoming", "outgoing", "unknown"]
        }
    },
    "required": ["transaction_type", "account_type_source", "account_type_target", "merchant", "channel", "direction"]
}

# Create the prompt template
PROMPT_TEMPLATE = """Analyze the following consumer complaint narrative and extract financial entities according to the specified schema.

Focus on identifying:
- transaction_type: The type of financial transaction mentioned
- account_type_source: The source account type (where money/transaction originates)
- account_type_target: The target account type (where money/transaction goes to, or "none" if not applicable)
- merchant: The merchant/vendor name if mentioned (normalize common variants, otherwise null)
- channel: The payment/transaction channel used
- direction: Whether the transaction is incoming (money coming to the consumer) or outgoing (money going from the consumer)

Complaint: "{complaint_text}"

Extract entities based on the information provided in the complaint. If information is not clear or not mentioned, use "unknown" or "none" as appropriate."""

def extract_entities_single(complaint_text: str, model_name: str = "gemini-2.5-flash-lite") -> Dict[str, Any]:
    """
    Extract entities from a single complaint text using Gemini.
    
    Args:
        complaint_text: The complaint narrative to analyze
        model_name: The Gemini model to use
    
    Returns:
        Dictionary with extracted entities or error information
    """
    try:
        # Initialize the model
        model = genai.GenerativeModel(model_name)
        
        # Create the prompt
        prompt = PROMPT_TEMPLATE.format(complaint_text=complaint_text)
        
        # Generate content with structured output
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                response_schema=ENTITY_SCHEMA
            )
        )
        
        # Parse the JSON response
        try:
            entities = json.loads(response.text)
        except json.JSONDecodeError as e:
            # If JSON parsing fails, try to extract JSON from markdown
            text = response.text.strip()
            if text.startswith('```json'):
                text = text[7:]
            if text.endswith('```'):
                text = text[:-3]
            entities = json.loads(text.strip())
        
        # Validate that all required fields are present and are strings
        required_fields = ["transaction_type", "account_type_source", "account_type_target", "merchant", "channel", "direction"]
        for field in required_fields:
            if field not in entities:
                entities[field] = "unknown" if field != "account_type_target" else "none"
            elif isinstance(entities[field], list):
                # If it's a list, take the first element
                entities[field] = entities[field][0] if entities[field] else "unknown"
            elif not isinstance(entities[field], str) and entities[field] is not None:
                # Convert to string if it's not a string or None
                entities[field] = str(entities[field])
        
        return {
            "success": True,
            "entities": entities,
            "error": None
        }
        
    except Exception as e:
        return {
            "success": False,
            "entities": None,
            "error": str(e)
        }

def process_complaints_batch(complaints_data: List[Dict], max_workers: int = 10, model_name: str = "gemini-2.5-flash-lite") -> Dict[str, Any]:
    """
    Process a batch of complaints using multi-threaded API calls.
    
    Args:
        complaints_data: List of dictionaries with 'index' and 'complaint' keys
        max_workers: Number of concurrent threads
        model_name: The Gemini model to use
    
    Returns:
        Dictionary with results, errors, and statistics
    """
    results = []
    errors = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(extract_entities_single, data['complaint'], model_name): data['index'] 
            for data in complaints_data
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(complaints_data), desc="Processing complaints") as pbar:
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    # Find the complaint data by index
                    complaint_data = next((data for data in complaints_data if data['index'] == index), None)
                    complaint_text = complaint_data['complaint'] if complaint_data else "Unknown"
                    
                    if result['success']:
                        results.append({
                            'index': index,
                            'complaint': complaint_text,
                            **result['entities']
                        })
                    else:
                        errors.append({
                            'index': index,
                            'complaint': complaint_text,
                            'error': result['error']
                        })
                except Exception as e:
                    # Find the complaint data by index
                    complaint_data = next((data for data in complaints_data if data['index'] == index), None)
                    complaint_text = complaint_data['complaint'] if complaint_data else "Unknown"
                    errors.append({
                        'index': index,
                        'complaint': complaint_text,
                        'error': str(e)
                    })
                
                pbar.update(1)
                # Small delay to avoid rate limiting
                time.sleep(0.1)
    
    return {
        'results': results,
        'errors': errors,
        'total_processed': len(complaints_data),
        'successful': len(results),
        'failed': len(errors)
    }

def test_extraction(df: pd.DataFrame, num_samples: int = 20, model_name: str = "gemini-2.5-flash-lite") -> Dict[str, Any]:
    """
    Test entity extraction on a small sample of complaints.
    
    Args:
        df: DataFrame with complaint data
        num_samples: Number of samples to test
        model_name: The Gemini model to use
    
    Returns:
        Dictionary with test results
    """
    print(f"\n🧪 Testing entity extraction on {num_samples} random samples...")
    
    # Get random samples
    test_indices = random.sample(range(len(df)), min(num_samples, len(df)))
    test_data = [
        {'index': idx, 'complaint': df.iloc[idx]['Consumer complaint narrative']}
        for idx in test_indices
    ]
    
    # Process the test batch
    results = process_complaints_batch(test_data, max_workers=5, model_name=model_name)
    
    # Print summary
    print(f"\n📊 Test Results:")
    print(f"✅ Successful extractions: {results['successful']}")
    print(f"❌ Failed extractions: {results['failed']}")
    print(f"📈 Success rate: {results['successful']/results['total_processed']*100:.1f}%")
    
    # Show sample results
    if results['results']:
        print(f"\n🔍 Sample extracted entities:")
        for i, result in enumerate(results['results'][:3]):
            print(f"\nSample {i+1}:")
            print(f"Complaint: {result['complaint'][:100]}...")
            print(f"Entities: {json.dumps({k: v for k, v in result.items() if k not in ['index', 'complaint']}, indent=2)}")
    
    # Show errors if any
    if results['errors']:
        print(f"\n⚠️ Sample errors:")
        for error in results['errors'][:3]:
            print(f"Index {error['index']}: {error['error']}")
    
    # Save results to file
    if results['results']:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"extracted_entities_test_{num_samples}_{timestamp}.csv"
        results_df = pd.DataFrame(results['results'])
        results_df.to_csv(results_file, index=False)
        print(f"\n💾 Saved {len(results['results'])} test results to: {results_file}")
    
    if results['errors']:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        errors_file = f"extraction_errors_test_{num_samples}_{timestamp}.csv"
        errors_df = pd.DataFrame(results['errors'])
        errors_df.to_csv(errors_file, index=False)
        print(f"💾 Saved {len(results['errors'])} test errors to: {errors_file}")
    
    return results

def process_all_complaints(df: pd.DataFrame, model_name: str = "gemini-2.5-flash-lite", max_workers: int = 10) -> Dict[str, Any]:
    """
    Process all complaints in the dataframe.
    
    Args:
        df: DataFrame with complaint data
        model_name: The Gemini model to use
        max_workers: Number of concurrent threads
    
    Returns:
        Dictionary with results
    """
    print(f"\n🚀 Processing all {len(df)} complaints...")
    
    # Prepare data
    all_data = [
        {'index': idx, 'complaint': df.iloc[idx]['Consumer complaint narrative']}
        for idx in range(len(df))
    ]
    
    # Process in batches to avoid memory issues
    batch_size = 100
    all_results = []
    all_errors = []
    
    for i in range(0, len(all_data), batch_size):
        batch = all_data[i:i+batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{(len(all_data) + batch_size - 1)//batch_size}")
        
        batch_results = process_complaints_batch(batch, max_workers=max_workers, model_name=model_name)
        all_results.extend(batch_results['results'])
        all_errors.extend(batch_results['errors'])
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save successful results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_file = f"extracted_entities_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\n💾 Saved {len(all_results)} successful extractions to: {results_file}")
    
    # Save errors
    if all_errors:
        errors_df = pd.DataFrame(all_errors)
        errors_file = f"extraction_errors_{timestamp}.csv"
        errors_df.to_csv(errors_file, index=False)
        print(f"💾 Saved {len(all_errors)} errors to: {errors_file}")
    
    # Final summary
    total_processed = len(all_results) + len(all_errors)
    success_rate = len(all_results) / total_processed * 100 if total_processed > 0 else 0
    
    print(f"\n📊 Final Results:")
    print(f"✅ Total successful: {len(all_results)}")
    print(f"❌ Total failed: {len(all_errors)}")
    print(f"📈 Overall success rate: {success_rate:.1f}%")
    
    return {
        'results': all_results,
        'errors': all_errors,
        'total_processed': total_processed,
        'successful': len(all_results),
        'failed': len(all_errors),
        'success_rate': success_rate
    }

def main():
    """Main function to run the entity extraction pipeline."""
    print("🔍 CFPB Entity Extraction using Gemini 2.5 Flash")
    print("=" * 50)
    
    # Setup
    try:
        setup_gemini()
        print("✅ Gemini API configured successfully")
    except Exception as e:
        print(f"❌ Error setting up Gemini API: {e}")
        return
    
    # Load data
    try:
        df = pd.read_csv('cfpb_sampled_6500.csv')
        print(f"✅ Loaded {len(df)} complaints from CSV")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return
    
    # Full processing phase
    print("\n" + "="*50)
    print("🚀 FULL PROCESSING PHASE")
    print("="*50)
    
    final_results = process_all_complaints(df)
    print("\n🎉 Entity extraction completed!")

if __name__ == "__main__":
    main()
