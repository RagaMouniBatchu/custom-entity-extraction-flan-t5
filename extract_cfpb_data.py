"""
Script to download CFPB complaints data from HuggingFace,
filter by financial keywords and token count, and randomly sample 6500 rows.
"""

import pandas as pd
from datasets import load_dataset
import re

def main():
    print("Loading CFPB Complaints dataset from HuggingFace...")
    # Load the dataset from HuggingFace
    dataset = load_dataset("23daVinci/CFPB_Complaints")
    
    # Convert to pandas DataFrame
    # The dataset might have different splits, so we'll check
    if 'train' in dataset:
        df = dataset['train'].to_pandas()
    else:
        # If there's no 'train' split, use the first available split
        split_name = list(dataset.keys())[0]
        df = dataset[split_name].to_pandas()
    
    print(f"Loaded {len(df)} total rows")
    print(f"Columns: {df.columns.tolist()}")
    
    # Define keywords to search for
    keywords = [
        'transfer', 'wire', 'ACH', 'deposit', 'withdraw', 'fee', 'charge',
        'interest', 'chargeback', 'refund', 'credit card', 'checking',
        'savings', 'brokerage', 'statement', 'balance', 'merchant',
        'Zelle', 'Venmo', 'cash app', 'ATM', 'overdraft', 'NSF'
    ]
    
    # Create a regex pattern for case-insensitive matching
    # Using word boundaries to match whole words
    pattern = '|'.join([r'\b' + re.escape(keyword) + r'\b' for keyword in keywords])
    
    print(f"\nFiltering rows containing keywords: {', '.join(keywords)}")
    
    # Find the text column(s) to search in
    # Common column names in CFPB data: 'Consumer complaint narrative', 'Complaint', 'complaint_text', etc.
    text_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':  # String columns
            text_columns.append(col)
    
    print(f"Searching in columns: {text_columns}")
    
    # Filter rows that contain any of the keywords in any text column
    mask = pd.Series([False] * len(df))
    for col in text_columns:
        # Convert to string and handle NaN values
        col_mask = df[col].fillna('').astype(str).str.contains(pattern, case=False, regex=True, na=False)
        mask = mask | col_mask
    
    filtered_df = df[mask]
    print(f"\nFound {len(filtered_df)} rows containing the specified keywords")
    
    # Filter for rows with less than 50 tokens in the complaint narrative
    # Assuming 'Consumer complaint narrative' is the main text column
    if 'Consumer complaint narrative' in filtered_df.columns:
        text_col = 'Consumer complaint narrative'
    else:
        # Use the first text column if 'Consumer complaint narrative' is not found
        text_col = text_columns[0]
    
    print(f"Applying token filter (< 50 tokens) on column: {text_col}")
    
    # Count tokens (splitting by whitespace)
    token_counts = filtered_df[text_col].fillna('').astype(str).str.split().str.len()
    token_mask = token_counts < 50
    
    filtered_df = filtered_df[token_mask]
    print(f"After token filtering: {len(filtered_df)} rows with < 50 tokens")
    
    # Randomly sample 6500 rows
    if len(filtered_df) >= 6500:
        sampled_df = filtered_df.sample(n=6500, random_state=42)
        print(f"Randomly sampled 6500 rows")
    else:
        sampled_df = filtered_df
        print(f"Warning: Only {len(filtered_df)} rows available (less than 6500 requested)")
    
    # Save to CSV
    output_file = 'cfpb_sampled_6500.csv'
    sampled_df.to_csv(output_file, index=False)
    print(f"\nData saved to '{output_file}'")
    print(f"Final dataset shape: {sampled_df.shape}")

if __name__ == "__main__":
    main()
