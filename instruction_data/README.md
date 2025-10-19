# FLAN-T5 Instruction Fine-tuning Dataset

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
