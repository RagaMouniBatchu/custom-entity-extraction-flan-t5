# custom-entity-extraction-flan-t5

### Performance Comparison Across Training Stages

| Training Stage    | Samples | Epochs | Accuracy | Improvement |
|------------------|:-------:|:------:|:--------:|:-----------:|
| Initial Model     | 500     | 2      | 42%      | Baseline    |
| Medium Dataset    | 1000    | 3      | 61%      | +19%        |
| Full Dataset      | 5000    | 3      | 63%      | +21%        |


### Final Entity-Specific Accuracy (Full Dataset)

| Entity Type            | Accuracy |
|----------------------|:--------:|
| merchant              | 82.4%    |
| account_type_target   | 70.0%    |
| direction             | 68.4%    |
| transaction_type      | 62.5%    |
| account_type_source   | 55.6%    |
| channel               | 45.0%    |

