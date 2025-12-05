# Hull Tactical Market Prediction Dataset

## Files

### `train_cleaned.csv` 
- **Full historical dataset**
- Period: ~1990 - 2025-06 (estimated)
- Total trading days: 8,841
- Time span: ~35.1 years

### `train_2007_2025.csv`
- **Truncated dataset for training**
- Period: ~2007-01 - 2025-06
- Total trading days: 4,625
- Time span: ~18.35 years
- Created by filtering: `date_id >= 4216` from original dataset
- **date_id reset to start from 0**

## Rationale for Truncation

The 2007-2025 period was selected because:
1. Covers recent market dynamics (post-2007 financial crisis)
2. Includes major events:
   - 2008 Financial Crisis
   - 2009-2021 QE era (ultra-low rates)
   - 2020 COVID-19 pandemic
   - 2022-2023 Rate hiking cycle
3. More relevant for current market predictions
4. Reduces data from outdated market regimes (pre-2007)

## Key Statistics (2007-2025 dataset)

- Average risk-free rate: 1.39%
- Rate range: -0.11% to 5.35%
- Ultra-low rate period (<0.5%): 10.3 years (mainly 2009-2021)
- Average daily return: 0.013%

## Usage

For model training, use `train_2007_2025.csv` which contains more relevant recent market data.
