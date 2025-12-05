# Description

Wisdom from most personal finance experts would suggest that it's irresponsible to try and time the market. The Efficient Market Hypothesis (EMH) would agree: everything knowable is already priced in, so don’t bother trying.

But in the age of machine learning, is it irresponsible to not try and time the market? Is the EMH an extreme oversimplification at best and possibly just…false?

This competition is about more than predictive modeling. Predicting market returns challenges the assumptions of market efficiency. Your work could help reshape how investors and academics understand financial markets. Participants could uncover signals others overlook, develop innovative strategies, and contribute to a deeper understanding of market behavior—potentially rewriting a fundamental principle of modern finance. Most investors don’t beat the S&P 500. That failure has been used for decades to prop up EMH: If even the professionals can’t win, it must be impossible. This observation has long been cited as evidence for the Efficient Market Hypothesis the idea that prices already reflect all available information and no persistent edge is possible. This story is tidy, but reality is less so. Markets are noisy, messy, and full of behavioral quirks that don’t vanish just because academic orthodoxy said they should.

Data science has changed the game. With enough features, machine learning, and creativity, it’s possible to uncover repeatable edges that theory says shouldn’t exist. The real challenge isn’t whether they exist—it’s whether you can find them and combine them in a way that is robust enough to overcome frictions and implementation issues.

Our current approach blends a handful of quantitative models to adjust market exposure at the close of each trading day. It points in the right direction, but with a blurry compass. Our model is clearly a sub-optimal way to model a complex, non-linear, adaptive system. This competition asks you to do better: to build a model that predicts excess returns and includes a betting strategy designed to outperform the S&P 500 while staying within a 120% volatility constraint. We’ll provide daily data that combines public market information with our proprietary dataset, giving you the raw material to uncover patterns most miss.

Unlike many Kaggle challenges, this isn’t just a theoretical exercise. The models you build here could be valuable in live investment strategies. And if you succeed, you’ll be doing more than improving a prediction engine—you’ll be helping to demonstrate that financial markets are not fully efficient, challenging one of the cornerstones of modern finance, and paving the way for better, more accessible tools for investors.

# Evaluation
The competition's metric is a variant of the Sharpe ratio that penalizes strategies that take on significantly more volatility than the underlying market or fail to outperform the market's return. The metric code is available here.

## Submission File
You must submit to this competition using the provided evaluation API, which ensures that models do not peek forward in time. For each trading day, you must predict an optimal allocation of funds to holding the S&P500. As some leverage is allowed, the valid range covers 0 to 2. See this example notebook [demo_submission.ipynb](demo_submission.ipynb) for more details.

# Timeline
This is a forecasting competition with an active training phase and a separate forecasting phase where models will be run against real market returns.
Training Timeline:
September 16, 2025 - Start Date.
December 8, 2025 - Entry Deadline. You must accept the competition rules before this date in order to compete.
December 8, 2025 - Team Merger Deadline. This is the last day participants may join or merge teams.
December 15, 2025 - Final Submission Deadline.
All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## Forecasting Timeline:
Starting after the final submission deadline there will be periodic updates to the leaderboard to reflect market data updates that will be run against selected notebooks.

June 16, 2026 - Competition End Date

# Code Requirements

Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met:

- CPU Notebook <= 8 hours run-time
- GPU Notebook <= 8 hours run-time
- Internet access disabled
- Freely & publicly available external data is allowed, including pre-trained models

## Forecasting Phase
The run-time limits for both CPU and GPU notebooks will be extended to 9 hours during the forecasting phase. You must ensure your submission completes within that time.

The extra hour is to help protect against time-out failures due to the extended size of the test set. You are still responsible for ensuring your submission completes within the 9 hour limit, however. See the Data page for details on the extended test set during the forecasting phase.

Please see the Code Competition FAQ for more information on how to submit. And review the code debugging doc if you are encountering submission errors.