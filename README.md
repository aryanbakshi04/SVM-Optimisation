# SVM Optimization on UCI Letter Recognition Dataset

## Overview
This project demonstrates how to optimize a Support Vector Machine (SVM) classifier on a multiclass dataset using randomized hyperparameter search across multiple train/test splits. We use the UCI dataset, which contains 20,000 samples of uppercase letters (A–Z) represented by 16 numerical features.

## Methodology
1. **Data Loading and Analytics**  
   - Load the Letter Recognition dataset via `sklearn.datasets.fetch_openml`.  
   - Display dataset shape and per-class sample counts to verify balance and integrity.

2. **Train/Test Splits**  
   - Generate **10 different random splits** (70% train, 30% test) using `train_test_split` with `random_state` = 1…10.  
   - Label each split as S1, S2, …, S10 for reproducibility.

3. **Randomized Hyperparameter Search**  
   - For each split, perform **100 iterations** of random sampling over:
     - Kernel: `['rbf', 'poly', 'sigmoid']`
     - Regularization C: log‑uniform from 1e-3 to 1e3 (1,000 candidates)
     - Gamma: `['scale', 'auto']`
   - Train an `SVC(**params)` on the training set, evaluate accuracy on the test set.
   - Track the **best test accuracy** and corresponding hyperparameters for that split.  
   - Also record the **running maximum** of accuracy per iteration for convergence plotting.

4. **Results Aggregation**  
   - Compile a **results table** (Table 1) listing, for each sample S1–S10:
     - Best Accuracy achieved.
     - Best SVM parameters (`kernel`, `C`, `gamma`).

5. **Convergence Visualization**  
   - Identify the split with the **highest overall test accuracy**.  
   - Plot a **convergence graph** of "best‑so‑far" accuracy vs. iteration number (Figure 1).  This shows how quickly the random search reaches high performance and whether it plateaus.

## Results Table (Table 1)
| Sample | Best Accuracy |        Kernel        |      C Value      | Gamma |
|:------:|:-------------:|:--------------------:|:-----------------:|:-----:|
|   S1   |     0.92      |         rbf          |       12.3        | scale |
|   S2   |     0.91      |         poly         |       45.7        | auto  |
|   S3   |     0.93      |         rbf          |        3.4        | scale |
|  ...   |      ...      |         ...          |        ...        |  ...  |
|   S10  |     0.94      |         rbf          |       27.1        | scale |

## Convergence Graph (Figure 1)

![Convergence Plot](convergence_plot.png)

- **X‑axis (Iteration)**: Each of the 100 random search iterations.
- **Y‑axis (Best Accuracy)**: The highest test accuracy observed up to that iteration.
- **Interpretation**: Rapid early gains indicate easy wins in hyperparameter space; any later improvements highlight fine-tuning of `C` or kernel choice. A plateau suggests diminishing returns from further random sampling.

## How to Reproduce
1. Install required packages:  
   ```bash
   pip install scikit-learn pandas matplotlib
   ```
2. Run the script/notebook:  
   ```bash
   python svm_optimization.py
   ```
3. Review the generated table in console output and view `Figure 1`.
4. Commit results and scripts to GitHub for submission.
---
