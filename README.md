# Deep Learning Challenge – Image Classification on ImageNet10

This repository documents the results of a deep learning challenge focused on CNN-based image classification using the **ImageNet10** dataset (10 classes, ~1300 images per class).

---

## 📁 Repository Structure

```text
.
├── data/                   # Contains ImageNet10 images (train/val/test split)
├── models/                 # Custom CNN classes and architecture variants
├── notebooks/
│   └── main_experiments.ipynb   # Jupyter notebook with all experiments and documentation
├── results/                # Saved plots and performance comparison charts
├── src/
│   └── utils.py            # Utility functions
├── README.md               # Project overview and usage instructions
└── requirements.txt        # Python dependencies
```

---

## What Was Done

This project involved an in-depth exploration of the effects of various hyperparameters and architectural choices on CNN performance. The goal was to:

1. Build a **baseline CNN model**.
2. Formulate hypotheses and **conduct experiments** across:
   - Learning rate  
   - Batch size  
   - Weight initialization  
   - Model depth and width (Conv/FC layers)  
   - Kernel sizes  
   - Dropout regularization  
   - Batch normalization  
   - Optimizer selection (SGD vs Adam)  
   - Data augmentation  
   - Transfer learning using pretrained **ResNet18**
3. Evaluate and compare results visually using **W&B plots**.
4. Combine the best settings into a **final model** achieving the highest test accuracy.
5. Compare the final custom CNN with ResNet18 and baseline models.

---

## 🏁 Results Summary

| Model             | Test Accuracy |
|------------------|---------------|
| Baseline CNN     | ~74%          |
| ResNet18 (FT)    | ~80%          |
| Final Custom CNN | **>83%**      |

---

## Notes

- All training and evaluation logic is encapsulated in the `Trainer` class.
- Models were trained for 40 epochs with consistent logging using **Weights & Biases (wandb)**.
- The notebook is fully documented with hypotheses, observations, and reflections.
- No code execution is required to explore the results — just open the notebook and view the visualizations and comments.
