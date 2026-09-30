# Project Architecture

## High-level architecture

```text
                    ┌──────────────────────┐
                    │       Farmer         │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Streamlit Frontend  │
                    └───────┬───────┬──────┘
                            │       │
             ┌──────────────┘       └──────────────┐
             ▼                                     ▼
┌────────────────────────┐             ┌────────────────────────┐
│   Market Predictor     │             │   Disease Detector     │
├────────────────────────┤             ├────────────────────────┤
│ Pandas / NumPy         │             │ PIL Image Processing   │
│ Statistics             │             │ Normalization           │
│ Feature Engineering   │             │ CNN                      │
│ Random Forest         │             │ Convolution + Pooling   │
└───────────┬────────────┘             └────────────┬───────────┘
            │                                       │
            ▼                                       ▼
   Estimated Market Price                    Disease Class
```

## ML pipeline

1. Collect/load data
2. Inspect and clean
3. Transform features
4. Split into training/testing data
5. Train regression model
6. Evaluate
7. Save model
8. Predict new input

## CNN pipeline

1. Load labeled image dataset
2. Resize images
3. Normalize pixels
4. Augment training images
5. Convolution
6. ReLU activation
7. Max pooling
8. Flatten
9. Dense layer
10. Softmax classification
11. Evaluate on validation/test data
12. Save model
