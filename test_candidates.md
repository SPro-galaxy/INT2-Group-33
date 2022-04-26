# Testing notes
All tests carried out to 90 epochs. To view performance logs...
```sh
tensorboard --logdir="logs"
```
Best results so far `tc1`, `tc3`, `tc8`.

# tc1
Original code, no changes made.

# tc2
Lowered dropout on all Conv layers to 0.15.

# tc3 
Adding a new Conv layer with 256 filters.

**This shows promise as the val_accuracy is still increasing at epoch 90.**

# tc4
Adding two new Conv layers with 256 and 512 filters.

**Failed.**

# tc5
Adding a new Conv layer with 256 filters and another fully connected layer.

**Results were almost the same as `tc3` but more sporadic, some clear outliers.**

# tc6
Adding a new Conv layer with 256 filters and three extra fully connected layers.

**Failed.**

# tc7
Adding a new Conv layer with 256 filters and lowering dropout on all Conv layers to 0.1.

**It seems like lowering the dropout to this level massivly increased the overfitting to the training data.**

# tc8
Adding a new Conv layer with 256 filters and raising dropout on all Conv layers to 0.35.

**This appears to have helped negate the overfitting issue and provides very similar performance to `tc3`.**
