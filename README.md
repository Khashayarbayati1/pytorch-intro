<<<<<<< HEAD
# PyTorch Fundamentals – Dose Response Neural Net
=======
# PyTorch + Lightning Fundamentals – Dose Response Neural Net
>>>>>>> 15268f3 (Add Lightning version of training pipeline)

This repository demonstrates a simple neural network built with PyTorch to model a dose-response curve using a minimal dataset. The model consists of fixed weights and biases (except for the final bias), and the training process tunes this final parameter to match the desired output.

---

## 🧠 Model Architecture

The model uses:
- Two parallel ReLU blocks
- A learnable final bias
- Fixed weights and biases for all other components

Here's the architecture:

![Model Architecture](images/model-diagram.png)

---

## 📈 Target Output

The training dataset consists of just 3 points, intended to produce a “Yes–No–Yes” (bump) pattern across dosage levels.

![Target Dose Response](images/target-dose-response.png)

- Input: `[0.0, 0.5, 1.0]`
- Labels: `[0.0, 1.0, 0.0]`

---

## 🚀 Training

- Optimizer: SGD with `lr = 0.1`
- Loss: Squared error between predicted and target values
<<<<<<< HEAD
- Only the final bias is updated during training


=======
- Only the final bias is updated during training
>>>>>>> 15268f3 (Add Lightning version of training pipeline)
