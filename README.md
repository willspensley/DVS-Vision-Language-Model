# DVS-Vision-Language-Model

# Brain Tumor Classification Using CLIP

## 📖 Project Overview
This project focuses on classifying MRI scans into four categories—`glioma`, `meningioma`, `notumor`, and `pituitary`—using a fine-tuned CLIP model (`clip-vit-base-patch32`). Leveraging CLIP's vision-language capabilities to combine MRI images with descriptive captions, enabling the model to learn meaningful representations for classification. The project was developed as part of the *Design of Visual Systems* course at Imperial College London.

### Key Achievements
- Fine-tuned the CLIP model on the Brain Tumor MRI dataset for 2 epochs.
- Implemented data augmentation (random horizontal flips and rotations) to enhance model generalisation.
- Logged training and evaluation metrics, including per-class accuracy, precision, recall, and F1-score.
- Generated comprehensive visualisations to analyse model performance, such as training loss, training vs. test accuracy, per-class accuracy, per-class F1-score, and confusion matrix.
- Saved the fine-tuned model weights for reproducibility.

---

## 🚀 Setup and Installation

### Prerequisites
- Python 3.8+
- A Mac M1 or compatible device (CPU/GPU support for PyTorch)
- Git installed for cloning the repository

 **Download the Dataset**:

- Download the Brain Tumor MRI dataset from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).
- Place the dataset in a folder named `brain_tumor_data`.


### Training Script (`train_vlm.py`)

Below is the content of the `train_vlm.py` script, which you can copy and paste into a file named `train_vlm.py` above to train the CLIP model. 


This script will:
- Train the CLIP model for 2 epochs.
- Evaluate the model on the test set.
- Generate the following outputs in the `code` folder:
  - `model_weights.pth`
  - `training_metrics.json`
  - `evaluation_metrics.json`
  - Visualisation PNGs (moved to `assets` for the repository)
 
- `model_weights.pth`: The fine-tuned model weights (~600MB, available upon request as too large to upload to Github).
- `training_metrics.json`, `evaluation_metrics.json`: Saved metrics are save on this repository.

## Model Evaluation and Results

### Overview of Performance

After training the CLIP model for 2 epochs, I achieved an overall test accuracy of 65.83%. Below are the detailed metrics, visualisations, and a comprehensive analysis of the model’s performance.

### Detailed Metrics

- **Overall Test Accuracy**: 65.83%
- **Per-class Accuracy**:
  - `glioma`: 33.00% (99/300)
  - `meningioma`: 55.88% (171/306)
  - `notumor`: 77.28% (313/405)
  - `pituitary`: 93.33% (280/300)
- **Per-class Precision, Recall, F1-Score**:
  - `glioma`:
    - Precision: 0.8761
    - Recall: 0.3300
    - F1-Score: 0.4794
  - `meningioma`:
    - Precision: 0.5198
    - Recall: 0.5588
    - F1-Score: 0.5386
  - `notumor`:
    - Precision: 0.8459
    - Recall: 0.7728
    - F1-Score: 0.8077
  - `pituitary`:
    - Precision: 0.5611
    - Recall: 0.9333
    - F1-Score: 0.7009

### Visualisations

I generated several visualizations to analyze the model’s performance:

<p align="center"> <img src="assets/training_loss.png" /> </p>

The training loss fluctuates between 1.0 and 5.0, with a general downward trend but significant variability, ending at 2.2812. This indicates the model hasn’t fully converged after 2 epochs.

<p align="center"> <img src="assets/train_vs_test_accuracy.png" /> </p>

Training accuracy increases from 34.21% (epoch 1) to 57.76% (epoch 2), while test accuracy is 65.83%, suggesting good generalisation but potential underfitting.

<p align="center"> <img src="assets/per_class_accuracy.png" /> </p>

The model performs best on `pituitary` (93.33%) and `notumor` (77.28%), but struggles with `meningioma` (55.88%) and `glioma` (33.00%).

<p align="center"> <img src="assets/per_class_f1_score.png" /> </p>

F1-scores reflect the accuracy trends: `notumor` (0.8077), `pituitary` (0.7009), `meningioma` (0.5386), and `glioma` (0.4794). The low F1-score for `glioma` is due to its low recall (0.3300).

<p align="center"> <img src="assets/confusion_matrix.png" /> </p>

The confusion matrix shows significant misclassification for `glioma` (96 as `meningioma`, 105 as `pituitary`) and `meningioma` (52 as `notumor`, 74 as `pituitary`).
### Evaluation

#### What the Application Can Do

- **Classify MRI Scans**: The model can classify MRI scans into four categories with an overall test accuracy of 65.83%.
- **High Performance on Some Classes**: The model performs well on `pituitary` (93.33% accuracy) and `notumor` (77.28% accuracy), indicating it has learned distinct features for these classes.
- **Generalisation**: The test accuracy (65.83%) is higher than the training accuracy after 2 epochs (57.76%), suggesting the model generalises well to unseen data.
- **Visualise Performance**: The application generates detailed visualisations to analyse training progress and evaluation results.

#### What the Application Cannot Do

- **Struggles with `glioma` and `meningioma`**: The model has lower accuracy for `glioma` (33.00%) and `meningioma` (55.88%). The confusion matrix shows significant misclassification of `glioma` as `meningioma` (96) and `pituitary` (105), and `meningioma` as `notumor` (52) and `pituitary` (74).
- **Limited Training**: Training for only 2 epochs is insufficient for the model to fully converge, as evidenced by the high and fluctuating training loss (e.g., ending at 2.2812). More epochs are needed for better performance.
- **Class Imbalance**: The dataset has slight class imbalance (`notumor` has 405 test samples, while `glioma` and `pituitary` have 300 each, and `meningioma` has 306), which may contribute to the model’s bias towards `notumor` and `pituitary`.
- **Low Recall for `glioma`**: Despite high precision for `glioma` (0.8761), the recall is low (0.3300), meaning the model misses many actual `glioma` cases.

## Reflections and Future Improvements

### Reflections

This project taught me how to fine-tune a vision-language model like CLIP for medical imaging. I learned the importance of logging metrics, saving model weights, and using visualisations to analyse performance. The process of setting up the dataset, training the model, and evaluating its performance provided valuable insights into deep learning workflows.

### Future Improvements

To improve the model’s performance, I could:

- Train for more epochs (e.g., 5-10) to allow convergence.
- Address class imbalance by oversampling `glioma` and `meningioma` or using a weighted loss function.
- Experiment with different learning rates or optimisers (e.g., AdamW).
- Use a GPU (e.g., on Google Colab) to speed up training and allow for larger batch sizes.

## Personal Statements

### William Spensley (Single submission, no partner was involved for this project)

I contributed to the project by:

- Setting up the initial CLIP model and dataset loading.
- Implementing the training loop and handling errors to ensure stability on my Mac M1.
- Working on saving metrics and generating visualisations to analyse the model’s performance.
- Managing the GitHub repository setup and ensuring all deliverables were included.

**What I Learned**:

- How to fine-tune a vision-language model like CLIP for a medical imaging task.
- The importance of logging metrics and saving model weights to avoid losing progress.
- Techniques for evaluating model performance, such as per-class accuracy, precision, recall, F1-score, and confusion matrices.
- How to interpret visualisations to understand model strengths and weaknesses.

**Design Decisions**:

- I chose CLIP because it combines vision and language, which I thought would be effective for MRI classification with captions.
- I used a small batch size (8) to accommodate my Mac M1’s limited RAM (8GB).
- I included data augmentation (random flips and rotations) to improve generalisation.
- I trained for 2 epochs initially to test the pipeline, with plans to increase epochs for better performance.

**Mistakes Made**:

- Initially, I didn’t save the model weights, which meant we had to retrain from scratch after interruptions.
- I underestimated the number of epochs needed for convergence, leading to suboptimal performance on `glioma` and `meningioma`.
- I didn’t address class imbalance, which likely contributed to the model’s bias towards `notumor` and `pituitary`.

**What I Would Do Differently**:

- Train for more epochs (e.g., 5-10) to improve performance and allow the model to converge.
- Use a GPU (e.g., on Google Colab) to speed up training and allow for larger batch sizes.
- Address class imbalance by oversampling `glioma` and `meningioma` images or using weighted loss functions.
- Experiment with different learning rates or optimisers to improve training stability.
