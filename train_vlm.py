print("Starting script...")

try:
    import os
    print("Imported os")
    from PIL import Image
    print("Imported PIL")
    import torch
    print("Imported torch")
    from torch.utils.data import Dataset, DataLoader
    print("Imported torch.utils.data")
    from transformers import CLIPProcessor, CLIPModel
    print("Imported transformers")
    import torch.nn.functional as F
    print("Imported torch.nn.functional")
    from torchvision import transforms
    print("Imported torchvision.transforms")
    import numpy as np
    print("Imported numpy")
    import matplotlib.pyplot as plt
    print("Imported matplotlib")
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
    print("Imported sklearn.metrics")
    import json
    print("Imported json")
except Exception as e:
    print(f"Error during imports: {e}")
    raise

# Custom Dataset
class MRIDataset(Dataset):
    def __init__(self, data_dir, processor, is_train=True):
        self.data_dir = data_dir
        self.processor = processor
        self.is_train = is_train
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5) if is_train else transforms.Lambda(lambda x: x),
            transforms.RandomRotation(10) if is_train else transforms.Lambda(lambda x: x),
        ])
        self.image_paths = []
        self.captions = []
        self.labels = []

        print("Loading dataset...")
        try:
            for label in os.listdir(data_dir):
                print(f"Processing label: {label}")
                label_dir = os.path.join(data_dir, label)
                if os.path.isdir(label_dir):
                    print(f"Entering directory: {label_dir}")
                    for img_file in os.listdir(label_dir):
                        if img_file.endswith(".jpg"):
                            self.image_paths.append(os.path.join(label_dir, img_file))
                            caption = f"MRI scan with {label} tumor" if label != "notumor" else "MRI scan with no tumor"
                            self.captions.append(caption)
                            self.labels.append(label)
                            if len(self.image_paths) % 1000 == 0:
                                print(f"Processed {len(self.image_paths)} images so far...")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

        print(f"Dataset loaded: {len(self.image_paths)} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        print(f"Loading image {idx}: {self.image_paths[idx]}")
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
            image = self.transform(image)  # Apply augmentations
            text = self.captions[idx]
            inputs = self.processor(text=[text], images=[image], return_tensors="pt", padding=True)
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
            inputs["label"] = self.labels[idx]
            return inputs
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            return None

# Custom collate function to pad variable-length sequences
def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    
    keys = batch[0].keys()
    padded_batch = {}
    
    for key in keys:
        if key == "label":
            padded_batch[key] = [item[key] for item in batch]
        else:
            tensors = [item[key] for item in batch]
            if key in ["input_ids", "attention_mask"]:
                max_length = max(tensor.size(0) for tensor in tensors)
                padded_tensors = torch.zeros(len(tensors), max_length, dtype=tensors[0].dtype)
                for i, tensor in enumerate(tensors):
                    length = tensor.size(0)
                    padded_tensors[i, :length] = tensor
                padded_batch[key] = padded_tensors
            else:
                padded_batch[key] = torch.stack(tensors, dim=0)
    
    return padded_batch

# Load CLIP model and processor
print("Loading CLIP model...")
try:
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.train()
    print(f"Model loaded on device: {device}")
except Exception as e:
    print(f"Error loading CLIP model: {e}")
    raise

# Create training dataset and dataloader
train_data_dir = "../brain_tumor_data/Training"
print(f"Creating training dataset with data_dir: {train_data_dir}")
try:
    train_dataset = MRIDataset(train_data_dir, processor, is_train=True)
    print("Creating training dataloader...")
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)
except Exception as e:
    print(f"Error creating training dataset/dataloader: {e}")
    raise

# Training loop with logging
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
num_epochs = 2
start_epoch = 0  # Start from the beginning since we don't have saved weights
start_batch = 0  # Start from the beginning of the epoch
train_losses = []
train_accuracies = []
possible_labels = ["glioma", "meningioma", "notumor", "pituitary"]
possible_captions = [f"MRI scan with {label} tumor" if label != "notumor" else "MRI scan with no tumor" for label in possible_labels]

print("Starting training...")
try:
    for epoch in range(start_epoch, num_epochs):
        model.train()
        correct = 0
        total = 0
        for batch_idx, batch in enumerate(train_dataloader):
            if epoch == start_epoch and batch_idx < start_batch:
                continue
            if batch is None:
                print(f"Skipping batch {batch_idx} due to failed image loading")
                continue
            print(f"Processing batch {batch_idx}/{len(train_dataloader)}")
            batch_labels = batch.pop("label")
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text

            labels = torch.arange(len(batch["input_ids"])).to(device)
            loss_i = F.cross_entropy(logits_per_image, labels)
            loss_t = F.cross_entropy(logits_per_text, labels)
            loss = (loss_i + loss_t) / 2
            train_losses.append(loss.item())

            # Compute training accuracy
            with torch.no_grad():
                image_embeds = outputs.image_embeds
                text_inputs = processor(text=possible_captions, return_tensors="pt", padding=True).to(device)
                text_outputs = model.get_text_features(**text_inputs)
                text_embeds = text_outputs
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                logits = image_embeds @ text_embeds.t()
                predictions = logits.argmax(dim=-1)
                predicted_labels = [possible_labels[pred] for pred in predictions]
                for pred_label, true_label in zip(predicted_labels, batch_labels):
                    total += 1
                    if pred_label == true_label:
                        correct += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_dataloader)}, Loss: {loss.item():.4f}")
        train_accuracies.append(correct / total)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Accuracy: {train_accuracies[-1]:.4f}")
except Exception as e:
    print(f"Error during training: {e}")
    raise

print("Training complete!")

# Save the model weights
print("Saving model weights...")
torch.save(model.state_dict(), "model_weights.pth")
print("Model weights saved to model_weights.pth")

# Save training metrics
print("Saving training metrics...")
with open("training_metrics.json", "w") as f:
    json.dump({
        "train_losses": train_losses,
        "train_accuracies": train_accuracies
    }, f)
print("Training metrics saved to training_metrics.json")

# Evaluation on test set with per-class accuracy
test_data_dir = "../brain_tumor_data/Testing"
print(f"Creating test dataset with data_dir: {test_data_dir}")
try:
    test_dataset = MRIDataset(test_data_dir, processor, is_train=False)
    print("Creating test dataloader...")
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate_fn)
except Exception as e:
    print(f"Error creating test dataset/dataloader: {e}")
    raise

print("Starting evaluation...")
model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []
class_correct = {label: 0 for label in possible_labels}
class_total = {label: 0 for label in possible_labels}

try:
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            if batch is None:
                print(f"Skipping batch {batch_idx} due to failed image loading")
                continue
            print(f"Evaluating batch {batch_idx}/{len(test_dataloader)}")
            batch_labels = batch.pop("label")
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            image_embeds = outputs.image_embeds
            text_inputs = processor(text=possible_captions, return_tensors="pt", padding=True).to(device)
            text_outputs = model.get_text_features(**text_inputs)
            text_embeds = text_outputs

            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            logits = image_embeds @ text_embeds.t()

            predictions = logits.argmax(dim=-1)
            predicted_labels = [possible_labels[pred] for pred in predictions]
            all_preds.extend(predicted_labels)
            all_labels.extend(batch_labels)

            for pred_label, true_label in zip(predicted_labels, batch_labels):
                total += 1
                class_total[true_label] += 1
                if pred_label == true_label:
                    correct += 1
                    class_correct[true_label] += 1

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(test_dataloader)}, Accuracy so far: {correct/total:.4f}")
except Exception as e:
    print(f"Error during evaluation: {e}")
    raise

accuracy = correct / total
print(f"Evaluation complete! Accuracy: {accuracy:.4f}")

# Calculate per-class accuracy
print("\nPer-class accuracy:")
class_accuracies = {}
for label in possible_labels:
    class_acc = class_correct[label] / class_total[label] if class_total[label] > 0 else 0
    class_accuracies[label] = class_acc
    print(f"{label}: {class_acc:.4f} ({class_correct[label]}/{class_total[label]})")

# Calculate precision, recall, F1-score
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, labels=possible_labels, average=None)
class_metrics = {}
for i, label in enumerate(possible_labels):
    class_metrics[label] = {
        "precision": precision[i],
        "recall": recall[i],
        "f1_score": f1[i]
    }
    print(f"\n{label} Metrics:")
    print(f"Precision: {precision[i]:.4f}")
    print(f"Recall: {recall[i]:.4f}")
    print(f"F1-Score: {f1[i]:.4f}")

# Save evaluation metrics
print("Saving evaluation metrics...")
with open("evaluation_metrics.json", "w") as f:
    json.dump({
        "overall_accuracy": accuracy,
        "class_accuracies": class_accuracies,
        "class_metrics": class_metrics,
        "all_preds": all_preds,
        "all_labels": all_labels
    }, f)
print("Evaluation metrics saved to evaluation_metrics.json")

# Visualize results
# 1. Training Loss Plot
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Training Loss")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.legend()
plt.savefig("training_loss.png")
plt.close()

# 2. Training vs Test Accuracy Plot
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_accuracies, marker='o', label="Training Accuracy")
plt.axhline(y=accuracy, color='r', linestyle='--', label=f"Test Accuracy ({accuracy:.4f})")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Test Accuracy")
plt.legend()
plt.savefig("train_vs_test_accuracy.png")
plt.close()

# 3. Per-class Accuracy Bar Plot
plt.figure(figsize=(10, 5))
plt.bar(class_accuracies.keys(), class_accuracies.values())
plt.xlabel("Class")
plt.ylabel("Accuracy")
plt.title("Per-class Accuracy on Test Set")
plt.savefig("per_class_accuracy.png")
plt.close()

# 4. Per-class F1-Score Bar Plot
f1_scores = [class_metrics[label]["f1_score"] for label in possible_labels]
plt.figure(figsize=(10, 5))
plt.bar(possible_labels, f1_scores)
plt.xlabel("Class")
plt.ylabel("F1-Score")
plt.title("Per-class F1-Score on Test Set")
plt.savefig("per_class_f1_score.png")
plt.close()

# 5. Confusion Matrix
cm = confusion_matrix(all_labels, all_preds, labels=possible_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=possible_labels)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

print("Visualizations saved as PNG files: training_loss.png, train_vs_test_accuracy.png, per_class_accuracy.png, per_class_f1_score.png, confusion_matrix.png")