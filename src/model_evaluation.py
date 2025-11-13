import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # FORCE CPU

import numpy as np
import cv2
from tqdm import tqdm
from keras.models import load_model
from skimage.io import imread
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, jaccard_score, f1_score, 
                             precision_score, recall_score, accuracy_score,
                             roc_curve, auc, precision_recall_curve)
import seaborn as sns


# Paths
TEST_PATH = '/home/volcani/Unet/data/plowing/test/'
MASK_TEST_PATH = '/home/volcani/Unet/data/plowing/mask_test/'
MODEL_PATH = '/home/volcani/Unet/model_for_plowing.h5'
OUTPUT_DIR = '/home/volcani/Unet/evaluation_results/'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Image settings
IMG_HEIGHT = 512
IMG_WIDTH = 512
IMG_CHANNELS = 3

# Load test images
test_images = sorted([f for f in os.listdir(TEST_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
print(f"Found {len(test_images)} test images")

# Prepare test data
X_test = np.zeros((len(test_images), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_test = np.zeros((len(test_images), IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)

print("Loading test data...")
for n, img_name in tqdm(enumerate(test_images), total=len(test_images)):
    img_path = os.path.join(TEST_PATH, img_name)
    img = imread(img_path)[:, :, :IMG_CHANNELS]
    X_test[n] = img
    
    mask_path = os.path.join(MASK_TEST_PATH, img_name)
    if os.path.exists(mask_path):
        mask = imread(mask_path, as_gray=True)
        mask = mask / 255.0
        Y_test[n] = np.expand_dims(mask, axis=-1)

print("Test data loaded!")

# Load model
print("\nLoading model...")
model = load_model(MODEL_PATH, compile=False, safe_mode=False)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("Model loaded!")

# Evaluate
print("\n" + "="*70)
print("EVALUATING MODEL ON TEST SET")
print("="*70)

test_loss, test_accuracy = model.evaluate(X_test, Y_test, batch_size=8, verbose=1)

# Get predictions
print("\nGenerating predictions...")
predictions = model.predict(X_test, batch_size=8, verbose=1)
preds_binary = (predictions > 0.5).astype(np.uint8)

# Flatten for metrics
y_true = Y_test.flatten()
y_pred = preds_binary.flatten()
y_pred_proba = predictions.flatten()

# Calculate all metrics
print("\nCalculating metrics...")

# Basic metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

# Segmentation-specific metrics
iou = jaccard_score(y_true, y_pred, zero_division=0)
dice = f1  # Dice is same as F1

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

# Specificity (True Negative Rate)
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

# Display results
print("\n" + "="*70)
print("EVALUATION RESULTS")
print("="*70)
print(f"\n{'Metric':<25} {'Value':<15} {'Percentage'}")
print("-"*70)
print(f"{'Test Loss':<25} {test_loss:<15.4f}")
print(f"{'Pixel Accuracy':<25} {accuracy:<15.4f} {accuracy*100:>6.2f}%")
print(f"{'Precision':<25} {precision:<15.4f} {precision*100:>6.2f}%")
print(f"{'Recall (Sensitivity)':<25} {recall:<15.4f} {recall*100:>6.2f}%")
print(f"{'Specificity':<25} {specificity:<15.4f} {specificity*100:>6.2f}%")
print(f"{'F1 Score':<25} {f1:<15.4f} {f1*100:>6.2f}%")
print(f"{'Dice Coefficient':<25} {dice:<15.4f} {dice*100:>6.2f}%")
print(f"{'IoU (Jaccard)':<25} {iou:<15.4f} {iou*100:>6.2f}%")
print("="*70)

# Confusion Matrix values
print(f"\n{'Confusion Matrix Values:'}")
print(f"  True Positives  (TP): {tp:>12,}")
print(f"  False Positives (FP): {tp:>12,}")
print(f"  True Negatives  (TN): {tn:>12,}")
print(f"  False Negatives (FN): {fn:>12,}")
print("="*70)

# Save metrics to text file
with open(os.path.join(OUTPUT_DIR, 'metrics_summary.txt'), 'w') as f:
    f.write("="*70 + "\n")
    f.write("U-NET SEGMENTATION MODEL - EVALUATION RESULTS\n")
    f.write("="*70 + "\n\n")
    f.write(f"Test Loss:              {test_loss:.4f}\n")
    f.write(f"Pixel Accuracy:         {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    f.write(f"Precision:              {precision:.4f} ({precision*100:.2f}%)\n")
    f.write(f"Recall (Sensitivity):   {recall:.4f} ({recall*100:.2f}%)\n")
    f.write(f"Specificity:            {specificity:.4f} ({specificity*100:.2f}%)\n")
    f.write(f"F1 Score:               {f1:.4f} ({f1*100:.2f}%)\n")
    f.write(f"Dice Coefficient:       {dice:.4f} ({dice*100:.2f}%)\n")
    f.write(f"IoU (Jaccard):          {iou:.4f} ({iou*100:.2f}%)\n\n")
    f.write("Confusion Matrix:\n")
    f.write(f"  TP: {tp:,}\n")
    f.write(f"  FP: {fp:,}\n")
    f.write(f"  TN: {tn:,}\n")
    f.write(f"  FN: {fn:,}\n")

# ==================== VISUALIZATIONS ====================

print("\n Creating visualizations...")

# 1. Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', 
            xticklabels=['Not Plowed', 'Plowed'],
            yticklabels=['Not Plowed', 'Plowed'])
plt.title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150)
plt.close()

# 2. Metrics Bar Chart
metrics_dict = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'Specificity': specificity,
    'F1 Score': f1,
    'Dice': dice,
    'IoU': iou
}

plt.figure(figsize=(10, 6))
bars = plt.bar(metrics_dict.keys(), metrics_dict.values(), color='steelblue')
plt.ylim(0, 1)
plt.ylabel('Score', fontsize=12)
plt.title('Model Performance Metrics', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'metrics_barchart.png'), dpi=150)
plt.close()

# 3. ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curve.png'), dpi=150)
plt.close()

# 4. Precision-Recall Curve
precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
pr_auc = auc(recall_curve, precision_curve)

plt.figure(figsize=(8, 6))
plt.plot(recall_curve, precision_curve, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'precision_recall_curve.png'), dpi=150)
plt.close()

# 5. Sample Predictions with Error Maps
num_samples = min(10, len(test_images))
for i in range(num_samples):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Original, GT, Prediction
    axes[0, 0].imshow(X_test[i])
    axes[0, 0].set_title(f'Image: {test_images[i]}', fontsize=10)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(np.squeeze(Y_test[i]), cmap='gray')
    axes[0, 1].set_title('Ground Truth', fontsize=10)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(np.squeeze(preds_binary[i]), cmap='gray')
    axes[0, 2].set_title('Prediction', fontsize=10)
    axes[0, 2].axis('off')
    
    # Row 2: Overlays
    overlay_gt = X_test[i].copy()
    overlay_gt[np.squeeze(Y_test[i]) == 1] = [0, 255, 0]  # Green
    axes[1, 0].imshow(overlay_gt)
    axes[1, 0].set_title('GT Overlay (Green)', fontsize=10)
    axes[1, 0].axis('off')
    
    overlay_pred = X_test[i].copy()
    overlay_pred[np.squeeze(preds_binary[i]) == 1] = [255, 0, 0]  # Red
    axes[1, 1].imshow(overlay_pred)
    axes[1, 1].set_title('Pred Overlay (Red)', fontsize=10)
    axes[1, 1].axis('off')
    
    # Error map
    error_map = np.zeros((512, 512, 3), dtype=np.uint8)
    gt = np.squeeze(Y_test[i])
    pred = np.squeeze(preds_binary[i])
    
    tp = (gt == 1) & (pred == 1)  # Correct positive
    fp = (gt == 0) & (pred == 1)  # False positive
    fn = (gt == 1) & (pred == 0)  # False negative
    tn = (gt == 0) & (pred == 0)  # Correct negative
    
    error_map[tp] = [255, 255, 255]  # White - Correct
    error_map[fp] = [255, 0, 0]      # Red - False Positive
    error_map[fn] = [0, 0, 255]      # Blue - False Negative
    error_map[tn] = [0, 0, 0]        # Black - Correct
    
    axes[1, 2].imshow(error_map)
    axes[1, 2].set_title('Error Map\n(White=TP, Red=FP, Blue=FN)', fontsize=10)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'sample_{i+1:02d}_{test_images[i]}'), dpi=100, bbox_inches='tight')
    plt.close()

print(f"\nSaved {num_samples} sample predictions")
print(f"\nEvaluation complete!")
print(f"All results saved to: {OUTPUT_DIR}")
print("\nGenerated files:")
print("  - metrics_summary.txt")
print("  - confusion_matrix.png")
print("  - metrics_barchart.png")
print("  - roc_curve.png")
print("  - precision_recall_curve.png")
print(f"  - {num_samples} sample prediction images")
print("="*70)
