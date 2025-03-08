import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report

def save_test_results(test_metrics, model_info, training_params, output_dir, exp_name):
    """
    Save comprehensive test results to files.
    
    Args:
        test_metrics (dict): Dictionary containing test metrics
        model_info (dict): Dictionary containing model information
        training_params (dict): Dictionary containing training parameters
        output_dir (str): Directory to save results
        exp_name (str): Experiment name
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary results
    summary_path = os.path.join(output_dir, f"{exp_name}_test_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=== TEST RESULTS SUMMARY ===\n\n")
        f.write(f"Experiment: {exp_name}\n")
        f.write(f"Date: {test_metrics.get('date', 'N/A')}\n\n")
        
        f.write("=== MODEL INFORMATION ===\n")
        for key, value in model_info.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("=== TRAINING PARAMETERS ===\n")
        for key, value in training_params.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("=== TEST METRICS ===\n")
        f.write(f"Accuracy: {test_metrics.get('accuracy', 0):.2f}%\n")
        f.write(f"Precision: {test_metrics.get('precision', 0):.4f}\n")
        f.write(f"Recall: {test_metrics.get('recall', 0):.4f}\n")
        f.write(f"F1 Score: {test_metrics.get('f1', 0):.4f}\n")
        f.write(f"Loss: {test_metrics.get('loss', 0):.6f}\n")
        f.write(f"Samples: {test_metrics.get('samples', 0)}\n")
        f.write(f"Batches: {test_metrics.get('batches', 0)}\n")
        
        if test_metrics.get('is_training_data', False):
            f.write("\nNote: Testing was performed on training data\n")
    
    print(f"Test summary saved to {summary_path}")
    
    # Save detailed per-class metrics if available
    if 'per_class_metrics' in test_metrics:
        per_class_path = os.path.join(output_dir, f"{exp_name}_per_class_metrics.txt")
        with open(per_class_path, "w") as f:
            f.write("=== PER-CLASS METRICS ===\n\n")
            f.write("Class\tPrecision\tRecall\tF1\tSupport\n")
            
            per_class = test_metrics['per_class_metrics']
            for i in range(len(per_class['precision'])):
                f.write(f"{i}\t{per_class['precision'][i]:.4f}\t{per_class['recall'][i]:.4f}\t")
                f.write(f"{per_class['f1'][i]:.4f}\t{per_class['support'][i]}\n")
        
        print(f"Per-class metrics saved to {per_class_path}")
    
    # Save confusion matrix if available
    if 'confusion_matrix' in test_metrics:
        cm_path = os.path.join(output_dir, f"{exp_name}_confusion_matrix.txt")
        with open(cm_path, "w") as f:
            f.write("=== CONFUSION MATRIX ===\n\n")
            cm = test_metrics['confusion_matrix']
            for row in cm:
                f.write("\t".join([str(x) for x in row]) + "\n")
        
        print(f"Confusion matrix saved to {cm_path}")
        
        # Also save as CSV for easier processing
        cm_csv_path = os.path.join(output_dir, f"{exp_name}_confusion_matrix.csv")
        pd.DataFrame(cm).to_csv(cm_csv_path, index=False)
        print(f"Confusion matrix CSV saved to {cm_csv_path}")
    
    # Save raw predictions if available
    if 'predictions' in test_metrics and 'targets' in test_metrics:
        try:
            # Convert to numpy arrays if they're not already
            predictions = np.array(test_metrics['predictions'])
            targets = np.array(test_metrics['targets'])
            
            # Save to CSV
            pred_path = os.path.join(output_dir, f"{exp_name}_predictions.csv")
            pd.DataFrame({
                'target': targets.flatten(),
                'prediction': predictions.flatten()
            }).to_csv(pred_path, index=False)
            
            print(f"Raw predictions saved to {pred_path}")
        except Exception as e:
            print(f"Error saving raw predictions: {e}")

def process_test_results(predictions, targets, threshold=0.5):
    """
    Process test predictions and targets to calculate metrics.
    
    Args:
        predictions: Model predictions (logits or probabilities)
        targets: Ground truth labels
        threshold: Classification threshold
    
    Returns:
        dict: Dictionary containing calculated metrics
    """
    # Convert to binary predictions if needed
    if hasattr(predictions, 'sigmoid'):
        pred_probs = predictions.sigmoid().numpy()
    elif hasattr(predictions, 'numpy'):
        pred_probs = predictions.numpy()
    else:
        pred_probs = np.array(predictions)
    
    if hasattr(targets, 'numpy'):
        targets_np = targets.numpy()
    else:
        targets_np = np.array(targets)
    
    # Convert to binary predictions
    pred_binary = (pred_probs > threshold).astype(float)
    
    # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        targets_np, pred_binary, average='weighted', zero_division=0
    )
    
    # Calculate per-class metrics
    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        targets_np, pred_binary, average=None, zero_division=0
    )
    
    # Calculate confusion matrix
    cm = confusion_matrix(targets_np.argmax(axis=1) if targets_np.ndim > 1 else targets_np, 
                          pred_binary.argmax(axis=1) if pred_binary.ndim > 1 else pred_binary)
    
    # Calculate accuracy
    accuracy = (pred_binary == targets_np).mean() * 100
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'per_class_metrics': {
            'precision': per_class_precision,
            'recall': per_class_recall,
            'f1': per_class_f1,
            'support': per_class_support
        },
        'confusion_matrix': cm,
        'predictions': pred_probs,
        'targets': targets_np
    }

def plot_test_results(test_metrics, output_dir, exp_name):
    """
    Create and save visualizations of test results.
    
    Args:
        test_metrics (dict): Dictionary containing test metrics
        output_dir (str): Directory to save results
        exp_name (str): Experiment name
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot confusion matrix if available
    if 'confusion_matrix' in test_metrics:
        plt.figure(figsize=(10, 8))
        cm = test_metrics['confusion_matrix']
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        # Add labels
        classes = range(cm.shape[0])
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        
        # Save figure
        cm_fig_path = os.path.join(output_dir, f"{exp_name}_confusion_matrix.png")
        plt.savefig(cm_fig_path, dpi=300)
        plt.close()
        print(f"Confusion matrix visualization saved to {cm_fig_path}")
    
    # Plot per-class metrics if available
    if 'per_class_metrics' in test_metrics:
        plt.figure(figsize=(12, 6))
        per_class = test_metrics['per_class_metrics']
        classes = range(len(per_class['precision']))
        
        width = 0.2
        x = np.arange(len(classes))
        
        plt.bar(x - width, per_class['precision'], width, label='Precision')
        plt.bar(x, per_class['recall'], width, label='Recall')
        plt.bar(x + width, per_class['f1'], width, label='F1')
        
        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.title('Per-Class Metrics')
        plt.xticks(x, classes)
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        metrics_fig_path = os.path.join(output_dir, f"{exp_name}_per_class_metrics.png")
        plt.savefig(metrics_fig_path, dpi=300)
        plt.close()
        print(f"Per-class metrics visualization saved to {metrics_fig_path}")

if __name__ == "__main__":
    # Example usage
    print("This module provides functions to process and save test results.")
    print("Import and use the functions in your training script.") 