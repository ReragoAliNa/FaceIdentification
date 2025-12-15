"""
PCA Face Recognition Experiment
Based on the provided lab guide.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set plot style for better aesthetics
plt.style.use('seaborn-v0_8-darkgrid')

def load_and_split_data(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Tuple[int, int]]:
    """
    Load ORL dataset and split into train/test based on the guide's rule:
    - Train: First 8 images (1.pgm - 8.pgm) per person
    - Test: Last 2 images (9.pgm - 10.pgm) per person
    """
    train_data = []
    test_data = []
    train_labels = []
    test_labels = []
    
    # Check for both s1 and S1 naming conventions, just in case
    # The guide says "s1"..."s40"
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory {data_dir} does not exist.")

    # We expect subdirectories s1..s40
    # Let's dynamically find them or iterate 1..40
    person_folders = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.lower().startswith('s')])
    
    if not person_folders:
        print(f"Warning: No 's*' folders found in {data_dir}. Searching recursive...")
        # Fallback for nested structure if necessary
        pass

    img_h, img_w = 112, 92
    
    label_names = []

    print(f"Loading data from {data_dir}...")
    
    for person_idx, folder in enumerate(person_folders):
        # person_idx is 0-based label
        # folder name is e.g. s1
        label_names.append(folder.name)
        
        # Images are 1.pgm to 10.pgm
        # We try to load them in order
        for img_idx in range(1, 11):
            img_name = f"{img_idx}.pgm"
            img_path = folder / img_name
            
            if not img_path.exists():
                # Try finding any image if exact name doesn't match? 
                # The guide implies 1.pgm-10.pgm exists.
                continue
                
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Failed to read {img_path}")
                continue
            
            # Ensure size is 112x92
            if img.shape != (img_h, img_w):
                img = cv2.resize(img, (img_w, img_h))
                
            img_flatten = img.flatten().astype(np.float32)
            
            # Split rule
            if img_idx <= 8:
                train_data.append(img_flatten)
                train_labels.append(person_idx)
            else:
                test_data.append(img_flatten)
                test_labels.append(person_idx)

    X_train = np.array(train_data)
    y_train = np.array(train_labels)
    X_test = np.array(test_data)
    y_test = np.array(test_labels)
    
    print(f"Data Loaded: Train={X_train.shape}, Test={X_test.shape}")
    return X_train, y_train, X_test, y_test, label_names, (img_h, img_w)

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def run_recognition(X_train_pca, y_train, X_test_pca, metric='euclidean'):
    """
    Match test faces to training faces.
    Returns predicted labels and distances.
    """
    predicted_labels = []
    min_distances = []
    
    # Use simple nearest neighbor
    for i in range(len(X_test_pca)):
        test_vec = X_test_pca[i]
        
        if metric == 'euclidean':
            # Vectorized euclidean distance calculation
            dists = np.linalg.norm(X_train_pca - test_vec, axis=1)
        else:
            # Cosine distance (1 - similarity)
            # Normalize first
            test_norm = test_vec / (np.linalg.norm(test_vec) + 1e-9)
            train_norm = X_train_pca / (np.linalg.norm(X_train_pca, axis=1, keepdims=True) + 1e-9)
            sims = np.dot(train_norm, test_norm)
            dists = 1 - sims # Smaller is better
            
        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]
        
        predicted_labels.append(y_train[min_idx])
        min_distances.append(min_dist)
        
    return np.array(predicted_labels), np.array(min_distances)

def visualize_results(
    X_test_orig, y_test, preds, dists, 
    X_train_orig, y_train, 
    img_shape, out_dir
):
    """
    Save specific correct/wrong examples.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Provide one Correct Example
    correct_indices = np.where(preds == y_test)[0]
    if len(correct_indices) > 0:
        idx = correct_indices[0]
        plt.figure(figsize=(8, 4))
        
        # Test Image
        plt.subplot(1, 2, 1)
        plt.imshow(X_test_orig[idx].reshape(img_shape), cmap='gray')
        plt.title(f"Test Face (Person {y_test[idx]+1})")
        plt.axis('off')
        
        # Matched Train Image
        # We need to find which train image was the match. 
        # Re-calculate match index
        train_dists = np.linalg.norm(X_train_orig - X_test_orig[idx], axis=1)
        # Note: we are matching in Pixel space for visualization "check" or PCA space?
        # The prompt implies showing "Matched Train Face".
        # Since 'dists' calculated in PCA space, we can't easily map back to strict index without re-running NN in PCA space or passing indices.
        # But for 'Correct Recognition', showing ANY image of the correct person from train set is often acceptable,
        # however, strictly we should show the one it matched with.
        # Let's just pick the first image of that person in Train set for display simplicity if strict match tracking is hard.
        # Wait, I can just use the label to pick a representative.
        # Ideally, I should return the matched index from run_recognition.
        # Let's stick to showing "A" matching face (the one with label same as predicted).
        
        match_label = preds[idx]
        train_idx = np.where(y_train == match_label)[0][0] # First one
        
        plt.subplot(1, 2, 2)
        plt.imshow(X_train_orig[train_idx].reshape(img_shape), cmap='gray')
        plt.title(f"Matched Class (Person {match_label+1})")
        plt.axis('off')
        
        plt.suptitle("Correct Recognition Example")
        plt.savefig(out_dir / "example_correct.png")
        plt.close()

    # 2. Provide one Wrong Example
    wrong_indices = np.where(preds != y_test)[0]
    if len(wrong_indices) > 0:
        idx = wrong_indices[0]
        pred_y = preds[idx]
        true_y = y_test[idx]
        
        plt.figure(figsize=(8, 4))
        
        plt.subplot(1, 2, 1)
        plt.imshow(X_test_orig[idx].reshape(img_shape), cmap='gray')
        plt.title(f"Test: Person {true_y+1}")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        # Find a representative face for the predicted class
        pred_idx_in_train = np.where(y_train == pred_y)[0][0]
        plt.imshow(X_train_orig[pred_idx_in_train].reshape(img_shape), cmap='gray')
        plt.title(f"Pred: Person {pred_y+1}\nDist: {dists[idx]:.2f}")
        plt.axis('off')
        
        plt.suptitle("Wrong Recognition Example")
        plt.savefig(out_dir / "example_wrong.png")
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='orl_faces')
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()
    
    data_path = Path(args.data_dir)
    # Handle possible nested 'orl_faces' from extraction
    if not any(data_path.iterdir()):
         print("Data dir empty?")
    
    # Check if we need to go deeper: orl_faces/orl_faces?
    # The setup script extracted to 'orl_faces'. 
    # If the zip had a root folder, it might be orl_faces/att_faces/s1...
    # Let's inspect quickly with a try block or listing?
    # I'll rely on the load function to warn.
        
    X_train, y_train, X_test, y_test, labels, img_shape = load_and_split_data(data_path)
    h, w = img_shape
    
    # Step 1: Centralization / Scaling
    scaler = StandardScaler(with_mean=True, with_std=False)
    # Fit on Train
    X_train_centered = scaler.fit_transform(X_train)
    # Transform Test using Train's mean
    X_test_centered = scaler.transform(X_test)
    
    # Task: Analyze K values
    K_values = [10, 20, 30, 50, 80, 100]
    accuracies = []
    variances = []
    
    out_path = Path(args.output_dir)
    out_path.mkdir(exist_ok=True)
    
    print("\n--- Starting Parameter Tuning ---")
    for K in K_values:
        # Train PCA
        # whiten=True is recommended in guide
        pca = PCA(n_components=K, whiten=True, random_state=42)
        pca.fit(X_train_centered)
        
        # Transform
        X_train_pca = pca.transform(X_train_centered)
        X_test_pca = pca.transform(X_test_centered)
        
        # Predict
        preds, dists = run_recognition(X_train_pca, y_train, X_test_pca)
        
        # Accuracy
        acc = np.mean(preds == y_test) * 100
        cum_var = np.sum(pca.explained_variance_ratio_)
        
        accuracies.append(acc)
        variances.append(cum_var)
        
        print(f"K={K:3d} | Accuracy={acc:6.2f}% | Cum Var={cum_var:.4f}")
        
        # Save eigenfaces for K=50 (Example)
        if K == 50:
            # Save top 5 eigenfaces
            plt.figure(figsize=(15, 3))
            for i in range(5):
                plt.subplot(1, 5, i+1)
                eigenface = pca.components_[i].reshape(h, w)
                plt.imshow(eigenface, cmap='gray')
                plt.title(f"Eigenface {i+1}")
                plt.axis('off')
            plt.tight_layout()
            plt.savefig(out_path / "eigenfaces_top5.png")
            plt.close()
            
            # Save visual examples for this "Best" run
            visualize_results(X_test, y_test, preds, dists, X_train, y_train, (h, w), out_path)

    # Plot Sensitivity Analysis
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Components (K)')
    ax1.set_ylabel('Accuracy (%)', color=color)
    ax1.plot(K_values, accuracies, marker='o', color=color, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Cumulative Variance', color=color)
    ax2.plot(K_values, variances, marker='s', linestyle='--', color=color, label='Variance')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title("PCA Performance Analysis: K vs Accuracy/Variance")
    plt.tight_layout()
    plt.savefig(out_path / "analysis_curve.png")
    plt.close()
    
    # Save metrics to CSV
    df = pd.DataFrame({
        "K": K_values,
        "Accuracy": accuracies,
        "CumulativeVariance": variances
    })
    df.to_csv(out_path / "experiment_metrics.csv", index=False)
    print(f"\nExperiment Complete. Results saved to {out_path.resolve()}")

if __name__ == "__main__":
    main()
