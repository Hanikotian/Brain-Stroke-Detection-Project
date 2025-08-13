import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Fixing TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ----- Configuration -----
class Config:
    BASE_DIR = r"C:/Users/hanik/OneDrive/Documents/college assignments/Brain Stroke Project"
    TRAIN_DIR = os.path.join(BASE_DIR, "Dataset_labelled")
    TEST_DIR = os.path.join(BASE_DIR, "Test_dataset")
    IMG_SIZE = 224  # Increased for better feature extraction
    BATCH_SIZE = 16  # Reduced for better gradient updates
    EPOCHS = 30
    LEARNING_RATE = 0.0001
    
config = Config()

def load_and_preprocess_images(folder_path, label, max_images=None):
    """Enhanced image loading with better preprocessing"""
    images = []
    labels = []
    print(f"Loading images from {folder_path}...")
    
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if max_images:
        files = files[:max_images]
    
    for i, filename in enumerate(files):
        if i % 500 == 0:
            print(f"Loaded {i}/{len(files)} images...")
            
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        
        if img is not None:
            # Resize image
            img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Enhanced preprocessing for medical images
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img_gray = clahe.apply(img_gray)
            img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
            
            images.append(img)
            labels.append(label)
    
    print(f"Successfully loaded {len(images)} images with label {label}")
    return images, labels

def create_medical_cnn_model(input_shape, num_classes=1):
    """Create a medical-focused CNN model"""
    model = Sequential([
        # First block - focus on edge detection
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Second block - texture patterns
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Third block - complex features
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Global average pooling instead of flatten
        GlobalAveragePooling2D(),
        
        # Classification layers
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='sigmoid')
    ])
    
    return model

def create_transfer_learning_model(input_shape, num_classes=1):
    """Create a transfer learning model using pre-trained ResNet50"""
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze base model initially
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='sigmoid')
    ])
    
    return model, base_model

def evaluate_medical_model(model, X_val, y_val, threshold=0.5):
    """Enhanced evaluation for medical models"""
    predictions = model.predict(X_val, verbose=0)
    y_pred = (predictions > threshold).astype(int).flatten()
    
    # Calculate metrics
    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)  # Recall for stroke detection
    specificity = tn / (tn + fp)  # Recall for normal detection
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    print(f"\n{'='*60}")
    print("DETAILED MEDICAL EVALUATION")
    print(f"{'='*60}")
    print(f"Threshold: {threshold}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Sensitivity (Stroke Detection Rate): {sensitivity:.3f} ⭐")
    print(f"Specificity (Normal Detection Rate): {specificity:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Negative Predictive Value: {npv:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print()
    print("Clinical Interpretation:")
    print(f"• {sensitivity*100:.1f}% of stroke cases are correctly identified")
    print(f"• {fn} stroke cases would be MISSED (FALSE NEGATIVES) ⚠️")
    print(f"• {fp} normal cases would be incorrectly flagged (FALSE POSITIVES)")
    print(f"• {specificity*100:.1f}% of normal cases are correctly identified")
    
    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'npv': npv,
        'f1': f1,
        'cm': cm,
        'predictions': predictions
    }

def find_optimal_threshold(model, X_val, y_val):
    """Find optimal threshold for medical diagnosis"""
    predictions = model.predict(X_val, verbose=0)
    
    thresholds = np.arange(0.1, 0.9, 0.05)
    results = []
    
    for threshold in thresholds:
        y_pred = (predictions > threshold).astype(int).flatten()
        cm = confusion_matrix(y_val, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        
        # Medical score prioritizing sensitivity (stroke detection)
        medical_score = 0.7 * sensitivity + 0.3 * specificity
        
        results.append({
            'threshold': threshold,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'f1': f1,
            'medical_score': medical_score,
            'missed_strokes': fn
        })
    
    # Find best threshold
    best_result = max(results, key=lambda x: x['medical_score'])
    
    print(f"\nOptimal Threshold Analysis:")
    print(f"Best threshold: {best_result['threshold']:.3f}")
    print(f"Sensitivity: {best_result['sensitivity']:.3f}")
    print(f"Specificity: {best_result['specificity']:.3f}")
    print(f"Missed strokes: {best_result['missed_strokes']}")
    
    return best_result['threshold'], results

def main():
    """Main training pipeline with medical focus"""
    
    print("="*60)
    print("MEDICAL BRAIN STROKE DETECTION PIPELINE")
    print("="*60)
    
    # Load data
    normal_path = os.path.join(config.TRAIN_DIR, "Normal")
    stroke_path = os.path.join(config.TRAIN_DIR, "Stroke")
    
    print("Loading training data with enhanced preprocessing...")
    normal_imgs, normal_labels = load_and_preprocess_images(normal_path, 0)
    stroke_imgs, stroke_labels = load_and_preprocess_images(stroke_path, 1)
    
    # Combine and prepare data
    X = np.array(normal_imgs + stroke_imgs)
    y = np.array(normal_labels + stroke_labels)
    X = X.astype("float32") / 255.0
    
    print(f"\nDataset Summary:")
    print(f"Normal images: {len(normal_imgs)}")
    print(f"Stroke images: {len(stroke_imgs)}")
    print(f"Total images: {len(X)}")
    print(f"Image shape: {X.shape}")
    
    # Calculate class weights with medical focus
    # Increase weight for stroke class to reduce false negatives
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1] * 1.5}  # Extra weight for stroke
    print(f"Adjusted class weights: {class_weight_dict}")
    
    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"\nData split:")
    print(f"Training: {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples")
    
    # Data augmentation (minimal for medical images)
    datagen = ImageDataGenerator(
        rotation_range=10,  # Reduced rotation
        width_shift_range=0.05,  # Reduced shifting
        height_shift_range=0.05,
        zoom_range=0.05,  # Reduced zoom
        horizontal_flip=False,  # No flipping for brain images
        fill_mode='nearest'
    )
    
    # Model selection
    print(f"\nBuilding models...")
    
    # Model 1: Custom CNN
    model_cnn = create_medical_cnn_model((config.IMG_SIZE, config.IMG_SIZE, 3))
    model_cnn.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    # Model 2: Transfer Learning
    model_transfer, base_model = create_transfer_learning_model((config.IMG_SIZE, config.IMG_SIZE, 3))
    model_transfer.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_recall',  # Monitor recall (sensitivity) for medical
        patience=10,
        mode='max',
        restore_best_weights=True,
        verbose=1
    )
    
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    # Train models
    models = {
        'Custom CNN': model_cnn,
        'Transfer Learning': model_transfer
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"TRAINING {name.upper()}")
        print(f"{'='*60}")
        
        # Train
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=config.BATCH_SIZE),
            steps_per_epoch=len(X_train) // config.BATCH_SIZE,
            epochs=config.EPOCHS,
            validation_data=(X_val, y_val),
            class_weight=class_weight_dict,
            callbacks=[early_stopping, lr_scheduler],
            verbose=1
        )
        
        # Evaluate
        print(f"\nEvaluating {name}...")
        result = evaluate_medical_model(model, X_val, y_val)
        
        # Find optimal threshold
        optimal_threshold, threshold_results = find_optimal_threshold(model, X_val, y_val)
        
        # Re-evaluate with optimal threshold
        result_optimal = evaluate_medical_model(model, X_val, y_val, optimal_threshold)
        
        results[name] = {
            'model': model,
            'history': history,
            'result_default': result,
            'result_optimal': result_optimal,
            'optimal_threshold': optimal_threshold
        }
    
    # Compare models
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    
    for name, data in results.items():
        opt_result = data['result_optimal']
        print(f"\n{name} (Optimal Threshold: {data['optimal_threshold']:.3f}):")
        print(f"  Sensitivity: {opt_result['sensitivity']:.3f}")
        print(f"  Specificity: {opt_result['specificity']:.3f}")
        print(f"  F1-Score: {opt_result['f1']:.3f}")
        print(f"  Accuracy: {opt_result['accuracy']:.3f}")
    
    # Select best model based on sensitivity (most important for medical)
    best_model_name = max(results.keys(), 
                         key=lambda x: results[x]['result_optimal']['sensitivity'])
    best_model_data = results[best_model_name]
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   Optimal for medical diagnosis with {best_model_data['result_optimal']['sensitivity']:.1%} sensitivity")
    
    # Save best model
    model_path = os.path.join(config.BASE_DIR, f"best_medical_model_{best_model_name.replace(' ', '_').lower()}.h5")
    best_model_data['model'].save(model_path)
    print(f"   Saved to: {model_path}")
    
    # Visualize best model results
    best_result = best_model_data['result_optimal']
    cm = best_result['cm']
    
    plt.figure(figsize=(10, 4))
    
    # Confusion matrix
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=['Normal', 'Stroke'], 
                yticklabels=['Normal', 'Stroke'])
    plt.title(f'{best_model_name} - Confusion Matrix\n(Threshold: {best_model_data["optimal_threshold"]:.3f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Sensitivity vs Specificity
    plt.subplot(1, 2, 2)
    threshold_data = find_optimal_threshold(best_model_data['model'], X_val, y_val)[1]
    thresholds = [r['threshold'] for r in threshold_data]
    sensitivities = [r['sensitivity'] for r in threshold_data]
    specificities = [r['specificity'] for r in threshold_data]
    
    plt.plot(thresholds, sensitivities, 'r-', label='Sensitivity', linewidth=2)
    plt.plot(thresholds, specificities, 'b-', label='Specificity', linewidth=2)
    plt.axvline(x=best_model_data['optimal_threshold'], color='green', 
                linestyle='--', label='Optimal Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Sensitivity vs Specificity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return best_model_data['model'], best_model_data['optimal_threshold']

if __name__ == "__main__":
    best_model, optimal_threshold = main()
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETED!")
    print(f"{'='*60}")
    print("Key Recommendations for Medical Use:")
    print(f"1. Use threshold: {optimal_threshold:.3f} for predictions")
    print("2. Always have radiologist review positive cases")
    print("3. Consider ensemble methods for critical diagnoses")
    print("4. Regularly validate on new data")
