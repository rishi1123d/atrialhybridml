!pip install wfdb
!wget -r -N -c -np https://physionet.org/files/afdb/1.0.0/
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import wfdb

record_base_path = './physionet.org/files/afdb/1.0.0/'
files = [
    '04015', '04043', '04048', '04126', '04746', '04908', '04936',
    '05091', '05121', '05261', '06426', '06453', '06995', '07162',
    '07859', '07879', '07910', '08215', '08219', '08378', '08405',
    '08434', '08455'
]

def process_single_ecg(file_base_path, segment_length=5000):
    try:
        record = wfdb.rdrecord(file_base_path)
        signal = record.p_signal[:, 0]
        fs = record.fs

        segments = [signal[i:i + segment_length] for i in range(0, len(signal) - segment_length, segment_length)]
        return np.array(segments)

    except Exception as e:
        print(f"Error processing ECG data from {file_base_path}: {e}")
        return None

def load_ecg_dataset(files, segment_length=5000):
    X = []
    y = []

    for file in files:
        file_base_path = os.path.join(record_base_path, file)

        if not os.path.exists(f"{file_base_path}.dat") or not os.path.exists(f"{file_base_path}.hea"):
            print(f"Missing corresponding .hea or .dat file for {file_base_path}. Skipping this file.")
            continue

        segments = process_single_ecg(file_base_path, segment_length=segment_length)
        if segments is not None:
            X.append(segments)
            label = 0 if 'normal' in file else 1
            y.append(np.full(segments.shape[0], label))

    if not X:
        raise ValueError("No valid data to process. Ensure .dat and .hea files are present and correct.")

    X = np.vstack(X)
    y = np.concatenate(y)

    return X, y

def augment_ecg_data(X):
    noise_factor = 0.05
    noise = np.random.normal(0, noise_factor, X.shape)

    scaling_factor = np.random.uniform(0.9, 1.1, X.shape)
    shifted = np.roll(X, shift=5, axis=1)

    augmented_data = X + noise
    augmented_data *= scaling_factor
    augmented_data += shifted

    return augmented_data

def plot_training_history(history, fold):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label=f'Training accuracy - Fold {fold}')
    plt.plot(epochs, val_acc, 'r', label=f'Validation accuracy - Fold {fold}')
    plt.title(f'Training and validation accuracy - Fold {fold}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label=f'Training loss - Fold {fold}')
    plt.plot(epochs, val_loss, 'r', label=f'Validation loss - Fold {fold}')
    plt.title(f'Training and validation loss - Fold {fold}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

def evaluate_model_performance(y_test, y_pred, fold):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Disease'], yticklabels=['Normal', 'Disease'])
    plt.title(f'Confusion Matrix - Fold {fold}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    if len(np.unique(y_test)) > 1 and len(np.unique(y_pred)) > 1:
        print(f"Classification Report - Fold {fold}:")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Disease']))

def plot_roc_curve(y_test, y_pred_prob, fold):
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f}) - Fold {fold}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Fold {fold}')
    plt.legend(loc="lower right")
    plt.show()

def build_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def main():
    X, y = load_ecg_dataset(files)

    X = X.reshape(X.shape[0], X.shape[1], 1)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_no = 1

    for train_index, test_index in kfold.split(X):
        print(f"Training Fold {fold_no}...")

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train_augmented = augment_ecg_data(X_train)

        input_shape = (X_train.shape[1], 1)
        cnn_model = build_cnn_model(input_shape)

        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

        history = cnn_model.fit(
            X_train_augmented,
            y_train,
            epochs=15,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[lr_scheduler, early_stopping],
            verbose=1
        )


        y_pred = (cnn_model.predict(X_test) > 0.5).astype("int32")
        y_pred_prob = cnn_model.predict(X_test).ravel()

        plot_training_history(history, fold_no)
        evaluate_model_performance(y_test, y_pred, fold_no)
        plot_roc_curve(y_test, y_pred_prob, fold_no)

        fold_no += 1

if __name__ == "__main__":
    main()

