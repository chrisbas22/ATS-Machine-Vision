import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import copy

from skimage.feature import hog
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed


# ---------------- CONFIGURATION 
# Input paths 
CSV_PATH = "emnist-letters-train.csv"  # prefer CSV if available
IDX_IMAGES = "emnist-letters-train-images-idx3-ubyte"
IDX_LABELS = "emnist-letters-train-labels-idx1-ubyte"

# Sampling config
CLASSES = 26
SAMPLES_PER_CLASS = 500  # 26 * 500 = 13000

# HOG params 
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = 'L2-Hys'

# SVM params (LinearSVC used for FAST_MODE)
SVM_C = 1.0

# Speed / memory options
FAST_MODE = True          # True: use LinearSVC (recommended for laptop). False: you'd need to change to SVC and expect heavy workload.
N_JOBS_CV = -1            # cross_val_predict n_jobs: -1 (all cores). scikit-learn may parallelize CV internal loops.

# Caching & output
FEATURE_CACHE = "hog_features_13000_joblib.pkl"
OUT_PRED_CSV = "predictions_loocv.csv"
METRICS_JSON = "metrics_loocv.json"
CM_COUNTS_PNG = "confusion_matrix_counts.png"
CM_NORM_PNG = "confusion_matrix_norm.png"

RANDOM_STATE = 42
# ----------------------------------------------------------------

def read_idx_images_labels(images_path, labels_path):
    """
    Read IDX ubyte files (MNIST/EMNIST style).
    Returns: images (N,28,28), labels (N,)
    """
    def _read_idx(filename):
        with open(filename, 'rb') as f:
            data = f.read()
        return data

    # images
    import struct
    with open(images_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape((num, rows, cols))
    with open(labels_path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return images, labels

def load_emnist_from_csv(csv_path, samples_per_class=500, seed=RANDOM_STATE):
    """
    Load EMNIST CSV where first column label, remaining 784 pixels.
    Returns images shaped (N,28,28) and labels (N,)
    """
    print("Loading CSV:", csv_path)
    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] != 785:
        raise ValueError(f"CSV has {df.shape[1]} columns (expected 785).")
    df.columns = ['label'] + [f'pixel{i}' for i in range(784)]
    labels_unique = sorted(df['label'].unique())
    if len(labels_unique) < CLASSES:
        raise ValueError(f"Found only {len(labels_unique)} classes in CSV, expected {CLASSES}.")
    rng = np.random.RandomState(seed)
    parts = []
    for cls in labels_unique:
        dfc = df[df['label'] == cls]
        if len(dfc) < samples_per_class:
            raise ValueError(f"Not enough samples for class {cls}: found {len(dfc)} < {samples_per_class}")
        sampled = dfc.sample(n=samples_per_class, random_state=rng)
        parts.append(sampled)
    df_sampled = pd.concat(parts).sample(frac=1.0, random_state=rng).reset_index(drop=True)
    X = df_sampled.iloc[:, 1:].values.astype(np.uint8)
    y = df_sampled.iloc[:, 0].values.astype(int)
    # reshape to images and apply re-orientation used for EMNIST
    imgs = []
    for i in range(X.shape[0]):
        img = X[i].reshape((28, 28))
        img = np.transpose(img)[:, ::-1]  # orientasi 
        imgs.append(img)
    imgs = np.stack(imgs, axis=0)
    return imgs, y

def load_emnist_balanced(samples_per_class=SAMPLES_PER_CLASS):
    """
    Attempt to load CSV; if not found, try IDX files.
    """
    if os.path.exists(CSV_PATH):
        return load_emnist_from_csv(CSV_PATH, samples_per_class)
    else:
        # try idx files
        if os.path.exists(IDX_IMAGES) and os.path.exists(IDX_LABELS):
            print("Loading IDX files...")
            imgs_all, labels_all = read_idx_images_labels(IDX_IMAGES, IDX_LABELS)
            # EMNIST uses labels 1..26 for letters often; we simply sample by unique label values present
            rng = np.random.RandomState(RANDOM_STATE)
            parts_imgs = []
            parts_lbls = []
            for cls in np.unique(labels_all):
                idxs = np.where(labels_all == cls)[0]
                if len(idxs) < samples_per_class:
                    raise ValueError(f"Class {cls} has only {len(idxs)} samples < {samples_per_class}")
                chosen = rng.choice(idxs, size=samples_per_class, replace=False)
                parts_imgs.append(imgs_all[chosen])
                parts_lbls.append(labels_all[chosen])
            X = np.vstack(parts_imgs)
            y = np.concatenate(parts_lbls)
            # shuffle
            perm = rng.permutation(len(y))
            return X[perm], y[perm]
        else:
            raise FileNotFoundError("Neither CSV nor IDX files found. Place emnist-letters-train.csv or IDX files in working dir.")

def extract_hog_features(X_images,
                         orientations=HOG_ORIENTATIONS,
                         pixels_per_cell=HOG_PIXELS_PER_CELL,
                         cells_per_block=HOG_CELLS_PER_BLOCK,
                         block_norm=HOG_BLOCK_NORM,
                         cache_file=FEATURE_CACHE):

    if cache_file and os.path.exists(cache_file):
        print("Loading cached HOG features from:", cache_file)
        data = joblib.load(cache_file)
        feats = data['features']
        return feats

    print("Extracting HOG features (parallel)... total =", X_images.shape[0])

    def hog_single(img):
        return hog(img,
                   orientations=orientations,
                   pixels_per_cell=pixels_per_cell,
                   cells_per_block=cells_per_block,
                   block_norm=block_norm,
                   feature_vector=True)

    feats = Parallel(n_jobs=-1, verbose=1)(
        delayed(hog_single)(img) for img in X_images
    )

    feats = np.asarray(feats)
    joblib.dump({'features': feats}, cache_file)
    print("Saved HOG cache to", cache_file)
    return feats

def run_loocv_sgd(X_feats, y):
    base_clf = SGDClassifier(loss="hinge", max_iter=1, tol=None, random_state=RANDOM_STATE)

    # init classes
    base_clf.partial_fit(X_feats[:1], y[:1], classes=np.unique(y))

    y_pred = np.zeros_like(y)
    total = len(y)
    start = time.time()

    for i in range(total):
        test_x = X_feats[i].reshape(1,-1)
        test_y = y[i]

        # create clone model
        clf = copy.deepcopy(base_clf)

        # train excluding i
        idx = np.arange(total)!=i
        clf.partial_fit(X_feats[idx], y[idx])

        y_pred[i] = clf.predict(test_x)

        if i%200==0:
            el=time.time()-start
            eta=(el/(i+1))*(total-(i+1))
            print(f"[{i}/{total}] elapsed {el:.1f}s ETA {eta:.1f}s")

    cm = confusion_matrix(y, y_pred)
    acc = accuracy_score(y,y_pred)
    prec = precision_score(y,y_pred, average="macro",zero_division=0)
    f1 = f1_score(y,y_pred,average="macro",zero_division=0)

    return y_pred, cm, {"accuracy":acc,"precision_macro":prec,"f1_macro":f1}

def plot_confusion_matrix(cm, classes, filename, normalize=False):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10,8))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else " (Counts)"))
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i,j])}",
                     horizontalalignment="center")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def main():
    print("=== EMNIST HOG + SVM (LOOCV) — FAST MODE for laptop ===")

    global_start = time.time()

    # --- TIMER stage 1 dataset load ---
    t0 = time.time()
    X_imgs, y = load_emnist_balanced(SAMPLES_PER_CLASS)
    t1 = time.time()
    print("Stage1 : Dataset Load + Sampling 13000 =", round(t1 - t0, 2), "detik")
    print("Loaded images:", X_imgs.shape, "labels:", y.shape)

    # --- TIMER stage 2 HOG ---
    t2 = time.time()
    X_feats = extract_hog_features(X_imgs,
                                   orientations=HOG_ORIENTATIONS,
                                   pixels_per_cell=HOG_PIXELS_PER_CELL,
                                   cells_per_block=HOG_CELLS_PER_BLOCK,
                                   block_norm=HOG_BLOCK_NORM,
                                   cache_file=FEATURE_CACHE)
    t3 = time.time()
    print("Stage2 : HOG Feature Extraction =", round(t3 - t2, 2), "detik")
    # ==== NEW : GLOBAL SCALER FIT SEKALI (ini sangat menghemat CV loop) ====
    global_scaler = StandardScaler().fit(X_feats)
    X_feats = global_scaler.transform(X_feats)

    # --- TIMER stage 3 LOOCV ---
    t4 = time.time()
    y_pred, cm, metrics = run_loocv_sgd(X_feats, y)
    t5 = time.time()
    print("Stage3 : LOOCV Total =", round(t5 - t4, 2), "detik")

    # Save outputs
    df_out = pd.DataFrame({'true': y, 'pred': y_pred})
    df_out.to_csv(OUT_PRED_CSV, index=False)
    with open(METRICS_JSON, 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Saved predictions and metrics.")

    # Confusion matrix plots
    unique_labels = np.unique(y)
    try:
        classes = [chr(ord('a') + (lab - 1)) for lab in unique_labels]
    except Exception:
        classes = [str(lab) for lab in unique_labels]
    plot_confusion_matrix(cm, classes, CM_COUNTS_PNG, normalize=False)
    plot_confusion_matrix(cm, classes, CM_NORM_PNG, normalize=True)

    print("\nClassification report:")
    print(classification_report(y, y_pred, zero_division=0))

    global_end = time.time()
    print("====================================================================================")
    print("TOTAL Runtime Script =", round(global_end - global_start, 2), "detik")
    print("====================================================================================")


if __name__ == "__main__":
    main() 

