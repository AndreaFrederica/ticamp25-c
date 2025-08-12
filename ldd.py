import cv2, glob, os
import numpy as np

X_data = []
y_labels = []
for filepath in glob.glob("dig/*.png"):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    X_data.append(img.flatten())
    file_name = os.path.basename(filepath)
    label = int(file_name.split("_")[0])
    y_labels.append(label)

X_data = np.array(X_data, dtype=np.float32)
y_labels = np.array(y_labels, dtype=np.int32)
print("Loaded", X_data.shape[0], "images with labels", set(y_labels))

# Normalize the data
X_data = X_data / 255.0

# Try the method that works with your OpenCV version

svm = cv2.ml.SVM()  # OpenCV 2.x

svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_RBF)
svm.setC(1.0)
svm.setGamma(0.001)

# Add termination criteria to prevent infinite training
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-6)
svm.setTermCriteria(criteria)

try:
    svm.train(X_data, cv2.ml.ROW_SAMPLE, y_labels)
    svm.save("svm.dat")
    print("Training completed successfully")
except Exception as e:
    print(f"Training failed: {e}")