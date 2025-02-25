# Image-Classification-Using-KNN
An image classification model that distinguishes between the images of a cat and dog using K-Nearest Neighbors (KNN). 

---

# **Image Classification Using k-NN**  

This repository contains an **image classification project using k-Nearest Neighbors (k-NN)**. The dataset consists of **25,000 images** (50% cats, 50% dogs), and the best-performing **k-value was found to be k = 5** based on the confusion matrix.  

## **Project Overview**  
- **Dataset**: (https://www.kaggle.com/datasets/ashfakyeafi/cat-dog-images-for-classification)
- **Algorithm**: k-Nearest Neighbors (k-NN)  
- **Best k-value**: 5  
- **Training Duration**: 0:01:44 

---

## **Contents**  
- **Import Libraries**: Load necessary Python libraries.  
- **Load Images in Python**: Read and process images.  
- **Plotting an Image**: Visualize sample images.  
- **Gray Scale Images**: Convert images to grayscale.  
- **k-NN for Image Classification**: Train and evaluate the k-NN model.  

---

## **Implementation**  
### **1. Data Preprocessing**
- Images are loaded and converted to grayscale.  
- They are resized to a uniform dimension (32x32 pixels).  
- Flattened into vectors for k-NN classification.  

### **2. k-NN Training & Classification**
- OpenCV’s k-NN implementation (`cv2.ml.KNearest_create()`) is used.  
- The model is trained using `train_samples` and `train_labels`.  
- The classifier is tested on `test_samples` with different k-values.  

```python
knn = cv2.ml.KNearest_create()
knn.train(train_samples, cv2.ml.ROW_SAMPLE, train_labels)

k_values = [1, 2, 3, 4, 5]
k_result = []
for k in k_values:
    ret, result, neighbours, dist = knn.findNearest(test_samples, k=k)
    k_result.append(result)
```

### **3. Model Performance**
- The best `k` value was determined based on performance metrics.  
- **Confusion Matrix for k = 5**:  
  ![Confusion Matrix](image.png)  
  - **True Positives (Cats):** 1892  
  - **False Positives (Dogs classified as Cats):** 662  
  - **False Negatives (Cats classified as Dogs):** 1446  
  - **True Negatives (Dogs):** 1000  

---

## **Results & Observations**
- The model performed **best at k = 5**, balancing accuracy and generalization.  
- Accuracy = **(TP + TN) / (Total samples)**  
- Training time: **_[Insert time]_**  
