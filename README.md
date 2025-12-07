# Car Damage Detection - Streamlit Deployment

## Multi-Model Car Damage Detection System

This application uses **14 different machine learning and deep learning models** to detect car damage from images.

### Models Included:
- **Deep Learning Models (6):** Custom CNN, VGG16, ResNet50, InceptionV3, MobileNetV2, EfficientNetB0
- **Machine Learning Models (8):** Logistic Regression, KNN, SVM, Naive Bayes, Decision Tree, Random Forest, Gradient Boosting, XGBoost

### Features:
- Upload car images for instant damage detection
- Get predictions from all 14 models simultaneously
- View consensus prediction with confidence scores
- Compare performance across different model types
- See detailed probability distributions

### Best Model Performance:
- **SVM:** 91.74% accuracy, 91.74% F1-score

### How to Use:
1. Upload a car image (JPG, JPEG, or PNG)
2. Click "Analyze with ALL Models"
3. View consensus prediction and detailed results from all models

### Technical Details:
- Image preprocessing: 224x224 RGB
- Feature extraction: VGG16 (for ML models)
- Binary classification: Damage vs All Fine
- Ensemble approach with majority voting

---

🎓 **ML Course Project** | Demonstrating practical application of multiple ML/DL algorithms
