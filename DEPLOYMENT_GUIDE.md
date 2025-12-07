# Streamlit Cloud Deployment Guide

## Quick Deployment Steps

### Option 1: Deploy via GitHub (Recommended)

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit - Car Damage Detection"
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to https://share.streamlit.io/
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path: `app.py`
   - Click "Deploy"

### Option 2: Direct Upload

1. **Go to Streamlit Cloud:**
   - Visit https://share.streamlit.io/
   - Sign in

2. **Create New App:**
   - Click "New app"
   - Choose "Upload files"

3. **Upload These Files:**
   - `app.py`
   - `requirements.txt`
   - All model files (*.h5, *.pkl)
   - `README.md` (optional)

4. **Deploy:**
   - Click "Deploy"
   - Wait 2-3 minutes for build

## File Structure
```
streamlit_deployment/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── best_model_info.pkl         # Model metadata
├── custom_cnn_best.h5         # Deep Learning models
├── vgg16_best.h5
├── resnet50_best.h5
├── inceptionv3_best.h5
├── mobilenetv2_best.h5
├── efficientnetb0_best.h5
├── logistic_regression.pkl    # Machine Learning models
├── knn.pkl
├── svm.pkl
├── naive_bayes.pkl
├── decision_tree.pkl
├── random_forest.pkl
├── gradient_boosting.pkl
└── xgboost.pkl
```

## Requirements
- Python 3.10+
- Total size: ~737 MB
- Streamlit Cloud free tier: 1GB storage ✅

## Troubleshooting

### Build Fails
- Check requirements.txt has all dependencies
- Verify all model files are uploaded
- Check logs for specific errors

### Models Not Loading
- Ensure model files are in same directory as app.py
- Check file names match exactly
- Verify file integrity (re-upload if needed)

### Slow Performance
- First run takes 30-60 seconds (loading models)
- Subsequent runs are faster (cached)
- Consider reducing number of models if needed

## Benefits of Streamlit vs Gradio
✅ No dependency conflicts
✅ Faster deployment
✅ Better caching mechanisms
✅ More stable builds
✅ Cleaner UI customization

## Support
For issues, check Streamlit docs: https://docs.streamlit.io/
