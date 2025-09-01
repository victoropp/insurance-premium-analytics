# Insurance Premium Analytics Platform - Production

## Videbimus AI
**Advanced Machine Learning Platform for Insurance Premium Prediction**

Developed by: Victor Collins Oppon  
Company: Videbimus AI  
Website: https://www.videbimusai.com  
Contact: consulting@videbimusai.com

---

## 📁 Production Folder Structure

```
production/
├── dashboard.py           # Main dashboard application
├── requirements.txt       # Python dependencies
├── src/                  # Source code
│   └── visualizations_updated.py  # Visualization engine
├── data/                 # Data files
│   ├── insurance_tranining_dataset.csv
│   ├── insurance_tranining_dataset_test.csv
│   ├── model_results.csv
│   ├── final_test_results.csv
│   ├── feature_importance.csv
│   └── predictions_holdout_test.csv
├── models/               # Trained ML models
│   ├── stacking_linear.pkl
│   ├── stacking_ridge.pkl
│   └── voting_ensemble.pkl
└── README.md            # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Dashboard
```bash
python dashboard.py
```

The dashboard will be available at: `http://127.0.0.1:8050`

---

## 🔧 Production Deployment

### Using Gunicorn (Recommended for Production)

```bash
gunicorn dashboard:server -b 0.0.0.0:8050 --workers 4
```

### Environment Variables (Optional)
Create a `.env` file for production settings:
```
DASH_DEBUG=False
DASH_HOST=0.0.0.0
DASH_PORT=8050
```

---

## 📊 Features

- **Executive Summary**: 6 high-level visualizations
- **Detailed Analysis**: 6 in-depth premium driver analyses  
- **Model Performance**: 6 model evaluation metrics
- **Premium Calculator**: Real-time premium predictions
- **Interactive Dashboards**: Powered by Plotly and Dash
- **Machine Learning Models**: Ensemble methods with 99.78% R² accuracy

---

## 🔒 Security Notes

- Never commit sensitive data or API keys
- Use environment variables for configuration
- Implement authentication for production deployment
- Enable HTTPS for production servers

---

## 📈 Model Performance

- **Best Model**: Stacking (Linear) Regression
- **Test R² Score**: 0.9978
- **Ensemble Models**: Stacking, Voting, Ridge

---

## 🛠️ Technical Stack

- **Frontend**: Dash, Plotly, Bootstrap
- **Backend**: Flask, Python
- **ML**: Scikit-learn, Pandas, NumPy
- **Deployment**: Gunicorn, Docker (optional)

---

## 📝 License

© 2024 Videbimus AI. All rights reserved.

---

## 🤝 Support

For technical support or consulting services:
- Email: consulting@videbimusai.com
- Website: https://www.videbimusai.com