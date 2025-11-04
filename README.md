# Healthcare Resource Allocation System

This project is a comprehensive Healthcare Resource Allocation System designed to optimize the management and distribution of resources in a healthcare facility. It leverages data-driven approaches to forecast demand, allocate resources, and improve patient care.

## Features
- **Dashboard:** Visual overview of resource status and key metrics
- **Demand Forecasting:** Predicts future resource needs using historical data
- **Priority Allocation:** Allocates resources based on urgency and priority
- **Medical Predictor:** Assists in predicting patient outcomes
- **Resource Tracker:** Monitors inventory and usage of resources
- **Alerts:** Notifies staff of critical shortages or issues
- **Procurement:** Manages resource procurement and restocking
- **Data Entry:** Interface for updating patient and resource data
- **User Authentication:** Secure login for staff

## Technologies Used
- Python (Flask)
- HTML, CSS (Bootstrap)
- Pandas, NumPy, Scikit-learn
- SQLite 

## Project Structure
```
app_enhanced.py           # Main Flask application
train_model.py            # Model training scripts
requirements.txt          # Python dependencies
database.py               # Database interactions
healthcare_dataset.csv    # Dataset

/data/                    # JSON data files
/static/                  # CSS and static assets
/templates/               # HTML templates
```

## Getting Started
1. **Clone the repository:**
   ```
   git clone https://github.com/OmkarDeshpande777/Healthcare-Resource-Allocation.git
   cd Healthcare-Resource-Allocation
   ```
2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
3. **Run the application:**
   ```
   python app_enhanced.py
   ```
4. **Access the app:**
   Open your browser and go to `http://localhost:5000`

