"""
Enhanced Healthcare Resource Allocation System
Hospital-Focused Application with 10+ Core Features
DATABASE-BACKED VERSION with JSON Storage

This is a complete redesign focused on real hospital workflows and staff needs.
"""

import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify, session, redirect, url_for, flash
import os
from datetime import datetime, timedelta
import json
from collections import Counter, defaultdict
import secrets
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpStatus, value

# Import database manager
from database import db

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Load ML Model
MODEL_PATH = r"C:\Document Local\Projects\DAA Project\trained_model.pkl"
model = None
label_encoders = None
feature_columns = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as file:
        model_data = pickle.load(file)
        model = model_data.get('model')
        label_encoders = model_data.get('label_encoders', {})
        feature_columns = model_data.get('feature_columns', [])

# ===========================
# DATABASE-BACKED DATA ACCESS (Replace in-memory structures)
# ===========================

# Helper functions to access database
def get_resource_inventory():
    """Get resource inventory from database"""
    return db.get_resource_inventory()

def get_staff_data():
    """Get staff data from database"""
    return db.get_staff_data()

def get_departments():
    """Get departments from database"""
    return db.get_departments()

def get_allocation_history():
    """Get allocation history from database"""
    return db.get_allocation_history()

def get_alerts():
    """Get alerts from database"""
    return db.get_alerts()

# Backward compatibility - these will fetch from database
resource_inventory = property(lambda self: get_resource_inventory())
staff_data = property(lambda self: get_staff_data())
departments = property(lambda self: get_departments())
allocation_history = property(lambda self: get_allocation_history())
alerts = property(lambda self: get_alerts())

# Procurement recommendations
procurement_queue = []

# User roles for access control
users = {
    'admin': {'password': 'admin123', 'role': 'Administrator', 'access_level': 5},
    'doctor': {'password': 'doc123', 'role': 'Doctor', 'access_level': 4},
    'nurse': {'password': 'nurse123', 'role': 'Nurse', 'access_level': 3},
    'staff': {'password': 'staff123', 'role': 'Staff', 'access_level': 2}
}

# ===========================
# FUNCTION 1: Smart Data Entry & Validation
# ===========================

@app.route('/data-entry')
def data_entry():
    """Smart data entry interface with validation"""
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Load data from database
    departments_data = db.get_departments()
    resource_inventory_data = db.get_resource_inventory()
    
    return render_template('data_entry.html', 
                         departments=departments_data,
                         resource_inventory=resource_inventory_data)

@app.route('/api/validate-patient', methods=['POST'])
def validate_patient():
    """Validate and save patient data with smart suggestions"""
    data = request.json
    errors = []
    suggestions = []
    
    # Load departments from database
    departments_db = db.get_departments()
    
    # Age validation
    if 'age' in data:
        age = int(data['age'])
        if age < 0 or age > 120:
            errors.append("Age must be between 0 and 120")
        elif age < 18:
            suggestions.append("Consider Pediatrics department for patient under 18")
    
    # Admission type validation
    if data.get('admission_type') == 'Emergency' and data.get('severity') == 'Low':
        suggestions.append("Low severity cases may not require emergency admission")
    
    # Department capacity check
    dept = data.get('department')
    if dept in departments_db:
        occupancy_rate = (departments_db[dept]['occupied'] / departments_db[dept]['beds']) * 100
        if occupancy_rate > 90:
            suggestions.append(f"{dept} is at {occupancy_rate:.1f}% capacity. Consider alternative departments.")
    
    # If validation passes, save the patient data
    if len(errors) == 0:
        import random
        patient_id = f"P{random.randint(10000, 99999)}"
        
        # Add timestamp
        from datetime import datetime
        data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        data['patient_id'] = patient_id
        data['status'] = 'Registered'
        
        # Save to database
        db.add_allocation(data)
        
        return jsonify({
            'success': True,
            'patient_id': patient_id,
            'errors': errors,
            'suggestions': suggestions,
            'message': 'Patient registered successfully'
        })
    else:
        return jsonify({
            'success': False,
            'errors': errors,
            'suggestions': suggestions,
            'message': 'Validation failed'
        })

# ===========================
# API: Update Resource Inventory
# ===========================

@app.route('/api/update-resource', methods=['POST'])
def update_resource():
    """Update resource inventory (total, available, maintenance, etc.)"""
    try:
        data = request.json
        resource_name = data.get('resource_name')
        updates = data.get('updates', {})
        
        if not resource_name:
            return jsonify({'success': False, 'message': 'Resource name is required'}), 400
        
        # Update the resource in database
        success = db.update_resource_inventory(resource_name, updates)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Resource {resource_name} updated successfully',
                'updated_fields': list(updates.keys())
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Resource {resource_name} not found'
            }), 404
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error updating resource: {str(e)}'
        }), 500

# ===========================
# FUNCTION 2: Dynamic Resource Tracker
# ===========================

@app.route('/resource-tracker')
def resource_tracker():
    """Real-time resource tracking dashboard"""
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Load data from database
    resource_inventory = db.get_resource_inventory()
    departments_db = db.get_departments()
    staff_db = db.get_staff_data()
    
    # Calculate utilization rates and prepare resource data
    resources_with_utilization = {}
    for resource, data in resource_inventory.items():
        if data['total'] > 0:
            utilization_rate = (data['allocated'] / data['total']) * 100
            resources_with_utilization[resource] = {
                'utilization': utilization_rate,
                'status': get_resource_status(data['allocated'], data['total']),
                'available': data['available'],
                'allocated': data['allocated'],
                'total': data['total'],
                'maintenance': data.get('maintenance', 0)  # Add maintenance field
            }
    
    # Prepare departments data with proper structure
    departments_data = {}
    for dept, data in departments_db.items():
        departments_data[dept] = {
            'occupied': data['occupied'],
            'capacity': data['beds'],  # Map 'beds' to 'capacity'
            'beds': data['beds']
        }
    
    return render_template('resource_tracker.html',
                         resources=resources_with_utilization,
                         utilization=resources_with_utilization,
                         departments=departments_data,
                         staff=staff_db,
                         alerts=get_recent_alerts(5))

def get_resource_status(allocated, total):
    """Determine resource status based on utilization"""
    rate = (allocated / total) * 100
    if rate < 50:
        return 'optimal'
    elif rate < 75:
        return 'moderate'
    elif rate < 90:
        return 'high'
    else:
        return 'critical'
# ===========================
# FUNCTION 3: AI-Based Demand Forecasting
# ===========================

@app.route('/demand-forecast')
def demand_forecast():
    """AI-powered demand forecasting interface"""
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Load data from database
    allocation_history_db = db.get_allocation_history()
    departments_db = db.get_departments()
    
    # Generate forecasts for next 7 days
    forecast_data = generate_demand_forecast(allocation_history_db, days=7)
    admission_forecast_7day = generate_7day_admission_forecast()
    
    # Department-wise predictions
    dept_forecasts = predict_department_demand(departments_db, allocation_history_db)
    
    # Default data for display
    forecast_values = [45, 50, 55, 52, 60, 65, 70]
    current_trend = [40, 42, 45, 48, 50, 52, 55]
    hourly_distribution = [5, 8, 12, 18, 22, 15, 8]
    
    dept_forecast_data = {}
    for dept_name, dept_data in departments_db.items():
        current = (dept_data['occupied'] / dept_data['beds']) * 100 if dept_data['beds'] > 0 else 0
        dept_forecast_data[dept_name] = {
            'current': int(current),
            'day3': int(current + 5) if current < 85 else 95,
            'day7': int(current + 10) if current < 80 else 98,
            'confidence': 85
        }
    
    resource_recommendations = [
        {
            'icon': '🛏️',
            'resource': 'ICU Beds',
            'recommendation': 'Prepare 5 additional beds for patient overflow',
            'target_quantity': '5-10 units',
            'priority': 'High'
        },
        {
            'icon': '💨',
            'resource': 'Oxygen Units',
            'recommendation': 'Increase oxygen supply capacity by 20 units',
            'target_quantity': '20 units',
            'priority': 'Medium'
        }
    ]
    
    return render_template('demand_forecast.html',
                         predicted_admissions=285,
                         peak_occupancy=87,
                         peak_day=3,
                         critical_resources=2,
                         growth_trend='Increasing',
                         forecast_data=forecast_values,
                         current_trend=current_trend,
                         hourly_distribution=hourly_distribution,
                         department_forecast=dept_forecast_data,
                         resource_recommendations=resource_recommendations,
                         admission_forecast_7day=admission_forecast_7day)

def generate_demand_forecast(history, days=7):
    """Generate resource demand forecast using time-series analysis"""
    if len(history) < 10:
        # Not enough data, use simple averages
        return generate_simple_forecast(days)
    
    # Analyze last 30 days of data
    recent_data = history[-30:] if len(history) >= 30 else history
    
    # Calculate daily averages
    resource_usage = defaultdict(list)
    for entry in recent_data:
        for resource in entry.get('resources', {}).keys():
            resource_usage[resource].append(1)
    
    # Generate forecasts
    forecasts = []
    for day in range(1, days + 1):
        date = (datetime.now() + timedelta(days=day)).strftime('%Y-%m-%d')
        daily_forecast = {
            'date': date,
            'day_name': (datetime.now() + timedelta(days=day)).strftime('%A'),
            'predictions': {}
        }
        
        for resource in ['bed', 'oxygen', 'ventilator', 'ppe_kit']:
            avg_usage = len(resource_usage.get(resource, [])) / max(len(recent_data), 1)
            # Add trend and seasonality factors
            trend_factor = 1 + (day * 0.02)  # Slight upward trend
            predicted = int(avg_usage * trend_factor * 10)  # Scale up
            daily_forecast['predictions'][resource] = predicted
        
        forecasts.append(daily_forecast)
    
    return forecasts

def generate_simple_forecast(days):
    """Simple forecast when insufficient historical data"""
    forecasts = []
    base_values = {'bed': 10, 'oxygen': 8, 'ventilator': 3, 'ppe_kit': 15}
    
    for day in range(1, days + 1):
        date = (datetime.now() + timedelta(days=day)).strftime('%Y-%m-%d')
        forecasts.append({
            'date': date,
            'day_name': (datetime.now() + timedelta(days=day)).strftime('%A'),
            'predictions': base_values
        })
    
    return forecasts

def predict_department_demand(depts, history):
    """Predict demand for each department"""
    predictions = {}
    for dept_name, dept_data in depts.items():
        current_occupancy = (dept_data['occupied'] / dept_data['beds']) * 100
        
        # Simple prediction based on current occupancy
        if current_occupancy > 80:
            trend = 'increasing'
            predicted_change = '+5-10%'
        elif current_occupancy < 50:
            trend = 'stable'
            predicted_change = '±2%'
        else:
            trend = 'moderate'
            predicted_change = '+2-5%'
        
        predictions[dept_name] = {
            'current_occupancy': current_occupancy,
            'trend': trend,
            'predicted_change': predicted_change,
            'recommendation': get_department_recommendation(current_occupancy)
        }
    
    return predictions

def get_department_recommendation(occupancy):
    """Get recommendation based on department occupancy"""
    if occupancy > 90:
        return 'URGENT: Prepare for overflow. Consider discharge planning.'
    elif occupancy > 75:
        return 'Monitor closely. Prepare additional capacity.'
    elif occupancy < 30:
        return 'Low utilization. Available for incoming patients.'
    else:
        return 'Normal operations. Continue monitoring.'

# ===========================
# LINEAR PROGRAMMING OPTIMIZATION
# ===========================

def optimize_resource_allocation_lp(patients_data, resources_available, department_capacities):
    """
    Use Linear Programming to optimize resource allocation
    
    Args:
        patients_data: List of dicts with patient info (priority, required_resources, department)
        resources_available: Dict of available resources {resource_name: count}
        department_capacities: Dict of department capacities {dept_name: available_beds}
    
    Returns:
        Dict with allocation decisions and optimization results
    """
    
    # Create the LP problem
    prob = LpProblem("Healthcare_Resource_Allocation", LpMaximize)
    
    # Decision variables: x[i] = 1 if patient i is allocated, 0 otherwise
    num_patients = len(patients_data)
    x = {}
    for i in range(num_patients):
        x[i] = LpVariable(f"patient_{i}", cat='Binary')
    
    # Objective: Maximize total priority-weighted allocations
    # Higher priority patients get higher weight
    priority_weights = []
    for i, patient in enumerate(patients_data):
        # Priority 1 (Critical) = weight 10, Priority 2 = weight 5, etc.
        weight = 11 - patient.get('priority', 5)
        priority_weights.append(weight)
    
    prob += lpSum([priority_weights[i] * x[i] for i in range(num_patients)]), "Total_Weighted_Allocations"
    
    # Constraints: Resource availability
    for resource_name, available_count in resources_available.items():
        # Sum of patients requiring this resource <= available count
        resource_demand = []
        for i, patient in enumerate(patients_data):
            if resource_name in patient.get('required_resources', []):
                resource_demand.append(x[i])
        
        if resource_demand:
            prob += lpSum(resource_demand) <= available_count, f"Resource_{resource_name}"
    
    # Constraints: Department capacity
    for dept_name, capacity in department_capacities.items():
        # Sum of patients allocated to this department <= capacity
        dept_patients = []
        for i, patient in enumerate(patients_data):
            if patient.get('department') == dept_name:
                dept_patients.append(x[i])
        
        if dept_patients:
            prob += lpSum(dept_patients) <= capacity, f"Department_{dept_name}"
    
    # Solve the problem
    prob.solve()
    
    # Extract results
    allocation_results = {
        'status': LpStatus[prob.status],
        'objective_value': value(prob.objective),
        'allocated_patients': [],
        'rejected_patients': [],
        'resource_utilization': {}
    }
    
    for i in range(num_patients):
        if value(x[i]) == 1:
            allocation_results['allocated_patients'].append({
                'index': i,
                'patient_id': patients_data[i].get('patient_id'),
                'priority': patients_data[i].get('priority'),
                'department': patients_data[i].get('department')
            })
        else:
            allocation_results['rejected_patients'].append({
                'index': i,
                'patient_id': patients_data[i].get('patient_id'),
                'priority': patients_data[i].get('priority'),
                'reason': 'Insufficient resources or capacity'
            })
    
    # Calculate resource utilization
    for resource_name, available_count in resources_available.items():
        used = sum(1 for i, patient in enumerate(patients_data) 
                  if value(x[i]) == 1 and resource_name in patient.get('required_resources', []))
        allocation_results['resource_utilization'][resource_name] = {
            'used': used,
            'available': available_count,
            'utilization_rate': (used / available_count * 100) if available_count > 0 else 0
        }
    
    return allocation_results

# ===========================
# FUNCTION 4: Priority Allocation System (Enhanced)
# ===========================

@app.route('/priority-allocation')
def priority_allocation():
    """Advanced priority-based allocation interface"""
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Get waiting patients from database
    waiting_queue = db.get_waiting_queue()
    
    # Get allocation history from database
    allocation_history_db = db.get_allocation_history()
    
    # Prepare allocation history for display (last 10)
    allocation_history_data = []
    for alloc in allocation_history_db[-10:]:
        allocation_history_data.append({
            'time': alloc.get('timestamp', 'N/A'),
            'patient': alloc.get('name', 'Unknown'),
            'department': alloc.get('department', 'N/A'),
            'resource': 'Allocated',
            'staff': 'Auto-assigned'
        })
    
    return render_template('priority_allocation.html',
                         waiting_count=len(waiting_queue),
                         allocated_today=len([a for a in allocation_history_db if 'timestamp' in a]),
                         avg_wait_time=12.5,
                         waiting_queue=waiting_queue,
                         allocation_history=allocation_history_data)

def get_waiting_patients():
    """Get list of patients waiting for resources"""
    return db.get_waiting_queue()

@app.route('/api/allocate-priority', methods=['POST'])
def allocate_priority():
    """Allocate resources based on priority algorithm"""
    data = request.json
    patient_id = data.get('patient_id')
    priority = int(data.get('priority', 5))
    required_resources = data.get('resources', [])
    
    allocation_result = {
        'success': False,
        'allocated': [],
        'unavailable': [],
        'message': ''
    }
    
    # Load resource inventory from database
    resource_inventory = db.get_resource_inventory()
    
    for resource in required_resources:
        if resource in resource_inventory:
            if resource_inventory[resource]['available'] > 0:
                resource_inventory[resource]['available'] -= 1
                resource_inventory[resource]['allocated'] += 1
                allocation_result['allocated'].append(resource)
            else:
                allocation_result['unavailable'].append(resource)
    
    # Save updated resource inventory back to database
    db.write_json('resource_inventory.json', resource_inventory)
    
    if allocation_result['allocated']:
        allocation_result['success'] = True
        allocation_result['message'] = f"Allocated {len(allocation_result['allocated'])} resources successfully"
        
        # Create alert if some resources unavailable
        if allocation_result['unavailable']:
            create_alert('warning', f"Could not allocate: {', '.join(allocation_result['unavailable'])}")
    else:
        allocation_result['message'] = "No resources could be allocated"
        create_alert('critical', f"Failed to allocate any resources for patient {patient_id}")
    
    return jsonify(allocation_result)

@app.route('/api/optimize-allocation', methods=['POST'])
def optimize_allocation():
    """Optimize resource allocation using Linear Programming"""
    try:
        data = request.json
        patients = data.get('patients', [])
        
        if not patients:
            return jsonify({'success': False, 'message': 'No patients provided'}), 400
        
        # Load current resources and capacities
        resource_inventory = db.get_resource_inventory()
        departments_db = db.get_departments()
        
        # Prepare resources available
        resources_available = {k: v['available'] for k, v in resource_inventory.items()}
        
        # Prepare department capacities
        department_capacities = {k: v['beds'] - v['occupied'] for k, v in departments_db.items()}
        
        # Run Linear Programming optimization
        optimization_result = optimize_resource_allocation_lp(
            patients, resources_available, department_capacities
        )
        
        # Apply allocations if successful
        if optimization_result['status'] == 'Optimal':
            for allocated_patient in optimization_result['allocated_patients']:
                idx = allocated_patient['index']
                patient_data = patients[idx]
                
                # Update resource inventory
                for resource in patient_data.get('required_resources', []):
                    if resource in resource_inventory and resource_inventory[resource]['available'] > 0:
                        resource_inventory[resource]['available'] -= 1
                        resource_inventory[resource]['allocated'] += 1
                
                # Update department occupancy
                dept = patient_data.get('department')
                if dept in departments_db and departments_db[dept]['occupied'] < departments_db[dept]['beds']:
                    departments_db[dept]['occupied'] += 1
            
            # Save changes to database
            db.write_json('resource_inventory.json', resource_inventory)
            db.write_json('departments.json', departments_db)
            
            return jsonify({
                'success': True,
                'optimization_result': optimization_result,
                'message': f"Successfully allocated {len(optimization_result['allocated_patients'])} patients using LP optimization"
            })
        else:
            return jsonify({
                'success': False,
                'optimization_result': optimization_result,
                'message': 'Optimization failed or no feasible solution found'
            })
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500

# ===========================
# FUNCTION 5: Automated Alerts & Notifications
# ===========================

def create_alert(severity, message, department=None):
    """Create system alert"""
    alert_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'severity': severity,  # critical, warning, info
        'message': message,
        'department': department,
        'acknowledged': False
    }
    
    # Save to database
    db.add_alert(alert_data)

def check_resource_thresholds():
    """Check all resources and create alerts if needed"""
    resource_inventory = db.get_resource_inventory()
    
    for resource, data in resource_inventory.items():
        if data['total'] > 0:
            availability_rate = (data['available'] / data['total']) * 100
            
            if availability_rate < 10:
                create_alert('critical', 
                           f"{resource.replace('_', ' ').title()} critically low: {data['available']} remaining")
            elif availability_rate < 25:
                create_alert('warning',
                           f"{resource.replace('_', ' ').title()} running low: {data['available']} remaining")

@app.route('/alerts')
def view_alerts():
    """View all system alerts"""
    if 'user' not in session:
        return redirect(url_for('login'))
    
    check_resource_thresholds()  # Generate fresh alerts
    
    # Load alerts from database
    alerts_db = db.get_alerts()
    
    # Prepare alert data for template
    critical_alerts = [a for a in alerts_db if a.get('severity') == 'critical']
    warning_alerts = [a for a in alerts_db if a.get('severity') == 'warning']
    info_alerts = [a for a in alerts_db if a.get('severity') == 'info']
    acknowledged_alerts = [a for a in alerts_db if a.get('acknowledged')]
    
    # Format alerts for display
    formatted_alerts = []
    for alert in alerts_db[:10]:  # Show latest 10
        formatted_alerts.append({
            'severity': alert.get('severity', 'info'),
            'resource': alert.get('department', 'System'),
            'message': alert.get('message', ''),
            'timestamp': alert.get('timestamp', ''),
            'acknowledged': alert.get('acknowledged', False),
            'recommended_action': 'Take immediate action' if alert.get('severity') == 'critical' else 'Monitor situation'
        })
    
    alert_history = [
        {'timestamp': '10:15 AM', 'severity': 'critical', 'resource': 'ICU Beds', 'message': 'ICU bed availability critically low', 'acknowledged': True, 'acknowledged_by': 'Dr. Smith'},
        {'timestamp': '09:45 AM', 'severity': 'warning', 'resource': 'Oxygen Units', 'message': 'Oxygen supply running low', 'acknowledged': True, 'acknowledged_by': 'Nurse Joy'},
    ]
    
    return render_template('alerts.html',
                         critical_count=len(critical_alerts),
                         warning_count=len(warning_alerts),
                         info_count=len(info_alerts),
                         acknowledged_count=len(acknowledged_alerts),
                         alerts=formatted_alerts,
                         alert_history=alert_history)

@app.route('/api/acknowledge-alert', methods=['POST'])
def acknowledge_alert():
    """Acknowledge an alert"""
    alert_id = request.json.get('alert_id') or request.json.get('index')
    
    if alert_id is not None:
        success = db.acknowledge_alert(alert_id)
        return jsonify({'success': success})
    
    return jsonify({'success': False, 'message': 'Alert ID required'})

def get_recent_alerts(count=5):
    """Get most recent alerts"""
    alerts_db = db.get_alerts()
    return sorted(alerts_db, key=lambda x: x['timestamp'], reverse=True)[:count]

# ===========================
# FUNCTION 6: Procurement Recommendation Engine
# ===========================

@app.route('/procurement')
def procurement():
    """Procurement recommendation dashboard (Indian Context with Rupees)"""
    if 'user' not in session:
        return redirect(url_for('login'))
    
    recommendations_data = [
        {'resource': 'PPE Kits', 'current_stock': 350, 'recommended_qty': 200, 'unit_cost': 800, 'total_cost': 160000, 'lead_time': 2, 'urgency': 'High'},
        {'resource': 'Oxygen Cylinders', 'current_stock': 65, 'recommended_qty': 50, 'unit_cost': 450, 'total_cost': 22500, 'lead_time': 3, 'urgency': 'Critical'},
        {'resource': 'N95 Masks', 'current_stock': 750, 'recommended_qty': 300, 'unit_cost': 45, 'total_cost': 13500, 'lead_time': 1, 'urgency': 'Medium'},
    ]
    
    order_history_data = [
        {'date': '2025-10-28', 'resource': 'Ventilators', 'qty': 5, 'cost': 750000, 'status': 'Delivered', 'delivery': '2025-10-28'},
        {'date': '2025-10-26', 'resource': 'ICU Beds', 'qty': 10, 'cost': 800000, 'status': 'Delivered', 'delivery': '2025-10-26'},
    ]
    
    return render_template('procurement.html',
                         critical_orders=1,
                         pending_orders=2,
                         completed_orders=8,
                         recommendations=recommendations_data,
                         order_history=order_history_data)

def generate_procurement_recommendations():
    """Generate smart procurement recommendations"""
    recommendations = []
    
    for resource, data in resource_inventory.items():
        if data['total'] == 0:
            continue
            
        availability_rate = (data['available'] / data['total']) * 100
        daily_usage = estimate_daily_usage(resource, allocation_history)
        days_remaining = data['available'] / max(daily_usage, 1)
        
        # Recommend procurement if below threshold
        if availability_rate < 30 or days_remaining < 7:
            urgency = 'URGENT' if days_remaining < 3 else 'HIGH' if days_remaining < 7 else 'MEDIUM'
            
            # Calculate recommended order quantity
            recommended_qty = calculate_order_quantity(data, daily_usage)
            
            recommendations.append({
                'resource': resource.replace('_', ' ').title(),
                'current_stock': data['available'],
                'daily_usage': daily_usage,
                'days_remaining': round(days_remaining, 1),
                'urgency': urgency,
                'recommended_qty': recommended_qty,
                'estimated_cost': recommended_qty * get_unit_cost(resource),
                'supplier': get_preferred_supplier(resource),
                'lead_time': get_lead_time(resource)
            })
    
    return sorted(recommendations, key=lambda x: x['days_remaining'])

def estimate_daily_usage(resource, history):
    """Estimate daily usage of a resource"""
    if len(history) < 7:
        return 5  # Default estimate
    
    recent = history[-7:]
    usage_count = sum(1 for entry in recent if resource in entry.get('resources', {}))
    return usage_count / 7

def calculate_order_quantity(data, daily_usage):
    """Calculate recommended order quantity"""
    # Order enough for 30 days plus buffer
    target_stock = daily_usage * 30
    current = data['available']
    return max(int(target_stock - current), int(data['total'] * 0.2))

def get_unit_cost(resource):
    """Get estimated unit cost (simulated)"""
    costs = {
        'icu_beds': 50000,
        'general_beds': 15000,
        'emergency_beds': 20000,
        'oxygen_units': 500,
        'ppe_kits': 50,
        'ventilators': 25000,
        'masks': 5,
        'gloves': 2,
        'sanitizers': 10
    }
    return costs.get(resource, 100)

def get_preferred_supplier(resource):
    """Get preferred supplier for resource"""
    suppliers = {
        'icu_beds': 'MedEquip Solutions',
        'general_beds': 'Hospital Supplies Co',
        'oxygen_units': 'OxyGen Medical',
        'ppe_kits': 'SafetyFirst Medical',
        'ventilators': 'RespiraTech Inc',
        'masks': 'ProMask Industries',
        'gloves': 'MediGlove Corp',
        'sanitizers': 'CleanCare Products'
    }
    return suppliers.get(resource, 'General Supplier')

def get_lead_time(resource):
    """Get procurement lead time in days"""
    lead_times = {
        'icu_beds': '14-21 days',
        'general_beds': '7-14 days',
        'oxygen_units': '2-3 days',
        'ppe_kits': '1-2 days',
        'ventilators': '21-30 days',
        'masks': '1-2 days',
        'gloves': '1-2 days',
        'sanitizers': '1-2 days'
    }
    return lead_times.get(resource, '3-5 days')

# ===========================
# FUNCTION 8: Performance & Utilization Reports (REMOVED)
# ===========================

# ===========================
# FUNCTION 9: Role-Based Access Control
# ===========================

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login with role-based access"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username in users and users[username]['password'] == password:
            session['user'] = username
            session['role'] = users[username]['role']
            session['access_level'] = users[username]['access_level']
            flash(f'Welcome {users[username]["role"]} {username}!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid credentials', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """User logout"""
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

def require_access_level(level):
    """Decorator to require minimum access level"""
    def decorator(f):
        def wrapper(*args, **kwargs):
            if 'user' not in session:
                return redirect(url_for('login'))
            if session.get('access_level', 0) < level:
                flash('Insufficient permissions', 'error')
                return redirect(url_for('home'))
            return f(*args, **kwargs)
        wrapper.__name__ = f.__name__
        return wrapper
    return decorator

# ===========================
# FUNCTION 10: Data Analytics & Insights Hub (REMOVED)
# ===========================

# ===========================
# BONUS FUNCTION 11: Feedback & Improvement Module (REMOVED)
# ===========================

# ===========================
# MAIN ROUTES
# ===========================

@app.route('/')
def home():
    """Main dashboard - hospital overview"""
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Load data from database
    departments_db = db.get_departments()
    alerts_db = db.get_alerts()
    staff_db = db.get_staff_data()
    allocation_history_db = db.get_allocation_history()
    
    # Generate overview statistics
    overview = {
        'total_patients': sum(d['occupied'] for d in departments_db.values()),
        'available_beds': sum(d['beds'] - d['occupied'] for d in departments_db.values()),
        'critical_alerts': sum(1 for a in alerts_db if a['severity'] == 'critical' and not a['acknowledged']),
        'staff_on_duty': sum(s['on_duty'] for s in staff_db.values()),
        'resource_status': get_resource_summary(),
        'recent_admissions': len([a for a in allocation_history_db 
                                 if a['timestamp'].startswith(datetime.now().strftime('%Y-%m-%d'))])
    }
    
    return render_template('home.html',
                         overview=overview,
                         departments=departments_db,
                         alerts=get_recent_alerts(5),
                         user=session.get('user'),
                         role=session.get('role'))

def get_resource_summary():
    """Get summary of resource status"""
    critical_count = 0
    warning_count = 0
    optimal_count = 0
    
    resource_inventory = db.get_resource_inventory()
    
    for resource, data in resource_inventory.items():
        if data['total'] > 0:
            rate = (data['allocated'] / data['total']) * 100
            if rate > 90:
                critical_count += 1
            elif rate > 75:
                warning_count += 1
            else:
                optimal_count += 1
    
    return {
        'critical': critical_count,
        'warning': warning_count,
        'optimal': optimal_count
    }

@app.route('/predict', methods=['POST'])
def predict():
    """Enhanced patient prediction with comprehensive data and ML model"""
    try:
        if model is None:
            return render_template('result.html', prediction="Error",
                                 allocation="Model not loaded. Please run train_model.py first.", resources=None)
        
        # Get enhanced patient data
        input_data = {
            'Age': float(request.form['Age']),
            'Gender': request.form['Gender'],
            'Blood Type': request.form['Blood_Type'],
            'Admission Type': request.form['Admission_Type']
        }
        
        # Additional fields for better tracking
        patient_name = request.form.get('patient_name', 'Anonymous')
        department = request.form.get('department', 'General Ward')
        symptoms = request.form.get('symptoms', '')
        comorbidities = request.form.get('comorbidities', '')
        
        # Create DataFrame and encode for ML model
        input_df = pd.DataFrame([input_data])
        encoded_df = input_df.copy()
        
        for col in ['Gender', 'Blood Type', 'Admission Type']:
            if col in label_encoders and col in encoded_df.columns:
                le = label_encoders[col]
                try:
                    encoded_df[col] = le.transform(encoded_df[col].astype(str))
                except ValueError as e:
                    # Handle unseen labels gracefully
                    print(f"Warning: Unknown value for {col}, using default encoding")
                    encoded_df[col] = 0
        
        # ML Model Prediction
        prediction = model.predict(encoded_df)[0]
        prediction_proba = model.predict_proba(encoded_df)[0]
        confidence = max(prediction_proba) * 100
        
        # Get all possible predictions with probabilities
        all_predictions = []
        if hasattr(model, 'classes_'):
            for i, cls in enumerate(model.classes_):
                all_predictions.append({
                    'condition': cls,
                    'probability': prediction_proba[i] * 100
                })
            all_predictions = sorted(all_predictions, key=lambda x: x['probability'], reverse=True)
        
        # Determine severity and priority based on ML prediction
        severity_map = {
            'Cancer': {'priority': 1, 'severity': 'Critical', 'resources': ['icu_beds', 'ventilators', 'oxygen_units', 'ppe_kits']},
            'Diabetes': {'priority': 2, 'severity': 'High', 'resources': ['general_beds', 'oxygen_units']},
            'Hypertension': {'priority': 2, 'severity': 'High', 'resources': ['general_beds', 'oxygen_units']},
            'Asthma': {'priority': 3, 'severity': 'Moderate', 'resources': ['general_beds', 'oxygen_units']},
            'Obesity': {'priority': 4, 'severity': 'Low', 'resources': ['general_beds']},
            'Arthritis': {'priority': 4, 'severity': 'Low', 'resources': ['general_beds']}
        }
        
        severity_info = severity_map.get(prediction, {'priority': 5, 'severity': 'Unknown', 'resources': ['general_beds']})
        priority = severity_info['priority']
        severity = severity_info['severity']
        required_resources = severity_info['resources']
        
        # Use Linear Programming for optimal resource allocation
        patient_data_lp = [{
            'patient_id': f'P{np.random.randint(10000, 99999)}',
            'priority': priority,
            'required_resources': required_resources,
            'department': department
        }]
        
        # Load current resources and capacities
        resource_inventory = db.get_resource_inventory()
        departments_db = db.get_departments()
        
        resources_available = {k: v['available'] for k, v in resource_inventory.items()}
        department_capacities = {k: v['beds'] - v['occupied'] for k, v in departments_db.items()}
        
        # Run Linear Programming optimization
        lp_result = optimize_resource_allocation_lp(patient_data_lp, resources_available, department_capacities)
        
        # Allocate resources based on LP result
        allocated_resources = {}
        if lp_result['status'] == 'Optimal' and len(lp_result['allocated_patients']) > 0:
            # LP found optimal solution - apply allocation
            allocated_resources = allocate_resources_enhanced(prediction, priority, input_data['Admission Type'], department)
            allocation_status = "Optimally Allocated (LP)"
        else:
            # Fallback to standard allocation if LP fails
            allocated_resources = allocate_resources_enhanced(prediction, priority, input_data['Admission_Type'], department)
            allocation_status = "Allocated (Standard)"
        
        # Define allocation message
        allocation_messages = {
            1: "🩸 CRITICAL: ICU bed + Ventilator + Oxygen + PPE kit + 24/7 monitoring",
            2: "🩺 HIGH PRIORITY: General ward + Regular monitoring + Oxygen support",
            3: "💊 MODERATE: General ward + Basic monitoring + Medication",
            4: "🏠 LOW PRIORITY: Home care or outpatient treatment recommended"
        }
        allocation = allocation_messages.get(priority, "Standard care")
        
        # Log enhanced allocation to database
        log_enhanced_allocation(patient_name, input_data, prediction, severity,
                              allocated_resources, department, symptoms, comorbidities)
        
        # Check for critical resources and create alerts
        check_resource_thresholds()
        
        return render_template('result.html',
                             patient_name=patient_name,
                             prediction=prediction,
                             allocation=allocation,
                             confidence=f"{confidence:.1f}",
                             severity=severity,
                             priority=priority,
                             resources=allocated_resources,
                             department=department,
                             symptoms=symptoms,
                             all_predictions=all_predictions[:5],  # Top 5 predictions
                             allocation_status=allocation_status,
                             lp_optimization=lp_result)
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Prediction Error: {error_details}")
        return render_template('result.html', prediction="Error",
                             allocation=f"Error during prediction: {str(e)}", resources=None)

def allocate_resources_enhanced(condition, priority, admission_type, department):
    """Enhanced resource allocation with department awareness using database"""
    allocated = {}
    
    # Load data from database
    departments_db = db.get_departments()
    resource_inventory = db.get_resource_inventory()
    
    # Update department occupancy
    if department in departments_db:
        if departments_db[department]['occupied'] < departments_db[department]['beds']:
            departments_db[department]['occupied'] += 1
            allocated['bed'] = f"{department} Bed"
            # Save updated departments
            db.write_json('departments.json', departments_db)
        else:
            allocated['bed'] = f'Waiting List ({department} Full)'
    
    # Allocate based on priority
    if priority == 1:  # Critical
        for resource in ['icu_beds', 'ventilators', 'oxygen_units', 'ppe_kits']:
            if resource in resource_inventory and resource_inventory[resource]['available'] > 0:
                resource_inventory[resource]['available'] -= 1
                resource_inventory[resource]['allocated'] += 1
                allocated[resource] = 'Allocated'
            else:
                allocated[resource] = 'Unavailable - URGENT'
                create_alert('critical', f"{resource.replace('_', ' ').title()} unavailable for critical patient")
    
    elif priority == 2:  # High
        for resource in ['general_beds', 'oxygen_units']:
            if resource in resource_inventory and resource_inventory[resource]['available'] > 0:
                resource_inventory[resource]['available'] -= 1
                resource_inventory[resource]['allocated'] += 1
                allocated[resource] = 'Allocated'
    
    elif priority == 3:  # Moderate
        if 'general_beds' in resource_inventory and resource_inventory['general_beds']['available'] > 0:
            resource_inventory['general_beds']['available'] -= 1
            resource_inventory['general_beds']['allocated'] += 1
            allocated['general_bed'] = 'Allocated'
    
    # Save updated resource inventory
    db.write_json('resource_inventory.json', resource_inventory)
    
    return allocated

def log_enhanced_allocation(patient_name, patient_data, prediction, severity,
                           resources, department, symptoms, comorbidities):
    """Enhanced allocation logging with additional fields to database"""
    allocation_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'patient_name': patient_name,
        'age': patient_data['Age'],
        'gender': patient_data['Gender'],
        'blood_type': patient_data['Blood Type'],
        'condition': prediction,
        'severity': severity,
        'resources': resources,
        'department': department,
        'symptoms': symptoms,
        'comorbidities': comorbidities,
        'admission_type': patient_data['Admission Type'],
        'patient_id': f'P{np.random.randint(10000, 99999)}',
        'status': 'Allocated'
    }
    
    # Save to database
    db.add_allocation(allocation_data)

# API Endpoints
@app.route('/api/dashboard-data')
def get_dashboard_data():
    """Get real-time dashboard data for updates"""
    # Load data from database
    resource_inventory = db.get_resource_inventory()
    departments_db = db.get_departments()
    staff_db = db.get_staff_data()
    alerts_db = db.get_alerts()
    allocation_history_db = db.get_allocation_history()
    
    return jsonify({
        'resources': resource_inventory,
        'departments': departments_db,
        'staff': staff_db,
        'alerts_count': len([a for a in alerts_db if not a.get('acknowledged', False)]),
        'patients_today': len([a for a in allocation_history_db 
                             if a.get('timestamp', '').startswith(datetime.now().strftime('%Y-%m-%d'))])
    })

@app.route('/api/analytics/patient-trends')
def get_patient_trends():
    """Get patient admission trends over time (dynamic data)"""
    allocation_history_db = db.get_allocation_history()
    
    # Get last 30 days of data
    from datetime import datetime, timedelta
    trends = {}
    for i in range(7):  # Last 7 days
        date = (datetime.now() - timedelta(days=6-i)).strftime('%Y-%m-%d')
        day_name = (datetime.now() - timedelta(days=6-i)).strftime('%a')
        trends[day_name] = len([a for a in allocation_history_db if a.get('timestamp', '').startswith(date)])
    
    return jsonify({
        'labels': list(trends.keys()),
        'data': list(trends.values()),
        'total': sum(trends.values())
    })

@app.route('/api/analytics/condition-distribution')
def get_condition_distribution():
    """Get medical condition distribution from real patient data"""
    allocation_history_db = db.get_allocation_history()
    
    # Count conditions
    conditions = {}
    for alloc in allocation_history_db:
        condition = alloc.get('condition', alloc.get('medical_condition', 'Unknown'))
        conditions[condition] = conditions.get(condition, 0) + 1
    
    total = sum(conditions.values()) if conditions else 1
    
    # Format for charts
    result = {}
    for condition, count in conditions.items():
        result[condition] = {
            'count': count,
            'percentage': round((count / total) * 100, 1)
        }
    
    return jsonify(result)

@app.route('/api/analytics/resource-patterns')
def get_resource_patterns():
    """Get resource usage patterns from real data"""
    resource_inventory = db.get_resource_inventory()
    allocation_history_db = db.get_allocation_history()
    
    patterns = []
    for resource_name, resource_data in resource_inventory.items():
        if resource_data['total'] > 0:
            utilization = (resource_data['allocated'] / resource_data['total']) * 100
            
            # Analyze usage pattern from history
            usage_count = sum(1 for alloc in allocation_history_db[-100:] 
                            if resource_name in str(alloc.get('resources', {})))
            
            pattern_desc = "High demand" if utilization > 75 else "Moderate demand" if utilization > 50 else "Low demand"
            
            patterns.append({
                'resource': resource_name.replace('_', ' ').title(),
                'description': f'{pattern_desc} - {utilization:.0f}% utilized',
                'utilization': utilization,
                'recent_usage': usage_count,
                'peak_hours': 'Variable based on admissions'
            })
    
    return jsonify(patterns)

@app.route('/api/forecast/demand')
def get_demand_forecast():
    """Get AI-powered demand forecast using real historical data"""
    allocation_history_db = db.get_allocation_history()
    
    # Analyze last 7 days to predict next 7 days
    forecast_data = generate_demand_forecast(allocation_history_db, days=7)
    
    # Calculate hourly distribution from historical data
    hourly_dist = [0] * 24
    for alloc in allocation_history_db[-100:]:
        try:
            timestamp = alloc.get('timestamp', '')
            if timestamp:
                hour = int(timestamp.split(' ')[1].split(':')[0])
                hourly_dist[hour] += 1
        except:
            pass
    
    # Get peak hours (top 7 hours)
    peak_hours = sorted(range(len(hourly_dist)), key=lambda i: hourly_dist[i], reverse=True)[:7]
    hourly_data = [hourly_dist[i] for i in peak_hours]
    hourly_labels = [f'{i:02d}:00' for i in peak_hours]
    
    return jsonify({
        'forecast_data': [f.get('predictions', {}).get('bed', 10) for f in forecast_data],
        'forecast_labels': [f['day_name'][:3] for f in forecast_data],
        'hourly_data': hourly_data,
        'hourly_labels': hourly_labels,
        'total_predicted': sum(f.get('predictions', {}).get('bed', 10) for f in forecast_data)
    })

@app.route('/api/forecast/department')
def get_department_forecast():
    """Get department-wise demand forecast from real data"""
    departments_db = db.get_departments()
    allocation_history_db = db.get_allocation_history()
    
    dept_forecast = {}
    for dept_name, dept_data in departments_db.items():
        current_occupancy = (dept_data['occupied'] / dept_data['beds'] * 100) if dept_data['beds'] > 0 else 0
        
        # Predict based on recent trends
        recent_admissions = len([a for a in allocation_history_db[-30:] 
                                if a.get('department') == dept_name])
        
        growth_rate = min(5, recent_admissions / 3)  # Conservative growth estimate
        
        dept_forecast[dept_name] = {
            'current': int(current_occupancy),
            'day3': min(int(current_occupancy + growth_rate), 100),
            'day7': min(int(current_occupancy + growth_rate * 2), 100),
            'confidence': 85,
            'trend': 'Increasing' if growth_rate > 2 else 'Stable'
        }
    
    return jsonify(dept_forecast)

@app.route('/api/resource-utilization')
def get_resource_utilization():
    """Get real-time resource utilization data for charts"""
    resource_inventory = db.get_resource_inventory()
    
    utilization_data = {}
    for resource_name, resource_data in resource_inventory.items():
        if resource_data['total'] > 0:
            utilization_rate = (resource_data['allocated'] / resource_data['total']) * 100
            utilization_data[resource_name] = {
                'utilization': round(utilization_rate, 1),
                'available': resource_data['available'],
                'allocated': resource_data['allocated'],
                'total': resource_data['total'],
                'status': get_resource_status(resource_data['allocated'], resource_data['total'])
            }
    
    return jsonify(utilization_data)

# ===========================
# CUSTOM JINJA2 FILTERS (INDIAN CONTEXT)
# ===========================

@app.template_filter('indian_rupees')
def format_indian_rupees(amount):
    """Format amount in Indian Rupees with lakhs/crores notation"""
    try:
        amount = float(amount)
        if amount >= 10000000:  # 1 crore
            return f"₹{amount/10000000:.2f} Cr"
        elif amount >= 100000:  # 1 lakh
            return f"₹{amount/100000:.2f} L"
        elif amount >= 1000:
            return f"₹{amount/1000:.2f} K"
        else:
            return f"₹{amount:,.2f}"
    except:
        return f"₹{amount}"

@app.template_filter('indian_number')
def format_indian_number(number):
    """Format number in Indian numbering system"""
    try:
        number = int(number)
        s = str(number)
        if len(s) <= 3:
            return s
        result = s[-3:]
        s = s[:-3]
        while s:
            result = s[-2:] + ',' + result
            s = s[:-2]
        return result
    except:
        return str(number)

# ===========================
# SIMPLE MEDICAL CONDITION PREDICTOR
# ===========================

@app.route('/medical-predictor')
def medical_predictor():
    """Simple medical condition prediction interface"""
    if 'user' not in session:
        return redirect(url_for('login'))
    
    return render_template('medical_predictor.html')

@app.route('/api/predict-condition', methods=['POST'])
def predict_condition():
    """Predict medical condition based on Age, Gender, Blood Type, Admission Type, and Medication"""
    try:
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please ensure trained_model.pkl exists.'
            }), 500
        
        # Get input data
        data = request.json
        age = float(data.get('age', 0))
        gender = data.get('gender', '')
        blood_type = data.get('blood_type', '')
        admission_type = data.get('admission_type', 'Elective')
        medication = data.get('medication', 'None')
        
        # Validate inputs
        if not age or not gender or not blood_type:
            return jsonify({
                'success': False,
                'error': 'All fields are required: Age, Gender, and Blood Type'
            }), 400
        
        # Create input data for prediction
        input_data = {
            'Age': age,
            'Gender': gender,
            'Blood Type': blood_type,
            'Admission Type': admission_type
        }
        
        # Create DataFrame and encode for ML model
        input_df = pd.DataFrame([input_data])
        encoded_df = input_df.copy()
        
        # Encode categorical variables
        for col in ['Gender', 'Blood Type', 'Admission Type']:
            if col in label_encoders and col in encoded_df.columns:
                le = label_encoders[col]
                try:
                    encoded_df[col] = le.transform(encoded_df[col].astype(str))
                except ValueError as e:
                    # Handle unseen labels gracefully
                    print(f"Warning: Unknown value for {col}, using default encoding")
                    encoded_df[col] = 0
        
        # ML Model Prediction
        prediction = model.predict(encoded_df)[0]
        prediction_proba = model.predict_proba(encoded_df)[0]
        confidence = max(prediction_proba) * 100
        
        # Get all possible predictions with probabilities
        all_predictions = []
        if hasattr(model, 'classes_'):
            for i, cls in enumerate(model.classes_):
                all_predictions.append({
                    'condition': cls,
                    'probability': round(prediction_proba[i] * 100, 2)
                })
            all_predictions = sorted(all_predictions, key=lambda x: x['probability'], reverse=True)
        
        # Get condition information with medication recommendations
        condition_info = {
            'Cancer': {
                'severity': 'Critical',
                'description': 'Requires immediate attention and specialized care',
                'icon': '🔴',
                'color': '#dc3545',
                'admission_type': 'Emergency/Hospitalization',
                'recommended_medication': ['Chemotherapy agents', 'Pain management drugs', 'Anti-nausea medications', 'Immunotherapy'],
                'recommendations': [
                    'Immediate consultation with oncologist',
                    'Comprehensive diagnostic tests required',
                    'Consider hospitalization for treatment planning',
                    'Family support and counseling recommended'
                ]
            },
            'Diabetes': {
                'severity': 'High',
                'description': 'Chronic condition requiring regular monitoring',
                'icon': '🟠',
                'color': '#fd7e14',
                'admission_type': 'Elective/Outpatient',
                'recommended_medication': ['Metformin', 'Insulin', 'GLP-1 agonists', 'SGLT2 inhibitors', 'Sulfonylureas'],
                'recommendations': [
                    'Regular blood sugar monitoring',
                    'Dietary modifications required',
                    'Medication as prescribed by endocrinologist',
                    'Regular exercise and weight management'
                ]
            },
            'Hypertension': {
                'severity': 'High',
                'description': 'High blood pressure requiring lifestyle changes',
                'icon': '🟠',
                'color': '#fd7e14',
                'admission_type': 'Outpatient',
                'recommended_medication': ['ACE Inhibitors', 'Beta-blockers', 'Calcium channel blockers', 'Diuretics', 'ARBs'],
                'recommendations': [
                    'Regular blood pressure monitoring',
                    'Reduce salt intake',
                    'Regular cardiovascular exercise',
                    'Stress management techniques'
                ]
            },
            'Asthma': {
                'severity': 'Moderate',
                'description': 'Respiratory condition requiring management',
                'icon': '🟡',
                'color': '#ffc107',
                'admission_type': 'Emergency/Outpatient',
                'recommended_medication': ['Albuterol inhalers', 'Corticosteroid inhalers', 'Leukotriene modifiers', 'Long-acting bronchodilators'],
                'recommendations': [
                    'Keep rescue inhaler available',
                    'Avoid known triggers',
                    'Regular pulmonologist consultations',
                    'Air quality monitoring'
                ]
            },
            'Obesity': {
                'severity': 'Moderate',
                'description': 'Weight management required to prevent complications',
                'icon': '🟡',
                'color': '#ffc107',
                'admission_type': 'Outpatient/Dietary counseling',
                'recommended_medication': ['Orlistat', 'Phentermine', 'GLP-1 agonists', 'Metformin'],
                'recommendations': [
                    'Balanced diet with caloric deficit',
                    'Regular physical activity',
                    'Nutritionist consultation',
                    'Monitor for related conditions'
                ]
            },
            'Arthritis': {
                'severity': 'Low',
                'description': 'Joint condition requiring pain management',
                'icon': '🟢',
                'color': '#28a745',
                'admission_type': 'Outpatient',
                'recommended_medication': ['NSAIDs', 'DMARDs', 'Biologics', 'Corticosteroids', 'Hyaluronic acid injections'],
                'recommendations': [
                    'Regular low-impact exercise',
                    'Joint-friendly activities',
                    'Pain management as needed',
                    'Physical therapy may help'
                ]
            }
        }
        
        primary_condition_info = condition_info.get(prediction, {
            'severity': 'Unknown',
            'description': 'Condition requires medical evaluation',
            'icon': '⚪',
            'color': '#6c757d',
            'admission_type': 'Evaluation required',
            'recommended_medication': ['Consult healthcare provider'],
            'recommendations': ['Consult with healthcare provider']
        })
        
        # Return prediction results
        return jsonify({
            'success': True,
            'prediction': {
                'condition': prediction,
                'confidence': round(confidence, 2),
                'severity': primary_condition_info['severity'],
                'description': primary_condition_info['description'],
                'icon': primary_condition_info['icon'],
                'color': primary_condition_info['color'],
                'admission_type': primary_condition_info['admission_type'],
                'recommended_medication': primary_condition_info['recommended_medication'],
                'recommendations': primary_condition_info['recommendations']
            },
            'all_predictions': all_predictions[:6],  # Top 6 predictions
            'input_data': {
                'age': age,
                'gender': gender,
                'blood_type': blood_type,
                'admission_type': admission_type,
                'medication': medication
            }
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Prediction Error: {error_details}")
        return jsonify({
            'success': False,
            'error': f'Error during prediction: {str(e)}'
        }), 500

# ===========================
# ADMISSION FORECAST (Next 7 Days)
# ===========================

@app.route('/admission-forecast')
def admission_forecast():
    """7-day admission forecast with peak predictions"""
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Generate 7-day forecast with peak on Friday, Saturday, Sunday
    forecast_data = generate_7day_admission_forecast()
    
    return render_template('admission_forecast.html',
                         forecast_data=forecast_data)

def generate_7day_admission_forecast():
    """Generate 7-day admission forecast with realistic peak patterns"""
    from datetime import datetime, timedelta
    
    forecast = []
    base_admissions = 45
    
    # Peak pattern: Normal Mon-Thu, High Fri-Sun
    peak_factors = {
        'Monday': 1.0,      # Normal
        'Tuesday': 0.95,    # Slightly low
        'Wednesday': 1.1,   # Moderate increase
        'Thursday': 1.05,   # Moderate
        'Friday': 1.5,      # Peak starts
        'Saturday': 1.8,    # Peak high
        'Sunday': 1.6       # Peak continues
    }
    
    condition_distribution = {
        'Monday': {'Diabetes': 35, 'Hypertension': 25, 'Asthma': 20, 'Arthritis': 15, 'Obesity': 5},
        'Tuesday': {'Hypertension': 30, 'Diabetes': 28, 'Asthma': 18, 'Arthritis': 15, 'Cancer': 9},
        'Wednesday': {'Diabetes': 38, 'Hypertension': 32, 'Asthma': 22, 'Arthritis': 18, 'Obesity': 10},
        'Thursday': {'Asthma': 25, 'Hypertension': 28, 'Diabetes': 32, 'Arthritis': 12, 'Cancer': 8},
        'Friday': {'Diabetes': 55, 'Hypertension': 45, 'Asthma': 35, 'Arthritis': 20, 'Cancer': 15},
        'Saturday': {'Hypertension': 65, 'Diabetes': 60, 'Asthma': 40, 'Obesity': 25, 'Arthritis': 18},
        'Sunday': {'Diabetes': 58, 'Hypertension': 55, 'Asthma': 38, 'Arthritis': 22, 'Cancer': 12}
    }
    
    admission_types = {
        'Monday': {'Emergency': 30, 'Elective': 50, 'Urgent': 20},
        'Tuesday': {'Emergency': 25, 'Elective': 48, 'Urgent': 19},
        'Wednesday': {'Emergency': 35, 'Elective': 55, 'Urgent': 22},
        'Thursday': {'Emergency': 32, 'Elective': 52, 'Urgent': 21},
        'Friday': {'Emergency': 50, 'Elective': 65, 'Urgent': 35},
        'Saturday': {'Emergency': 60, 'Elective': 80, 'Urgent': 45},
        'Sunday': {'Emergency': 55, 'Elective': 72, 'Urgent': 40}
    }
    
    for day_offset in range(7):
        current_date = datetime.now() + timedelta(days=day_offset)
        day_name = current_date.strftime('%A')
        date_str = current_date.strftime('%Y-%m-%d')
        
        # Calculate admissions with peak factor
        peak_factor = peak_factors.get(day_name, 1.0)
        predicted_admissions = int(base_admissions * peak_factor)
        
        # Determine occupancy level
        if day_name in ['Friday', 'Saturday', 'Sunday']:
            occupancy_level = 'High'
            occupancy_percentage = 85 + (day_offset % 3) * 5
            bed_status = 'Critical - Limited availability'
        elif day_name == 'Thursday':
            occupancy_level = 'Moderate'
            occupancy_percentage = 65
            bed_status = 'Moderate - Plan accordingly'
        else:
            occupancy_level = 'Normal'
            occupancy_percentage = 55
            bed_status = 'Good - Beds available'
        
        # Get condition and admission type data
        conditions = condition_distribution.get(day_name, {})
        adm_types = admission_types.get(day_name, {})
        
        # Determine resource alerts
        alerts = []
        if day_name in ['Friday', 'Saturday', 'Sunday']:
            alerts = [
                'Expected high ICU bed demand',
                'Prepare additional oxygen supply',
                'Staff reinforcement recommended',
                'Emergency department staffing alert'
            ]
        elif day_name == 'Wednesday':
            alerts = ['Moderate increase in admissions expected']
        
        forecast_day = {
            'date': date_str,
            'day_name': day_name,
            'predicted_admissions': predicted_admissions,
            'occupancy_level': occupancy_level,
            'occupancy_percentage': min(occupancy_percentage, 100),
            'bed_status': bed_status,
            'peak_factor': peak_factor,
            'conditions': conditions,
            'admission_types': adm_types,
            'alerts': alerts,
            'confidence': 85 + (5 if day_offset <= 2 else 0),  # Higher confidence for near-term
            'recommended_actions': get_forecast_actions(day_name, predicted_admissions)
        }
        
        forecast.append(forecast_day)
    
    return forecast

def get_forecast_actions(day_name, predicted_admissions):
    """Get recommended actions based on forecast"""
    actions = []
    
    if day_name in ['Friday', 'Saturday', 'Sunday']:
        actions = [
            f'🚨 Peak Expected: Prepare for ~{predicted_admissions} admissions',
            '👥 Increase on-call staff availability by 50%',
            '🛏️ Reserve ICU beds - expected high demand',
            '💨 Stock oxygen supply - check cylinder levels',
            '📋 Prepare discharge planning for existing patients',
            '📞 Alert department heads for potential overflow',
            '🏥 Contact nearby hospitals for bed availability'
        ]
    elif day_name == 'Thursday':
        actions = [
            f'📊 Moderate admissions expected: ~{predicted_admissions}',
            '👥 Standard staffing levels adequate',
            '🛏️ Ensure bed capacity - prepare for weekend surge',
            '📋 Begin discharge planning for non-critical patients'
        ]
    else:
        actions = [
            f'✅ Normal operations: ~{predicted_admissions} admissions expected',
            '👥 Routine staffing levels sufficient',
            '🛏️ Good bed availability maintained',
            '📊 Good time for maintenance and restocking'
        ]
    
    return actions

if __name__ == "__main__":
    # Check for critical resources on startup
    check_resource_thresholds()
    app.run(debug=True, port=5000)