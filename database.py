"""
Database Manager for Hospital Resource Management System
Handles all JSON file operations with thread-safe read/write
"""
import json
import os
from threading import Lock
from datetime import datetime

class DatabaseManager:
    """Manages JSON-based database operations"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.locks = {}
        
        # Ensure data directory exists
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # Initialize locks for each file
        self.files = [
            'resource_inventory.json',
            'staff_data.json',
            'departments.json',
            'allocation_history.json',
            'alerts.json',
            'patients.json',
            'waiting_queue.json'
        ]
        
        for file in self.files:
            self.locks[file] = Lock()
    
    def _get_file_path(self, filename):
        """Get full file path"""
        return os.path.join(self.data_dir, filename)
    
    def read_json(self, filename):
        """Thread-safe JSON file read"""
        file_path = self._get_file_path(filename)
        
        with self.locks[filename]:
            try:
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                else:
                    # Return empty structure based on file type
                    if filename.endswith('_history.json') or filename.endswith('_queue.json') or filename == 'alerts.json' or filename == 'patients.json':
                        return []
                    else:
                        return {}
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON in {filename}, returning empty structure")
                return [] if filename.endswith('_history.json') or filename.endswith('_queue.json') or filename == 'alerts.json' or filename == 'patients.json' else {}
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                return [] if filename.endswith('_history.json') or filename.endswith('_queue.json') or filename == 'alerts.json' or filename == 'patients.json' else {}
    
    def write_json(self, filename, data):
        """Thread-safe JSON file write"""
        file_path = self._get_file_path(filename)
        
        with self.locks[filename]:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                return True
            except Exception as e:
                print(f"Error writing {filename}: {e}")
                return False
    
    # Resource Inventory Operations
    def get_resource_inventory(self):
        """Get all resources"""
        return self.read_json('resource_inventory.json')
    
    def update_resource_inventory(self, resource_name, updates):
        """Update specific resource"""
        inventory = self.get_resource_inventory()
        if resource_name in inventory:
            inventory[resource_name].update(updates)
            return self.write_json('resource_inventory.json', inventory)
        return False
    
    def update_full_resource_inventory(self, inventory):
        """Update entire inventory"""
        return self.write_json('resource_inventory.json', inventory)
    
    # Staff Data Operations
    def get_staff_data(self):
        """Get all staff data"""
        return self.read_json('staff_data.json')
    
    def update_staff_data(self, staff_data):
        """Update staff data"""
        return self.write_json('staff_data.json', staff_data)
    
    # Department Operations
    def get_departments(self):
        """Get all departments"""
        return self.read_json('departments.json')
    
    def update_departments(self, departments):
        """Update departments"""
        return self.write_json('departments.json', departments)
    
    def update_department(self, dept_name, updates):
        """Update specific department"""
        departments = self.get_departments()
        if dept_name in departments:
            departments[dept_name].update(updates)
            return self.write_json('departments.json', departments)
        return False
    
    # Patient Operations
    def get_patients(self):
        """Get all patients"""
        return self.read_json('patients.json')
    
    def add_patient(self, patient_data):
        """Add new patient"""
        patients = self.get_patients()
        patients.append(patient_data)
        return self.write_json('patients.json', patients)
    
    def get_patient_by_id(self, patient_id):
        """Get patient by ID"""
        patients = self.get_patients()
        for patient in patients:
            if patient.get('patient_id') == patient_id:
                return patient
        return None
    
    # Allocation History Operations
    def get_allocation_history(self):
        """Get allocation history"""
        return self.read_json('allocation_history.json')
    
    def add_allocation(self, allocation_data):
        """Add allocation record"""
        history = self.get_allocation_history()
        history.append(allocation_data)
        return self.write_json('allocation_history.json', history)
    
    # Waiting Queue Operations
    def get_waiting_queue(self):
        """Get waiting queue"""
        return self.read_json('waiting_queue.json')
    
    def add_to_queue(self, patient_data):
        """Add patient to waiting queue"""
        queue = self.get_waiting_queue()
        queue.append(patient_data)
        return self.write_json('waiting_queue.json', queue)
    
    def remove_from_queue(self, patient_id):
        """Remove patient from waiting queue"""
        queue = self.get_waiting_queue()
        queue = [p for p in queue if p.get('patient_id') != patient_id]
        return self.write_json('waiting_queue.json', queue)
    
    def update_waiting_queue(self, queue):
        """Update entire waiting queue"""
        return self.write_json('waiting_queue.json', queue)
    
    # Alert Operations
    def get_alerts(self):
        """Get all alerts"""
        return self.read_json('alerts.json')
    
    def add_alert(self, alert_data):
        """Add new alert"""
        alerts = self.get_alerts()
        alerts.append(alert_data)
        return self.write_json('alerts.json', alerts)
    
    def update_alerts(self, alerts):
        """Update all alerts"""
        return self.write_json('alerts.json', alerts)
    
    def acknowledge_alert(self, alert_id):
        """Acknowledge specific alert"""
        alerts = self.get_alerts()
        for alert in alerts:
            if alert.get('id') == alert_id:
                alert['acknowledged'] = True
                alert['acknowledged_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                break
        return self.write_json('alerts.json', alerts)

# Global database instance
db = DatabaseManager()
