import pandas as pd 
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
clients = pd.DataFrame({
    'client_id': range(1, 101),
    'age': np.random.randint(18, 75, 100),
    'risk_score': np.random.uniform(0, 1, 100) # 0 to 1 scale
})


staff = pd.DataFrame({
    'staff_id': range(1, 11),
    'years_exp': [2, 5, 10, 1, 8, 3, 12, 4, 6, 7],
    'workload': np.random.randint(5, 20, 10)
})


cases = pd.DataFrame({
    'case_id': range(1, 501),
    'client_id': np.random.randint(1, 101, 500),
    'staff_id': np.random.randint(1, 11, 500),
    'priority': np.random.choice([1, 2, 3], 500),
    # Target: 1 for 'Resolved', 0 for 'Not Resolved' (Pending/Escalated)
    'is_resolved': np.random.choice([0, 1], 500) 
})
clients.head(30)
staff.head(30)
cases.head(20)