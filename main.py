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
df = cases.merge(clients,on="client_id").merge(staff,on="staff_id")
X= df[['age', 'risk_score', 'years_exp', 'workload', 'priority']]
y= df[["is_resolved"]]

df.head()
# 20% of the data to "test" the model's accuracy
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2, random_state=42)
# train the model 
tree_model = DecisionTreeClassifier(max_depth =3,random_state=42 )
tree_model.fit(X_train,Y_train)
y_pred = tree_model.predict(X_test)
print("Model Performance ")
print(classification_report(Y_test, y_pred))
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# 1. Defined the input shape (5 features)
initial_type = [('float_input', FloatTensorType([None, 5]))]

# 2. Disable 'zipmap'. This forces probabilities to be a standard 
# Tensor (array) instead of a complex Map object.
options = {type(tree_model): {'zipmap': False}}

# 3. Convert with the new options
onx = convert_sklearn(
    tree_model, 
    initial_types=initial_type, 
    options=options
)

# 4. Saved the new file 
with open("case_predictor.onnx", "wb") as f:
    f.write(onx.SerializeToString())

print("✅ Model re-exported with ZipMap disabled!")