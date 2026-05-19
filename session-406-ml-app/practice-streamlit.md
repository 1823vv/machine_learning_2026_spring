# Streamlit


https://streamlit.io/

---

# Part 1 – Environment Setup

### Step 1: Install dependencies


```bash
pip install streamlit numpy pandas scikit-learn matplotlib
```

If you want to use the Tsinghua source:

```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple streamlit numpy pandas scikit-learn matplotlib 
```

---

### Step 2: Project structure

Create the following directory structure:

```
streamlit-ml-demo/

model/
    train_model.py
    saved_model.pkl

app/
    app.py

data/
    sample_data.csv
```

---

# Part 2 – Train a Simple Machine Learning Model

First, train a model that we will later load inside Streamlit.

Create:

```
model/train_model.py
```

Example training code:

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

np.random.seed(0)

X = np.random.randn(500,2)
y = (X[:,0] + X[:,1] > 0).astype(int)

model = LogisticRegression()
model.fit(X,y)

with open("saved_model.pkl","wb") as f:
    pickle.dump(model,f)

print("Model saved.")
```

Run:

```
python train_model.py
```

This produces:

```
saved_model.pkl
```

This file contains the trained model.

---

# Part 3 – First Streamlit Application

Now build the Streamlit interface.

Create:

```
app/app.py
```

Basic example:

```python
import streamlit as st

st.title("Machine Learning Demo")

st.write("This is a simple Streamlit application.")
```

Run the app:

```
cd app.py
streamlit run app.py
```

The application opens in the browser:

```
http://localhost:8501
```

This confirms the environment is working.

---

# Part 4 – Interactive Input Controls

Next, create interactive inputs.

Update the code:

```python
import streamlit as st

st.title("Interactive Prediction Demo")

x1 = st.slider("Feature 1", -5.0, 5.0, 0.0)
x2 = st.slider("Feature 2", -5.0, 5.0, 0.0)

st.write("Input features:", x1, x2)
```

Now the user can move sliders.

Each change **reruns the script automatically**.

This is the core design of Streamlit.

---

# Part 5 – Load the Trained Model

Now connect the ML model.

Add model loading:

```python
import pickle

with open("../model/saved_model.pkl","rb") as f:
    model = pickle.load(f)
```

Prediction code:

```python
import numpy as np

X = np.array([[x1,x2]])

prediction = model.predict(X)[0]

st.write("Prediction:", prediction)
```

Full interaction:

```
User moves sliders
        ↓
Streamlit reruns script
        ↓
Model predicts
        ↓
Result displayed
```

---

# Part 6 – Decision Boundary Visualization

Now visualize the model.

Add a plot.

```python
import numpy as np
import matplotlib.pyplot as plt

xx = np.linspace(-5,5,100)
yy = np.linspace(-5,5,100)

grid = np.array([[x,y] for x in xx for y in yy])
pred = model.predict(grid)

Z = pred.reshape(100,100)

fig, ax = plt.subplots()

ax.contourf(xx,yy,Z,alpha=0.3)

ax.scatter(x1,x2,color="red",s=100)

st.pyplot(fig)
```

Now the interface shows:

* classification regions
* the current input point
* model behavior

This makes the ML system **visually interpretable**.

---

# Part 7 – File Upload Prediction

Next, allow users to upload datasets.

Add:

```python
uploaded_file = st.file_uploader("Upload CSV file")

if uploaded_file is not None:

    import pandas as pd

    data = pd.read_csv(uploaded_file)

    st.write("Uploaded Data")
    st.dataframe(data)

    predictions = model.predict(data.values)

    st.write("Predictions")
    st.write(predictions)
```

Users can now:

1. upload CSV data
2. view the dataset
3. obtain predictions

---

# Part 8 – Sidebar Controls

Use the sidebar for configuration.

```python
st.sidebar.title("Model Settings")

threshold = st.sidebar.slider(
    "Decision Threshold",
    0.0,
    1.0,
    0.5
)
```

Sidebars help separate:

* configuration
* results

Common ML interface design pattern.

---

# Part 9 – Performance Optimization

Streamlit reruns scripts frequently.

Expensive operations should be cached.

Example:

```python
@st.cache_data
def load_model():

    import pickle

    with open("../model/saved_model.pkl","rb") as f:
        model = pickle.load(f)

    return model

model = load_model()
```

Caching prevents repeated loading.

Important when working with:

* large models
* deep learning models
* heavy preprocessing

---

# Part 10 – Final Application Workflow

Final architecture:

```
User Interaction
        ↓
Streamlit Interface
        ↓
Load Model
        ↓
Generate Prediction
        ↓
Display Visualization
```

This pipeline is typical for **machine learning demos and prototypes**.

