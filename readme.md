# Healthcare: Stroke Prediction

&nbsp;

<u>**Context**</u>

I've been hired (hypothetically) by the Johns Hopkins Hospital to create a machine learning model to predict whether or not a patient is likely to suffer a stroke. Being able to predict this will allow doctors to advise patients and their families on how to reduce cet risk, but also on how to act in the case of an emergency.

The project is based upon the [Stroke Prediction Dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset) on Kaggle that's been uploaded by the dataset grandmaster [Fedesoriano](https://www.kaggle.com/fedesoriano).

&nbsp;

<u>**Project Structure**</u>

```
.
└── Project home ->                                     # Main directory
    ├── models ->                                       # Saved models directory
    │   ├── deployment                                  # Deployment model directory  
    ├── src ->                                          # Source file directory
    │   ├── data                                        # Data directory
    │   │   └── healthcare-dataset-stroke-data.csv      # Dataset in csv format
    │   └── lib                                         # Code directory
    │       └── helper_functions.py                     # File with functions used in the notebook
    ├── 325.ipynb                                       # Assignment
    ├── my.log                                          # Log file for all logged outputs
    ├── project.ipynb                                   # Notebook containing the project
    ├── readme.md                                       # This readme file
    └── requirements.txt                                # Requirements file for package installation
```

&nbsp;

<u>**Usage**</u>

Open the `project.ipynb` here on GitHub, or open it in your preferred editor. Installation of the necessary modules is done in the notebook itself so no need to do anything else.

&nbsp;

<u>**Table of Contents**</u>    

- Healthcare: Stroke Prediction    
  - Setup
    - Installation and Imports    
    - Initial Setup
  - Data Loading and Exploration    
    - Data Loading    
    - First Exploration    
    - Data Cleaning    
  - EDA: Exploratory Data Analysis    
    - Univariate Analysis    
      - Gender    
      - Age    
      - Hypertension    
      - Heart Disease    
      - Ever Married    
      - Work Type    
      - Residence Type    
      - Average Glucose Level    
      - BMI : Body Mass Index    
      - Smoking Status    
      - Stroke    
    - Multivariate Exploration    
      - Gender    
      - Age    
      - Hypertension    
      - Heart Disease    
      - Ever Married    
      - Work Type    
      - Residence Type    
      - Average Glucose Level    
      - BMI: Body Mass Index    
      - Smoking    
    - Correlations    
  - Statistical Analysis    
    - Hypothesis 1    
  - Machine Learning    
    - Data Loading    
    - Data Preparation    
      - Train - test split    
      - Data Preprocessing    
    - Model Training and Evaluation    
      - Logistic Regression    
        - Hyperparameter Optimization    
        - Model Evaluation    
      - Random Forest    
      - Support Vector Machine    
      - K-Nearest Neighbors    
      - Model Ensembling    
      - XGBoost    
    - Optimization    
    - Model Deployment
    
&nbsp;

<u>**Attributes in the dataset**</u>

If you'd like, you can already read the attributes that are in the dataset below. These are of course also covered in the project itself.

1) **id** : unique identifier

2) **gender** : "Male", "Female" or "Other"

3) **age** : age of the patient

4) **hypertension** : 0 if the patient doesn't have hypertension, 1 if the patient has hypertension

5) **heart_disease** : 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease

6) **ever_married** : "No" or "Yes"

7) **work_type** : "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"

8) **Residence_type** : "Rural" or "Urban"

9) **avg_glucose_level** : average glucose level in blood

10) **bmi** : body mass index

11) **smoking_status** : "formerly smoked", "never smoked", "smokes" or "Unknown"*

12) **stroke** : 1 if the patient had a stroke or 0 if not

*Note: "Unknown" in smoking_status means that the information is unavailable for this patient

&nbsp;


    
