
import pandas as pd

diabetes_overfit = pd.read_csv("diabetes.csv")

diabetes_overfit.head()

diabetes_overfit.isnull().sum()

diabetes_overfit['smoking_history'].unique()

diabetes_overfit.drop('smoking_history', axis=1, inplace=True)

diabetes_overfit.head()

"""Encode The Gender"""

diabetes_overfit['gender'].unique()

diabetes_overfit.groupby('gender').size()

diabetes_overfit = diabetes_overfit[diabetes_overfit['gender'] != 'Other']

diabetes_overfit.groupby('gender').size()

diabetes_overfit.groupby('diabetes').size()

class_1 = diabetes_overfit[diabetes_overfit['diabetes'] == 1]
class_0 = diabetes_overfit[diabetes_overfit['diabetes'] == 0].sample(n=8500, random_state=42)
diabetes = pd.concat([class_0, class_1])
diabetes.groupby('diabetes').size()

# diabetes=diabetes_overfit

# diabetes['gender'].replace(['Female','Male'],[0,1],inplace = True)

from sklearn.preprocessing import LabelEncoder
lable_encoder = LabelEncoder()
diabetes['gender'] = lable_encoder.fit_transform(diabetes['gender'])
diabetes['gender'].unique()

diabetes.head()

import seaborn as sns

sns.heatmap(diabetes[['diabetes','age','bmi','HbA1c_level','blood_glucose_level','gender','heart_disease']].corr(),annot = True)

# sns.boxplot(diabetes)

import numpy as np

# Assuming 'bmi' is the name of the column containing the BMI values in your DataFrame
bmi_values = diabetes['bmi']

# Calculate the first quartile (Q1) and third quartile (Q3)
Q1 = np.percentile(bmi_values, 25)
Q3 = np.percentile(bmi_values, 75)

# Calculate the interquartile range (IQR)
IQR = Q3 - Q1

# Define the outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Count the number of outliers
outliers_count = sum((bmi_values < lower_bound) | (bmi_values > upper_bound))

print("Number of outliers in BMI feature:", outliers_count)

diabetes.info()

"""Imputing Outliers"""

import numpy as np

# Calculate the first quartile (Q1) and third quartile (Q3)
Q1 = np.percentile(diabetes['bmi'], 25)
Q3 = np.percentile(diabetes['bmi'], 75)

# Calculate the interquartile range (IQR)
IQR = Q3 - Q1

# Define the outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter the DataFrame to remove outliers
diabetes_cleaned = diabetes[(diabetes['bmi'] >= lower_bound) & (diabetes['bmi'] <= upper_bound)]
load_diabetes=diabetes_cleaned.copy()
# Print the shape of the cleaned DataFrame to see how many outliers were removed
print("Shape of original DataFrame:", diabetes.shape)
print("Shape of cleaned DataFrame:", diabetes_cleaned.shape)

sns.boxplot(diabetes_cleaned)

X = diabetes_cleaned.drop(['diabetes'], axis = 1)
y = diabetes_cleaned['diabetes']

X.head()

y.head()

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size = 0.8,random_state=42)

from sklearn import preprocessing
stand = preprocessing.StandardScaler()
X_train = stand.fit_transform(X_train)
X_test = stand.transform(X_test)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
deepneuralnetworks_model = tf.keras.models.Sequential()
deepneuralnetworks_model.add(tf.keras.layers.Dense(units = 7, activation = 'relu'))
deepneuralnetworks_model.add(tf.keras.layers.Dense(units = 7, activation = 'relu'))
deepneuralnetworks_model.add(tf.keras.layers.Dense(units = 7, activation = 'relu'))
deepneuralnetworks_model.add(tf.keras.layers.Dense(units = 7, activation = 'relu'))
deepneuralnetworks_model.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))
deepneuralnetworks_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
deep_history = deepneuralnetworks_model.fit(X_train, y_train, epochs=100,
                              validation_data = (X_test, y_test),
                              callbacks=[early_stop])

predict = deepneuralnetworks_model.predict(X_test)

import pandas as pd

# فرض کنید مدل و مقیاس‌کننده (scaler) قبلا تعریف و بارگذاری شده‌اند
# مقادیر ورودی مثال؛ جایگزین با مقادیر واقعی خود کنید
gender_convert = 0
age = 20.0
hypertension_convert = 0
heart_disease_convert = 0
bmi = 35.0
HbA1c_level = 7.5
blood_glucose_level = 160.0

# آماده‌سازی ورودی برای پیش‌بینی
input_data = pd.DataFrame([[gender_convert, age, hypertension_convert, heart_disease_convert, bmi, HbA1c_level, blood_glucose_level]],
                          columns=['gender', 'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level'])

# استانداردسازی داده‌های ورودی
standardized_input = stand.transform(input_data)

# پیش‌بینی
prediction = deepneuralnetworks_model.predict(standardized_input)

# بررسی احتمال پیش‌بینی
probability = prediction[0].item()
if probability > 0.5:
    risk_level = "High Risk"
else:
    risk_level = "Low Risk"

# چاپ نتایج
print(f"Risk Level: {risk_level}")
print(f"The probability of having diabetes is: {probability * 100:.2f} Percent")
print(f"The probability of not having diabetes is: {(1 - probability) * 100:.2f} Percent")

print(predict)

import pickle

#saving the model and encoder

data = {"deepneuralnetworks_model": deepneuralnetworks_model,"lable_encoder": lable_encoder }
with open('linear_regression_model3.pkl','wb') as file:
    pickle.dump(data,file)

import joblib
joblib.dump(deepneuralnetworks_model,'deepneuralnetworks_model.pkl')
joblib.dump(stand,'standard_scaler.pkl')
joblib.dump(load_diabetes,'load_diabetes.pkl')

import tensorflow as tf
path = './model.h5'
deepneuralnetworks_model.save(path )
loaded_model= tf.keras.models.load_model(path )