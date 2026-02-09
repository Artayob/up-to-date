import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

class DataLoader():
    def __init__(self, filepath):
        self.filepath = filepath
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File was not found at: {self.filepath}")
        
    def load_content(self):
        try:
            df = pd.read_csv(self.filepath, delimiter =',')
            return df
        except Exception as e:
            raise FileNotFoundError(f"Error Reading file: {e}")

class DataCleaner():
    def __init__(self, data):
        self.data = data
    
    def Missing_values(self):
        missing_values =  self.data.isnull().sum()
        if missing_values.any():
            print("missing Values found at this columns:")
        else:
            print("No Missing values found anywhere")
        return missing_values 
    
    def Duplicate_values(self):
        duplicates = self.data.duplicated().sum()
        if duplicates >0:
            print(f"found {duplicates} duplicated rows")
        else:
            print("No duplicates found")
        return duplicates 
    

loader = DataLoader("C:/Users/sakit/Desktop/up to date/deaths_and_causes_synthetic.csv")
content = loader.load_content()

counts = content["Cause_of_Death"].value_counts()

plt.figure(figsize=(8,8))
plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
plt.title(f"Pie Chart of {"Cause_of_Death"}")
plt.show()


year_group = content.groupby("Year")["Number_of_Deaths"].sum()
print(year_group)

year_group2 = content.groupby(["Year", "Cause_of_Death"]).size()
print(year_group2)

year_group2.plot(kind="area", stacked=True, figsize=(12,6), alpha=0.7)
plt.title("Deaths per Year by Cause of Death (Area Chart)")
plt.xlabel("Year")
plt.ylabel("Number of Deaths")
plt.legend(title="Cause of Death", bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()


content = content.sort_values(by="Number_of_Deaths", ascending= False)
print(content.head(15)) 
cleaner = DataCleaner(content)
cleaner.Missing_values()
cleaner.Duplicate_values()

content2 = content.sort_values(by="Country")
print(content2.head(5))

df = pd.read_csv("deaths_and_causes_synthetic.csv")

cause_dummies = pd.get_dummies(df["Cause_of_Death"])
X = pd.concat([df["Year"], cause_dummies], axis=1)
y = df["Number_of_Deaths"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Predictions", predictions[:10])
print("Actual", y_test.values[:10])

df["Predicted_Deaths"] = model.predict(X)
print(df[["Year", "Cause_of_Death", "Number_of_Deaths", "Predicted_Deaths"]].to_string(index=False))