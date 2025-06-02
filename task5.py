import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('C:/Users/shivpoojan mishra/Downloads/titanic.csv')
df.head()
df.info()
df.describe()
df.isnull().sum()
df['Survived'].value_counts()
df['Age'].hist(bins=30)
plt.title('Age Distribution')

df['Pclass'].value_counts().plot(kind='bar')
plt.title('Passenger Class Count')
sns.countplot(x='Sex', hue='Survived', data=df)
sns.countplot(x='Pclass', hue='Survived', data=df)
sns.pairplot(df[['Survived', 'Age', 'Fare', 'Pclass']])
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')






