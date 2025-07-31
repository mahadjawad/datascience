#!/usr/bin/env python
# coding: utf-8

# # task2

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt



# In[2]:


df = pd.read_csv(r'C:\Users\mahad\Downloads\archive\Mall_Customers.csv')


# In[3]:


df.columns = [col.strip().replace('"', '') for col in df.columns]


# In[4]:


print("shape",df.shape)
print('info',df.info)


# In[5]:


print(df.head)


# In[6]:


print(df.tail)


# In[7]:


print(df)


# In[8]:


df.dropna(inplace = True)


# In[9]:


print("shape",df.shape)


# In[10]:


print(df.dtypes)


# In[11]:


df.drop_duplicates(inplace = True)


# In[12]:


print(df.shape)


# In[13]:


df.reset_index(drop=True, inplace=True)


# In[14]:


print("✅ Cleaned data shape:", df.shape)
print("\n✅ Column names:", df.columns.tolist())
print("\n✅ Missing values:\n", df.isnull().sum())
print("\n✅ First 5 rows:\n", df.head())


# In[15]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[16]:


df.columns


# In[17]:


plt.figure(figsize=(10, 6))
sns.boxplot(data = df.drop(columns = 'Gender'))
plt.title( "Boxplot of  Gender " )
plt.xticks(rotation =45)
plt.show()


# In[18]:


plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins = 20)
plt.title('Distribution of Age')
plt.xlabel("Age")
plt.ylabel('Annual Income')
plt.show


# In[25]:


# 💡 New EDA Section: Correlation Heatmap
plt.figure(figsize=(8, 6))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("🔍 Correlation Matrix")
plt.show()

# 💡 New EDA Section: Pairplot for numeric features
sns.pairplot(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']], diag_kind='kde')
plt.suptitle("📊 Pairplot of Age, Income, and Spending Score", y=1.02)
plt.show()

# 💡 New EDA Section: Spending Score vs. Age by Gender
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='Gender', y='Spending Score (1-100)', palette='pastel')
plt.title("🎯 Spending Score Distribution by Gender")
plt.show()

# 💡 New EDA Section: Spending Score vs. Age
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Age', y='Spending Score (1-100)', hue='Gender', palette='Set1')
plt.title("📈 Age vs Spending Score (colored by Gender)")
plt.show()

# 💡 New EDA Section: Income vs. Spending grouped by Gender
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Gender', style='Gender')
plt.title("💰 Income vs Spending Score by Gender")
plt.show()
plt.figure(figsize=(7, 4))
sns.countplot(data = df,x = 'Gender')
plt.title("Impact of customer gender distribution ")
plt.show()


# In[20]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data (skip if already loaded)
# df = pd.read_csv("your_file.csv")

# Drop 'CustomerID' since it's not useful for clustering
df = df.drop('CustomerID', axis=1)

# Encode 'Gender' to numbers
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])  # Male = 1, Female = 0

# Scale the features
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)

# Apply K-Means
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_df)

# Visualize clusters
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='viridis')
plt.title("Customer Segments")
plt.show()


# In[21]:


print(df.columns)


# In[22]:


# 1. Import Libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 2. Drop target column ('y') if present
if 'y' in df.columns:
    X = df.drop('y', axis=1)
else:
    X = df.copy()


# 3. Scale the features (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Find the Optimal Number of Clusters using Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# 5. Plot the Elbow Graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method to Determine Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Inertia)')
plt.grid(True)
plt.show()

# 6. Apply K-Means Clustering (Assume optimal K = 3, adjust based on elbow)
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X_scaled)



# 8. Visualize Clusters using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
plt.title('Customer Segments (K-Means Clustering)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid(True)
plt.show()

# 9. (Optional) View summary of each cluster
cluster_summary = df.groupby('Cluster').mean(numeric_only=True)
print("📊 Cluster-wise Summary:\n")
print(cluster_summary)


# In[23]:


from sklearn.decomposition import PCA

# Reduce to 2 components for better visualization
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_df)

# Run K-Means on reduced data
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(pca_data)

# Visualize
plt.figure(figsize=(10,6))
sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=df['Cluster'], palette='Set2')
plt.title("PCA-based Customer Segmentation")
plt.show()


# In[27]:


if income > 70 and score > 60:
    print("💎 **Affluent Loyalists**")
    print("- High income & high spenders.")
    print("- 🎯 Strategy: Promote luxury and exclusive products.")
    print("- 📣 Channels: Instagram ads, influencer partnerships, premium email campaigns.")
    print("- 🎁 Campaign Idea: VIP memberships, early access to sales, luxury bundles.\n")

elif income < 40 and score > 60:
    print("🛍️ **Value-Conscious Spenders**")
    print("- Low income but high spending behavior.")
    print("- 🎯 Strategy: Focus on loyalty programs and cashback offers.")
    print("- 📣 Channels: SMS, WhatsApp, in-app notifications.")
    print("- 🎁 Campaign Idea: 'Spend more, save more' tiers, reward points system.\n")

elif income > 70 and score < 40:
    print("📢 **Unengaged Affluents**")
    print("- High income but low spenders.")
    print("- 🎯 Strategy: Educate on product value and promote personalized recommendations.")
    print("- 📣 Channels: Personalized email journeys, YouTube ads, webinars.")
    print("- 🎁 Campaign Idea: 'Discover Your Style' quiz + tailored bundles or offers.\n")

elif income < 40 and score < 40:
    print("💤 **Low-Engagement Budget Buyers**")
    print("- Low income & low spenders.")
    print("- 🎯 Strategy: Minimal marketing, focus on budget items and discounts.")
    print("- 📣 Channels: In-store flyers, SMS, discount portals.")
    print("- 🎁 Campaign Idea: Clearance sales, bundled budget-friendly deals.\n")

else:
    print("🔍 **Mid-Tier Mixers**")
    print("- Average income and spending; mixed behavior.")
    print("- 🎯 Strategy: Use A/B testing to determine best conversion tactics.")
    print("- 📣 Channels: Multi-platform (email, social, app), seasonal nudges.")
    print("- 🎁 Campaign Idea: Personalized offers based on browsing/purchase history.\n")



# # (Mahad you made it you are great you did something which many people arent able to do and you completed 2 highly difficult task in 2 days)

# In[ ]:




