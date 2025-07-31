Advanced Task 2: Customer Segmentation Using Unsupervised Learning
Objective:
Cluster customers based on spending habits and propose marketing strategies tailored to each segment.

Dataset:
Name: Mall Customers Dataset
Source: Available on Kaggle (https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial)
File Used: Mall_Customers.csv
Key Features:
Gender
Age
Annual Income (k$)
Spending Score (1-100)
Workflow Summary:
Exploratory Data Analysis (EDA):
Explored distributions of age, income, and spending score
Visualized feature relationships using:
Scatter plots
Pair plots
Box plots
Count plots
Clustering (K-Means):
Applied K-Means Clustering after selecting features
Used the Elbow Method to determine optimal number of clusters (k = 5)
Assigned each customer to a segment
Dimensionality Reduction:
Applied Principal Component Analysis (PCA) to reduce data to 2D for visualization
Visualized customer clusters in PCA-reduced space
Alternative t-SNE technique was considered for non-linear projection
Cluster Insights & Marketing Strategy:
Cluster	Characteristics	Strategy
0	High spenders, high income	Target premium products
1	Low spenders, low income	Budget offerings or improve engagement
2	Average income, moderate spend	Upsell or loyalty programs
3	Younger high spenders	Promote trending or luxury brands
4	Older low spenders	Personalized discounts or retention strategies
Skills Gained:
Unsupervised learning using K-Means
Dimensionality reduction using PCA
EDA and cluster visualization
Business-oriented thinking in customer segmentation
Tools & Libraries:
Python
pandas, numpy
seaborn, matplotlib
scikit-learn (KMeans, PCA)
