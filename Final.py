# Import functions
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Import the CSV file into a dataframe and stores it as pd
df = pd.read_csv(r'C:\Users\Zach\OneDrive\Desktop\mental-heath-in-tech-2016_20161114.csv')


#
# --- The following contains the code that is required to clean the data. This includes removing columns. ---
# --- Removing these columns helps significantly with empty entries. ---
#

# Remove all rows that have self-employed people. And then remove the column.
df = df[df['Are you self-employed?'] != 1]
df = df.drop(['Are you self-employed?'], axis=1)

# Remove 'Is your employer primarily a tech company/organization?' and 'Is your primary role within your company related to tech/IT?'
# This is because all current data entries are for tech oriented employees. There is no need for these columns.
df = df.drop(['Is your employer primarily a tech company/organization?','Is your primary role within your company related to tech/IT?'], axis=1)

#We then remove 8 columns that do not contain any data since we removed the 'Are you self-employed columns.
df = df.drop(['Do you have medical coverage (private insurance or state-provided) which includes treatment of \xa0mental health issues?',
              'Do you know local or online resources to seek help for a mental health disorder?',
              'If you have been diagnosed or treated for a mental health disorder, do you ever reveal this to clients or business contacts?',
              'If you have revealed a mental health issue to a client or business contact, do you believe this has impacted you negatively?',
              'If you have been diagnosed or treated for a mental health disorder, do you ever reveal this to coworkers or employees?',
              'If you have revealed a mental health issue to a coworker or employee, do you believe this has impacted you negatively?',
              'Do you believe your productivity is ever affected by a mental health issue?',
              'If yes, what percentage of your work time (time performing primary or secondary job functions) is affected by a mental health issue?'
              ], axis=1)

#Using the 'print(df.isnull().sum()*100/len(df))' function we see that are columns with missing or NA values.
#Below, a list of columns is stored in col_to_impute. They have different reasons for imputing.

cols_to_impute = [
#There are 11 columns with 131 null values. This is 11.4% of the remaining (1146) rows after the above cleaning.
#The percentage is still low enough to impute these values instead of dropping the columns.
            'Have your previous employers provided mental health benefits?',
            'Were you aware of the options for mental health care provided by your previous employers?',
            'Did your previous employers ever formally discuss mental health (as part of a wellness campaign or other official communication)?',
            'Did your previous employers provide resources to learn more about mental health issues and how to seek help?',
            'Was your anonymity protected if you chose to take advantage of mental health or substance abuse treatment resources with previous employers?',
            'Do you think that discussing a mental health disorder with previous employers would have negative consequences?',
            'Do you think that discussing a physical health issue with previous employers would have negative consequences?',
            'Would you have been willing to discuss a mental health issue with your previous co-workers?',
            'Would you have been willing to discuss a mental health issue with your direct supervisor(s)?',
            'Did you feel that your previous employers took mental health as seriously as physical health?',
            'Did you hear of or observe negative consequences for co-workers with mental health issues in your previous workplaces?',
# 11.6% of the rows in the column below contain NA values. For this reason we impute the mode value to remove the NA values.
            'Do you know the options for mental health care available under your employer-provided coverage?',
# 4.5% of the rows in the following column contain NA values. For this reason we impute the mode value to remove the NA values.
            'Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your current or previous workplace?',

                 ]

for col in cols_to_impute:
    mode_value = df[col].mode()[0]
    df[col] = df[col].fillna(mode_value)

#Below, a list of columns is to be dropped. They have different reasons for being dropped.
df = df.drop([
# The column below has 55.4% of its rows with missing values. For this reason we will drop the column.
        'Have your observations of how another individual who discussed a mental health disorder made you less likely to reveal a mental health issue yourself in your current workplace?',
# We will drop the following 6 columns as they are free text columns. These add complexity and will be a challenge to convert to numerical data for the ML algorithms.
        'If yes, what condition(s) have you been diagnosed with?',
        'If maybe, what condition(s) do you believe you have?',
        'If so, what condition(s) were you diagnosed with?',
        'Why or why not?',
        'Why or why not?.1',
        'Which of the following best describes your work position?',
#The following two columns will be dropped as they apply specifically to US. We will keep the country information, and remove the state information.
        'What US state or territory do you live in?',
        'What US state or territory do you work in?',
#
        'What country do you live in?',
        'What country do you work in?',
#The following column is dropped as it has been make obsolete when the columns after this one were imputed.
        'Do you have previous employers?'
            ], axis = 1)

#Formatting the data in 'What is your gender? to contain either Male, Female or Other. And then impute the missing values.
def standardize_gender(gender):
    gender = str(gender).strip().lower()
    # Normalize female
    if gender in ['female', 'f']:
        return 'Female'
    # Normalize male
    elif gender in ['male', 'm']:
        return 'Male'
    # Everything else
    else:
        return 'Other'
df['What is your gender?'] = df['What is your gender?'].apply(standardize_gender)



#
# --- The following code is used to encode the data set. We do this because we have a number of columns that are categorical in nature. ---
# --- We need to encode them to obtain numerical results to be able to apply machine learning algorithms. ---
# --- Once encoded the encoded columns replace the original column. ---
#

yes_no_map = {
    'Yes': 1,
    'No': 0
}
df['Have you been diagnosed with a mental health condition by a medical professional?'] =df['Have you been diagnosed with a mental health condition by a medical professional?'].map(yes_no_map)

#The following encodes using OrdinalEncoder. This is because there is an order to the categorical data than needs to be preserved.
ordinal_col_1 = 'How many employees does your company or organization have?'
# Define ordered categories
size_order = [['1-5','6-25', '26-100', '100-500', '500-1000', 'More than 1000']]
# Fit ordinal encoder and Replace the original column with the encoded version
ordinal_encoder = OrdinalEncoder(categories=size_order)
df[ordinal_col_1] = ordinal_encoder.fit_transform(df[[ordinal_col_1]])

ordinal_col_2 = 'If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:'
size_order = [[ 'Very difficult', 'Somewhat difficult',  "I don't know", 'Neither easy nor difficult','Somewhat easy','Very easy']]
ordinal_encoder = OrdinalEncoder(categories=size_order)
df[ordinal_col_2] = ordinal_encoder.fit_transform(df[[ordinal_col_2]])

ordinal_col_3 ='Did your previous employers provide resources to learn more about mental health issues and how to seek help?'
size_order = [['None did', 'Some did', 'Yes, they all did']]
ordinal_encoder = OrdinalEncoder(categories=size_order)
df[ordinal_col_3] = ordinal_encoder.fit_transform(df[[ordinal_col_3]])

ordinal_col_4 = 'Do you think that discussing a physical health issue with previous employers would have negative consequences?'
size_order = [['None of them', 'Some of them', 'Yes, all of them']]
ordinal_encoder = OrdinalEncoder(categories=size_order)
df[ordinal_col_4] = ordinal_encoder.fit_transform(df[[ordinal_col_4]])

ordinal_col_5 = 'Would you have been willing to discuss a mental health issue with your previous co-workers?'
size_order = [['No, at none of my previous employers', 'Some of my previous employers','Yes, at all of my previous employers']]
ordinal_encoder = OrdinalEncoder(categories=size_order)
df[ordinal_col_5] = ordinal_encoder.fit_transform(df[[ordinal_col_5]])

ordinal_col_6 = 'Did you hear of or observe negative consequences for co-workers with mental health issues in your previous workplaces?'
size_order = [['None of them', 'Some of them','Yes, all of them']]
ordinal_encoder = OrdinalEncoder(categories=size_order)
df[ordinal_col_6] = ordinal_encoder.fit_transform(df[[ordinal_col_6]])

ordinal_col_7 = 'Would you be willing to bring up a physical health issue with a potential employer in an interview?'
size_order = [['No', 'Maybe','Yes']]
ordinal_encoder = OrdinalEncoder(categories=size_order)
df[ordinal_col_7] = ordinal_encoder.fit_transform(df[[ordinal_col_7]])

ordinal_col_8 = 'Would you bring up a mental health issue with a potential employer in an interview?'
size_order = [['No', 'Maybe','Yes']]
ordinal_encoder = OrdinalEncoder(categories=size_order)
df[ordinal_col_8] = ordinal_encoder.fit_transform(df[[ordinal_col_8]])

ordinal_col_9 = 'Do you feel that being identified as a person with a mental health issue would hurt your career?'
size_order = [['No, it has not', "No, I don't think it would", 'Maybe','Yes, I think it would', 'Yes, it has']]
ordinal_encoder = OrdinalEncoder(categories=size_order)
df[ordinal_col_9] = ordinal_encoder.fit_transform(df[[ordinal_col_9]])

ordinal_col_10 = 'Do you think that team members/co-workers would view you more negatively if they knew you suffered from a mental health issue?'
size_order = [['No, they do not', "No, I don't think they would", 'Maybe','Yes, I think they would','Yes, they do']]
ordinal_encoder = OrdinalEncoder(categories=size_order)
df[ordinal_col_10] = ordinal_encoder.fit_transform(df[[ordinal_col_10]])

ordinal_col_11 = 'How willing would you be to share with friends and family that you have a mental illness?'
size_order = [['Not applicable to me (I do not have a mental illness)', 'Not open at all','Somewhat not open','Neutral','Somewhat open','Very open']]
ordinal_encoder = OrdinalEncoder(categories=size_order)
df[ordinal_col_11] = ordinal_encoder.fit_transform(df[[ordinal_col_11]])

ordinal_col_12 = 'Have you had a mental health disorder in the past?'
size_order = [['No', 'Maybe','Yes']]
ordinal_encoder = OrdinalEncoder(categories=size_order)
df[ordinal_col_12] = ordinal_encoder.fit_transform(df[[ordinal_col_12]])

ordinal_col_13 = 'Do you currently have a mental health disorder?'
size_order = [['No', 'Maybe','Yes']]
ordinal_encoder = OrdinalEncoder(categories=size_order)
df[ordinal_col_13] = ordinal_encoder.fit_transform(df[[ordinal_col_13]])

ordinal_col_14 = 'If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?'
size_order = [['Not applicable to me', 'Never','Rarely','Sometimes','Often']]
ordinal_encoder = OrdinalEncoder(categories=size_order)
df[ordinal_col_14] = ordinal_encoder.fit_transform(df[[ordinal_col_14]])

ordinal_col_15 = 'If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?'
size_order = [['Not applicable to me', 'Never','Rarely','Sometimes','Often']]
ordinal_encoder = OrdinalEncoder(categories=size_order)
df[ordinal_col_15] = ordinal_encoder.fit_transform(df[[ordinal_col_15]])

ordinal_col_16 = 'Do you work remotely?'
size_order = [['Never', 'Sometimes','Always']]
ordinal_encoder = OrdinalEncoder(categories=size_order)
df[ordinal_col_16] = ordinal_encoder.fit_transform(df[[ordinal_col_16]])


#We then use One hot encoding a number of columns. This is because they all contain nominal categorical data.
df = pd.get_dummies(df, columns=[
    'Does your employer provide mental health benefits as part of healthcare coverage?'
    ,'Do you know the options for mental health care available under your employer-provided coverage?'
    ,'Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?'
    ,'Does your employer offer resources to learn more about mental health concerns and options for seeking help?'
    ,'Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?'
    ,'Do you think that discussing a mental health disorder with your employer would have negative consequences?'
    ,'Do you think that discussing a physical health issue with your employer would have negative consequences?'
    ,'Would you feel comfortable discussing a mental health disorder with your coworkers?'
    ,'Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?'
    ,'Do you feel that your employer takes mental health as seriously as physical health?'
    ,'Have you heard of or observed negative consequences for co-workers who have been open about mental health issues in your workplace?'
    ,'Have your previous employers provided mental health benefits?'
    ,'Were you aware of the options for mental health care provided by your previous employers?'
    ,'Did your previous employers ever formally discuss mental health (as part of a wellness campaign or other official communication)?'
    ,'Was your anonymity protected if you chose to take advantage of mental health or substance abuse treatment resources with previous employers?'
    ,'Do you think that discussing a mental health disorder with previous employers would have negative consequences?'
    ,'Would you have been willing to discuss a mental health issue with your direct supervisor(s)?'
    ,'Did you feel that your previous employers took mental health as seriously as physical health?'
    ,'Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your current or previous workplace?'
    ,'Do you have a family history of mental illness?'
    ,'What is your gender?'
    # ,'What country do you live in?'
    # ,'What country do you work in?'
    ]
    ,dtype=int) # This ensures output is 0/1 instead of True/False



#
# ---- Feature Selection using feature correlation ----
#
corr_df = df.corr().abs()
mask =np.triu(np.ones_like(corr_df, dtype=bool))
tri_df = corr_df.mask(mask)
to_drop = [c for c in tri_df.columns if any(tri_df[c]>0.9)]
df = df.drop(to_drop, axis = 1)


#
# ---- Scaling the entire dataset ----
#
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
scaled_df = pd.DataFrame(X_scaled, columns=df.columns)



#
# ----Find the optimal number of PCs for PCA ----
#
pca = PCA()
pca.fit(scaled_df)
#Visualises the variances of the PCs
plt.figure(figsize = (10,6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker = 'o')
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.title('Explained variance by the PCA Components')
plt.grid(True)
plt.show()

#
# ---- Fit and transform the scaled dataset using PCA. 65 PCs is the optimal according to the figure created above.----
#
X_pca = PCA(n_components = 65).fit_transform(scaled_df)

#
# ----elbow method ----
#
inertia = []
for i in range (1,8):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(X_pca)
    inertia.append(kmeans.inertia_)
plt.plot(range(1,8), inertia )
plt.show()


#
# ---- Clustering ----
#

n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(X_pca)

labels = kmeans.labels_
centers = kmeans.cluster_centers_

# ---- Silhouette Analysis ----
silhouette_vals = silhouette_samples(X_pca, labels)
silhouette_avg = silhouette_score(X_pca, labels)

plt.figure(figsize=(10, 6))
y_lower = 10

for i in range(n_clusters):
    cluster_vals = silhouette_vals[labels == i]
    cluster_vals.sort()
    size = cluster_vals.shape[0]
    y_upper = y_lower + size

    plt.fill_betweenx(np.arange(y_lower, y_upper),
                      0, cluster_vals,
                      alpha=0.7, label=f'Cluster {i + 1}')
    y_lower = y_upper + 10

plt.axvline(silhouette_avg, color="red", linestyle="--", label='Avg Silhouette Score')
plt.title("Silhouette Plot for KMeans Clustering")
plt.xlabel("Silhouette Coefficient")
plt.ylabel("Cluster")
plt.legend()
plt.tight_layout()
plt.show()

#
# ---- Cluster Profiling ----
#

# scaled_df is your full scaled and encoded dataset
# This assigns the cluster label to each original row
scaled_df['cluster'] = labels + 1  # optional +1 for 1-based cluster IDs

# Get mean values of each feature per cluster
group_df = scaled_df.groupby('cluster').mean().reset_index()


#
# ---- Plot Profile of Each Cluster (Top 5 & Bottom 5 Sorted Features) ----
#

for i in group_df['cluster'].unique():
    # Extract and transpose data for plotting
    example_df = group_df[group_df['cluster'] == i].T
    example_df.columns = ['Value']
    example_df.drop(index='cluster', inplace=True)

    # Sort by value
    sorted_df = example_df.sort_values(by='Value', ascending=False)

    # Select top 5 and bottom 5
    top_bottom_df = pd.concat([sorted_df.head(5), sorted_df.tail(5)])

    # Sort again for nicer plotting
    top_bottom_df = top_bottom_df.sort_values(by='Value', ascending=False)

    # Plot
    top_bottom_df.plot(kind='bar', legend=False, color='skyblue')
    plt.title(f'Top/Bottom Features - Cluster {i}')
    plt.ylabel('Mean Feature Value')
    plt.xticks(rotation=45, ha='right')
    plt.show()


