# =================================== IMPORTS ================================= #

import os
import sys

# import json
import re
import numpy as np 
import pandas as pd 
from datetime import datetime, timedelta
from collections import Counter

# import seaborn as sns 
import plotly.graph_objects as go
import plotly.express as px

import time
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import folium
from folium.plugins import MousePosition

import dash
from dash import dcc, html, dash_table

# Google Web Credentials
import json
import base64
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# 'data/~$bmhc_data_2024_cleaned.xlsx'
# print('System Version:', sys.version)
# =================================== DATA ==================================== #

print("="*50)
print("PYTHON INTERPRETER DEBUG INFO:")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path[0]}")
print("="*50)

current_dir = os.getcwd()
current_file = os.path.basename(__file__)
script_dir = os.path.dirname(os.path.abspath(__file__))
# data_path = 'data/Navigation_Responses.xlsx'
# file_path = os.path.join(script_dir, data_path)
# data = pd.read_excel(file_path)
# df = data.copy()

# Define the Google Sheets URL
sheet_url = "https://docs.google.com/spreadsheets/d/1bpnjogZp1gSIfaM6wRZohShlG1WHoMNko1HxPNH61jU/edit?resourcekey=&gid=1836855474#gid=1836855474"

# Define the scope
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# Load credentials
encoded_key = os.getenv("GOOGLE_CREDENTIALS")

if encoded_key:
    json_key = json.loads(base64.b64decode(encoded_key).decode("utf-8"))
    creds = ServiceAccountCredentials.from_json_keyfile_dict(json_key, scope)
else:
    creds_path = r"C:\Users\CxLos\OneDrive\Documents\BMHC\Data\bmhc-timesheet-4808d1347240.json"
    if os.path.exists(creds_path):
        creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
    else:
        raise FileNotFoundError("Service account JSON file not found and GOOGLE_CREDENTIALS is not set.")

# Authorize and load the sheet
client = gspread.authorize(creds)
sheet = client.open_by_url(sheet_url)
worksheet = sheet.worksheet("Form Responses 1")
data = pd.DataFrame(worksheet.get_all_records())
df = data.copy()

# Filtered df where 'Date of Activity:' is in May
# df["Date of Activity"] = pd.to_datetime(df["Date of Activity"], errors='coerce')
# df["Date of Activity"] = df["Date of Activity"].dt.tz_localize('UTC')  # or local timezone first, then convert to UTC
# df = df[df['Date of Activity'].dt.month == 7]

# Get the reporting month:
int_month = 10
mo = "Oct"
report_month = datetime(2025, 10, 1).strftime("%B")
report_year = datetime(2025, 10, 1).year

# Strip whitespace from string entries in the whole DataFrame
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
    
df["Date of Activity"] = pd.to_datetime(df["Date of Activity"], errors='coerce')
df = df[(df['Date of Activity'].dt.month == int_month) & (df['Date of Activity'].dt.year == int(report_year))]

# df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Define a discrete color sequence
# color_sequence = px.colors.qualitative.Plotly

# -----------------------------------------------
# print(df.head())
# print('Total entries: ', len(df))
# print('Column Names: \n', df.columns.tolist())
# print('DF Shape:', df.shape)
# print('Dtypes: \n', df.dtypes)
# print('Info:', df.info())
# print("Amount of duplicate rows:", df.duplicated().sum())

# print('Current Directory:', current_dir)
# print('Script Directory:', script_dir)
# print('Path to data:',file_path)

# ================================= Columns Navigation ================================= #

columns = [
    'Timestamp', 
    'Date of Activity',
    'First Name', 
    'Last Name', 
    'E-mail',
    'Phone number', 
    'Column 17',
    # ----------------------
    'Birthdate', 
    'Age', 
    'Gender',
    'Race', 
    'Systolic Blood Pressure', 
    'Diastolic Blood Pressure',
    'Heart Rate', 
    'Topics of Interest', 
    'Learning Outcome', 
    'Are you Interested in continuing MIM?', 
    'BMHC Enrollment',
    'ZIP Code'
  ]

# ============================== Data Preprocessing ========================== #

df.rename(
    columns={
        "Systolic Blood Pressure" : "Systolic",
        "Diastolic Blood Pressure" : "Diastolic",
        "Heart Rate" : "HR",
        # ----------------------------------------
        "Birthdate" : "Birthdate",
        "Age" : "Age",
        "Gender" : "Gender",
        "Race" : "Ethnicity",
        "Topics of Interest" : "Interest",
        "Learning Outcome" : "Outcome",
        "Are you Interested in continuing MIM?" : "Continue",
        "BMHC Enrollment" : "Enrollment",
        "ZIP Code" : "ZIP Code",
    },
    inplace=True
)


# ================ Clients Serviced ============== #

# # Clients Serviced:
df_len = len(df)
df_len = str(df_len)
# print(f'MIM Participants {report_month}:', df_len)

# ================ Systolic ============== #

# df['Systolic'] = df['Systolic'].astype(str).str.strip().replace({'': np.nan})

# Exclude null values:
df['Systolic'] = pd.to_numeric(df['Systolic'], errors='coerce')
# print('Systolic Unique:', df['Systolic'].unique().tolist())

# Average Systolic Blood Pressure:
systolic_avg = df['Systolic'].mean()
systolic_avg = round(systolic_avg)

# ================ Diastolic ============== #

# df['Diastolic'] = df['Diastolic'].astype(str).str.strip().replace({'': np.nan})

df['Diastolic'] = pd.to_numeric(df['Diastolic'], errors='coerce')
# print('Diastolic Unique:', df['Diastolic'].unique().tolist())

# Diastolic Blood Pressure:
diastolic_avg = df['Diastolic'].mean()
diastolic_avg = round(diastolic_avg)
# print('Diastolic Average:', diastolic_avg)

# ================ HR ============== #

# print("Heart Rate Unique Before:", df['HR'].unique().tolist())

df['HR'] = df['HR'].astype(str).str.strip().replace({'': np.nan})
df['HR'] = pd.to_numeric(df['HR'], errors='coerce')

# df['HR'] = (
#     df['HR']
#     .astype(str)
#     .str.strip()
#     .replace({
#         "": ""
#     })
# )

# Average Heart Rate:
hr_avg = df['HR'].mean(skipna=True)
hr_avg = round(hr_avg)
# print('Heart Rate Average:', hr_avg)

# ------------------------------- Race Graphs ---------------------------- #

df['Ethnicity'] = (
    df['Ethnicity']
        .astype(str)
        .str.strip()
        .replace({
            "Hispanic/Latino": "Hispanic/ Latino", 
            "White": "White/ European Ancestry", 
            "Group search": "N/A", 
            "Group search": "N/A", 
        })
)

# Groupby Race/Ethnicity:
df_race = df['Ethnicity'].value_counts().reset_index(name='Count')

# Race Bar Chart
race_bar=px.bar(
    df_race,
    x='Ethnicity',
    y='Count',
    color='Ethnicity',
    text='Count',
).update_layout(
    # height=700, 
    title=dict(
        text='Race Distribution Bar Chart',
        x=0.5, 
        font=dict(
            size=25,
            family='Calibri',
            color='black',
            )
    ),
    font=dict(
        family='Calibri',
        size=18,
        color='black'
    ),
    xaxis=dict(
        tickangle=-20,  # Rotate x-axis labels for better readability
        tickfont=dict(size=18),  # Adjust font size for the tick labels
        showticklabels=False,  # Hide x-tick labels
        title=dict(
            # text=None,
            text="Race/ Ethnicity",
            font=dict(size=20),  # Font size for the title
        ),
    ),
    yaxis=dict(
        title=dict(
            text='Count',
            font=dict(size=20),  # Font size for the title
        ),
    ),
    legend=dict(
        title='',
        orientation="v",  # Vertical legend
        x=1.05,  # Position legend to the right
        y=1,  # Position legend at the top
        xanchor="left",  # Anchor legend to the left
        yanchor="top",  # Anchor legend to the top
        # visible=False,
        visible=True,
    ),
    hovermode='closest', # Display only one hover label per trace
    bargap=0.07,  # Reduce the space between bars
    bargroupgap=0,  # Reduce space between individual bars in groups
).update_traces(
    textposition='auto',
    hovertemplate='<b>Race:</b> %{label}<br><b>Count</b>: %{y}<extra></extra>'
)

# Race Pie Chart
race_pie=px.pie(
    df_race,
    names='Ethnicity',
    values='Count'
).update_layout(
    # height=700, 
    title='Race Distribution Pie Chart',
    title_x=0.5,
    font=dict(
        family='Calibri',
        size=17,
        color='black'
    )
).update_traces(
    # textinfo='value+percent',
    texttemplate='%{value}<br>(%{percent:.2%})',
    hovertemplate='<b>%{label}</b>: %{value}<extra></extra>'
)

# ------------------------------- Gender Distribution ---------------------------- #

# print("Gender Unique Before:", df['Gender'].unique().tolist())

gender_unique =[
    'Male', 
    'Transgender', 
    'Female', 
    'Group search ', 
    'Prefer Not to Say'
]

# print("Gender Value Counts Before: \n", df_gender)

df['Gender'] = (
    df['Gender']
        .astype(str)
            .str.strip()
            .replace({
                "Group search": "N/A", 
            })
)

# Groupby 'Gender:'
df_gender = df['Gender'].value_counts().reset_index(name='Count')

# print("Gender Unique After:", df['Gender'].unique().tolist())
# print("Gender Value Counts After: \n", df_gender)

# Gender Bar Chart
gender_bar=px.bar(
    df_gender,
    x='Gender',
    y='Count',
    color='Gender',
    text='Count',
).update_layout(
    # height=700, 
    # width=1000,
    title=dict(
        text='Sex Distribution Bar Chart',
        x=0.5, 
        font=dict(
            size=25,
            family='Calibri',
            color='black',
            )
    ),
    font=dict(
        family='Calibri',
        size=18,
        color='black'
    ),
    xaxis=dict(
        tickangle=0,  # Rotate x-axis labels for better readability
        tickfont=dict(size=18),  # Adjust font size for the tick labels
        title=dict(
            # text=None,
            text="Gender",
            font=dict(size=20),  # Font size for the title
        ),
    ),
    yaxis=dict(
        title=dict(
            text='Count',
            font=dict(size=20),  # Font size for the title
        ),
    ),
    legend=dict(
        title='',
        orientation="v",  # Vertical legend
        x=1.05,  # Position legend to the right
        y=1,  # Position legend at the top
        xanchor="left",  # Anchor legend to the left
        yanchor="top",  # Anchor legend to the top
        visible=False
        
    ),
    hovermode='closest', # Display only one hover label per trace
    bargap=0.07,  # Reduce the space between bars
    bargroupgap=0,  # Reduce space between individual bars in groups
).update_traces(
    textposition='auto',
    hovertemplate='<b>Gender</b>: %{label}<br><b>Count</b>: %{y}<extra></extra>'
)

# Gender Pie Chart
gender_pie=px.pie(
    df,
    names='Gender'
).update_layout(
    # height=700,
    title='Patient Visits by Sex',
    title_x=0.5,
    font=dict(
        family='Calibri',
        size=17,
        color='black'
    )
).update_traces(
    # textinfo='value+percent',
    texttemplate='%{value}<br>(%{percent:.2%})',
    hovertemplate='<b>%{label} Visits</b>: %{value}<extra></extra>'
)

# ------------------------------- Age Distribution ---------------------------- #

print("Age null:", df['Age'].isnull().sum())

df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
# print("Age Unique Before:", df['Age'].unique().tolist())

# # Define a function to categorize ages into age groups
def categorize_age(age):
    # treat missing values explicitly as 'N/A'
    if pd.isna(age):
        return 'N/A'
    try:
        age = int(age)
    except (TypeError, ValueError):
        return 'N/A'

    if age < 10:
        return '0-9'
    elif 10 <= age <= 19:
        return '10-19'
    elif 20 <= age <= 29:
        return '20-29'
    elif 30 <= age <= 39:
        return '30-39'
    elif 40 <= age <= 49:
        return '40-49'
    elif 50 <= age <= 59:
        return '50-59'
    elif 60 <= age <= 69:
        return '60-69'
    elif 70 <= age <= 79:
        return '70-79'
    elif age >= 80:
        return '80+'
    else:
        return 'N/A'

# # Apply the function to create the 'Age_Group' column
df['Age_Group'] = df['Age'].apply(categorize_age)
# print("Age Group Value Counts:", df['Age_Group'].value_counts())

# # Group by 'Age_Group' and count the number of patient visits
df_decades = df.groupby('Age_Group',  observed=True).size().reset_index(name='Patient_Visits')

# # Sort the result by the minimum age in each group
age_order = [
            '10-19',
             '20-29', 
             '30-39', 
             '40-49', 
             '50-59', 
             '60-69', 
             '70-79',
             '80+'
             ]

df_decades['Age_Group'] = pd.Categorical(df_decades['Age_Group'], categories=age_order, ordered=True)
df_decades = df_decades.sort_values('Age_Group')
# print(df_decades.value_counts())

# Age Bar Chart
age_bar=px.bar(
    df_decades,
    x='Age_Group',
    y='Patient_Visits',
    color='Age_Group',
    text='Patient_Visits',
).update_layout(
    # height=700, 
    # width=1000,
    title=dict(
        text='Client Age Distribution',
        x=0.5, 
        font=dict(
            size=25,
            family='Calibri',
            color='black',
            )
    ),
    font=dict(
        family='Calibri',
        size=18,
        color='black'
    ),
    xaxis=dict(
        tickangle=0,  # Rotate x-axis labels for better readability
        tickfont=dict(size=18),  # Adjust font size for the tick labels
        title=dict(
            # text=None,
            text="Age Group",
            font=dict(size=20),  # Font size for the title
        ),
    ),
    yaxis=dict(
        title=dict(
            text='Number of Visits',
            font=dict(size=20),  # Font size for the title
        ),
    ),
    legend=dict(
        title_text='',
        orientation="v",  # Vertical legend
        x=1.05,  # Position legend to the right
        y=1,  # Position legend at the top
        xanchor="left",  # Anchor legend to the left
        yanchor="top",  # Anchor legend to the top
        visible=False
    ),
    hovermode='closest', # Display only one hover label per trace
    bargap=0.08,  # Reduce the space between bars
    bargroupgap=0,  # Reduce space between individual bars in groups
).update_traces(
    textposition='auto',
    hovertemplate='<b>Age:</b>: %{label}<br><b>Count</b>: %{y}<extra></extra>'
)

age_pie = px.pie(
    df_decades,
    names='Age_Group',
    values='Patient_Visits',
).update_layout(
    # height=700, 
    title='Client Age Distribution',
    title_x=0.5,
    font=dict(
        family='Calibri',
        size=17,
        color='black'
    )
).update_traces(
    rotation=190,
    texttemplate='%{value}<br>(%{percent:.2%})',
    hovertemplate='<b>%{label}</b>: %{value}<extra></extra>'
)

# ==================== Topics of Interest ==================== #

# Clean and normalize the 'Interest' column
df['Interest'] = (
    df['Interest']
    .astype(str)
    .str.strip()
    .replace({
        "": "N/A", 
    })
)

# Split multiple interests (if comma-separated), flatten, and count
all_interests = []
for entry in df['Interest']:
    items = [i.strip() for i in str(entry).split(",") if i.strip() and i.strip().lower() != "n/a"]
    all_interests.extend(items)

interest_counter = Counter(all_interests)
df_interest = pd.DataFrame(interest_counter.items(), columns=['Interest', 'Count']).sort_values(by='Count', ascending=False)

# Bar Chart
interest_bar = px.bar(
    df_interest,
    x='Interest',
    y='Count',
    color='Interest',
    text='Count',
).update_layout(
    title=dict(
        text='Topics of Interest Bar Chart',
        x=0.5,
        font=dict(
            size=25,
            family='Calibri',
            color='black',
        )
    ),
    font=dict(
        family='Calibri',
        size=18,
        color='black'
    ),
    xaxis=dict(
        tickangle=-20,
        tickfont=dict(size=18),
        title=dict(
            text="Interest",
            font=dict(size=20),
        ),
    ),
    yaxis=dict(
        title=dict(
            text='Count',
            font=dict(size=20),
        ),
    ),
    legend=dict(
        title='',
        orientation="v",
        x=1.05,
        y=1,
        xanchor="left",
        yanchor="top",
        visible=True,
    ),
    hovermode='closest',
    bargap=0.07,
    bargroupgap=0,
).update_traces(
    textposition='auto',
    hovertemplate='<b>Interest:</b> %{label}<br><b>Count</b>: %{y}<extra></extra>'
)

# Pie Chart
interest_pie = px.pie(
    df_interest,
    names='Interest',
    values='Count'
).update_layout(
    title='Topics of Interest Pie Chart',
    title_x=0.5,
    font=dict(
        family='Calibri',
        size=17,
        color='black'
    )
).update_traces(
    texttemplate='%{value}<br>(%{percent:.2%})',
    hovertemplate='<b>%{label}</b>: %{value}<extra></extra>'
)

# ================ Learning Outcome ============== #

# Clean and normalize the 'Outcome' column
df['Outcome'] = (
    df['Outcome']
    .astype(str)
    .str.strip()
    .replace({"": "N/A", "nan": "N/A", "None": "N/A"})
)

df_outcome = df['Outcome'].value_counts().reset_index(name='Count')
df_outcome.rename(columns={'index': 'Outcome'}, inplace=True)

# Bar Chart
outcome_bar = px.bar(
    df_outcome,
    x='Outcome',
    y='Count',
    color='Outcome',
    text='Count',
).update_layout(
    title=dict(
        text='Learning Outcome Bar Chart',
        x=0.5,
        font=dict(
            size=25,
            family='Calibri',
            color='black',
        )
    ),
    font=dict(
        family='Calibri',
        size=18,
        color='black'
    ),
    xaxis=dict(
        tickangle=0,
        tickfont=dict(size=18),
        title=dict(
            text="Learning Outcome",
            font=dict(size=20),
        ),
    ),
    yaxis=dict(
        title=dict(
            text='Count',
            font=dict(size=20),
        ),
    ),
    legend=dict(
        title='',
        orientation="v",
        x=1.05,
        y=1,
        xanchor="left",
        yanchor="top",
        visible=True,
    ),
    hovermode='closest',
    bargap=0.07,
    bargroupgap=0,
).update_traces(
    textposition='auto',
    hovertemplate='<b>Outcome:</b> %{label}<br><b>Count</b>: %{y}<extra></extra>'
)

# Pie Chart
outcome_pie = px.pie(
    df_outcome,
    names='Outcome',
    values='Count'
).update_layout(
    title='Learning Outcome Pie Chart',
    title_x=0.5,
    font=dict(
        family='Calibri',
        size=17,
        color='black'
    )
).update_traces(
    texttemplate='%{value}<br>(%{percent:.2%})',
    hovertemplate='<b>%{label}</b>: %{value}<extra></extra>'
)

# ================ Continuing MIM? ============== #

# Clean and normalize the 'Continue' column
df['Continue'] = (
    df['Continue']
    .astype(str)
    .str.strip()
    .replace({"": "N/A", "nan": "N/A", "None": "N/A"})
)

df_continue = df['Continue'].value_counts().reset_index(name='Count')
df_continue.rename(columns={'index': 'Continue'}, inplace=True)

# Bar Chart
continue_bar = px.bar(
    df_continue,
    x='Continue',
    y='Count',
    color='Continue',
    text='Count',
).update_layout(
    title=dict(
        text='Interested in Continuing MIM Bar Chart',
        x=0.5,
        font=dict(
            size=25,
            family='Calibri',
            color='black',
        )
    ),
    font=dict(
        family='Calibri',
        size=18,
        color='black'
    ),
    xaxis=dict(
        tickangle=0,
        tickfont=dict(size=18),
        title=dict(
            text="Continue MIM?",
            font=dict(size=20),
        ),
    ),
    yaxis=dict(
        title=dict(
            text='Count',
            font=dict(size=20),
        ),
    ),
    legend=dict(
        title='',
        orientation="v",
        x=1.05,
        y=1,
        xanchor="left",
        yanchor="top",
        visible=True,
    ),
    hovermode='closest',
    bargap=0.07,
    bargroupgap=0,
).update_traces(
    textposition='auto',
    hovertemplate='<b>Continue:</b> %{label}<br><b>Count</b>: %{y}<extra></extra>'
)

# Pie Chart
continue_pie = px.pie(
    df_continue,
    names='Continue',
    values='Count'
).update_layout(
    title='Interested in Continuing MIM Pie Chart',
    title_x=0.5,
    font=dict(
        family='Calibri',
        size=17,
        color='black'
    )
).update_traces(
    texttemplate='%{value}<br>(%{percent:.2%})',
    hovertemplate='<b>%{label}</b>: %{value}<extra></extra>'
)

# ================ Continuing MIM? ============== #

# Clean and normalize the 'Continue' column
df['Continue'] = (
    df['Continue']
    .astype(str)
    .str.strip()
    .replace({"": "N/A", "nan": "N/A", "None": "N/A"})
)

df_continue = df['Continue'].value_counts().reset_index(name='Count')
df_continue.rename(columns={'index': 'Continue'}, inplace=True)

# Bar Chart
continue_bar = px.bar(
    df_continue,
    x='Continue',
    y='Count',
    color='Continue',
    text='Count',
).update_layout(
    title=dict(
        text='Interested in Continuing MIM Bar Chart',
        x=0.5,
        font=dict(
            size=25,
            family='Calibri',
            color='black',
        )
    ),
    font=dict(
        family='Calibri',
        size=18,
        color='black'
    ),
    xaxis=dict(
        tickangle=0,
        tickfont=dict(size=18),
        title=dict(
            text="Continue MIM?",
            font=dict(size=20),
        ),
    ),
    yaxis=dict(
        title=dict(
            text='Count',
            font=dict(size=20),
        ),
    ),
    legend=dict(
        title='',
        orientation="v",
        x=1.05,
        y=1,
        xanchor="left",
        yanchor="top",
        visible=True,
    ),
    hovermode='closest',
    bargap=0.07,
    bargroupgap=0,
).update_traces(
    textposition='auto',
    hovertemplate='<b>Continue:</b> %{label}<br><b>Count</b>: %{y}<extra></extra>'
)

# Pie Chart
continue_pie = px.pie(
    df_continue,
    names='Continue',
    values='Count'
).update_layout(
    title='Interested in Continuing MIM Pie Chart',
    title_x=0.5,
    font=dict(
        family='Calibri',
        size=17,
        color='black'
    )
).update_traces(
    texttemplate='%{value}<br>(%{percent:.2%})',
    hovertemplate='<b>%{label}</b>: %{value}<extra></extra>'
)



# ==================== Enrollment ==================== #

# Clean and normalize the 'Enrollment' column
df['Enrollment'] = (
    df['Enrollment']
    .astype(str)
    .str.strip()
    .replace({"": "N/A", "nan": "N/A", "None": "N/A"})
)

# Value counts for Enrollment
df_enroll = df['Enrollment'].value_counts().reset_index(name='Count')
df_enroll.rename(columns={'index': 'Enrollment'}, inplace=True)

# Bar Chart
enroll_bar = px.bar(
    df_enroll,
    x='Enrollment',
    y='Count',
    color='Enrollment',
    text='Count',
).update_layout(
    title=dict(
        text='BMHC Enrollment Bar Chart',
        x=0.5,
        font=dict(
            size=25,
            family='Calibri',
            color='black',
        )
    ),
    font=dict(
        family='Calibri',
        size=18,
        color='black'
    ),
    xaxis=dict(
        tickangle=0,
        tickfont=dict(size=18),
        title=dict(
            text="Enrollment",
            font=dict(size=20),
        ),
    ),
    yaxis=dict(
        title=dict(
            text='Count',
            font=dict(size=20),
        ),
    ),
    legend=dict(
        title='',
        orientation="v",
        x=1.05,
        y=1,
        xanchor="left",
        yanchor="top",
        visible=True,
    ),
    hovermode='closest',
    bargap=0.07,
    bargroupgap=0,
).update_traces(
    textposition='auto',
    hovertemplate='<b>Enrollment:</b> %{label}<br><b>Count</b>: %{y}<extra></extra>'
)

# Pie Chart
enroll_pie = px.pie(
    df_enroll,
    names='Enrollment',
    values='Count'
).update_layout(
    title='BMHC Enrollment Pie Chart',
    title_x=0.5,
    font=dict(
        family='Calibri',
        size=17,
        color='black'
    )
).update_traces(
    texttemplate='%{value}<br>(%{percent:.2%})',
    hovertemplate='<b>%{label}</b>: %{value}<extra></extra>'
)



# ---------------------- Zip 2 --------------------- #

# df['ZIP2'] = df['ZIP Code:']
# print('ZIP2 Unique Before: \n', df['ZIP2'].unique().tolist())

# zip2_unique =[
# 78753, '', 78721, 78664, 78725, 78758, 78724, 78660, 78723, 78748, 78744, 78752, 78745, 78617, 78754, 78653, 78727, 78747, 78659, 78759, 78741, 78616, 78644, 78757, 'UnKnown', 'Unknown', 'uknown', 'Unknown ', 78729
# ]

# zip2_mode = df['ZIP2'].mode()[0]

# df['ZIP2'] = (
#     df['ZIP2']
#     .astype(str)
#     .str.strip()
#     .replace({
#         'Texas': zip2_mode,
#         'Unhoused': zip2_mode,
#         'UNHOUSED': zip2_mode,
#         'UnKnown': zip2_mode,
#         'Unknown': zip2_mode,
#         'uknown': zip2_mode,
#         'Unknown': zip2_mode,
#         'NA': zip2_mode,
#         'nan': zip2_mode,
#         '': zip2_mode,
#         ' ': zip2_mode,
#     })
# )

# df['ZIP2'] = df['ZIP2'].fillna(zip2_mode)
# df_z = df['ZIP2'].value_counts().reset_index(name='Count')

# print('ZIP2 Unique After: \n', df_z['ZIP2'].unique().tolist())
# print('ZIP2 Value Counts After: \n', df_z['ZIP2'].value_counts())

df['ZIP2'] = df['ZIP Code'].astype(str).str.strip()

valid_zip_mask = df['ZIP2'].str.isnumeric()
zip2_mode = df.loc[valid_zip_mask, 'ZIP2'].mode()[0]  # still a string

invalid_zip_values = [
    'Texas', 'Unhoused', 'UNHOUSED', 'UnKnown', 'Unknown', 'uknown',
    'Unknown ', 'NA', 'nan', 'NaN', 'None', '', ' '
]
df['ZIP2'] = df['ZIP2'].replace(invalid_zip_values, zip2_mode)

# Step 3: Coerce to numeric, fill any remaining NaNs, then convert back to string
df['ZIP2'] = pd.to_numeric(df['ZIP2'], errors='coerce')
df['ZIP2'] = df['ZIP2'].fillna(int(zip2_mode)).astype(int).astype(str)

# Step 4: Create value count dataframe for the bar chart
df_z = df['ZIP2'].value_counts().reset_index(name='Count')
df_z.columns = ['ZIP2', 'Count']  # Rename columns for Plotly

df_z['Percentage'] = (df_z['Count'] / df_z['Count'].sum()) * 100
df_z['text_label'] = df_z['Count'].astype(str) + ' (' + df_z['Percentage'].round(1).astype(str) + '%)'
# df_z['text_label'] = df_z['Percentage'].round(1).astype(str) + '%'


zip_fig =px.bar(
    df_z,
    x='Count',
    y='ZIP2',
    color='ZIP2',
    text='text_label',
    # text='Count',
    orientation='h'  # Horizontal bar chart
).update_layout(
    title='Number of Clients by Zip Code',
    xaxis_title='Residents',
    yaxis_title='Zip Code',
    title_x=0.5,
    # height=950,
    # width=1500,
    font=dict(
        family='Calibri',
        size=17,
        color='black'
    ),
        yaxis=dict(
        tickangle=0  # Keep y-axis labels horizontal for readability
    ),
        legend=dict(
        title='ZIP Code',
        orientation="v",  # Vertical legend
        x=1.05,  # Position legend to the right
        xanchor="left",  # Anchor legend to the left
        y=1,  # Position legend at the top
        yanchor="top"  # Anchor legend at the top
    ),
).update_traces(
    textposition='auto',  # Place text labels inside the bars
    textfont=dict(size=30),  # Increase text size in each bar
    # insidetextanchor='middle',  # Center text within the bars
    textangle=0,            # Ensure text labels are horizontal
    hovertemplate='<b>ZIP Code</b>: %{y}<br><b>Count</b>: %{x}<extra></extra>'
)

zip_pie = px.pie(
    df_z,
    names='ZIP2',
    values='Count',
    title='Client Distribution by ZIP Code',
    color_discrete_sequence=px.colors.qualitative.Safe
).update_layout(
    title_x=0.5,
    # height=700,
    # width=900,
    font=dict(
        family='Calibri',
        size=17,
        color='black'
    ),
    legend_title='ZIP Code'
).update_traces(
    textinfo='percent+label',
    textfont_size=16,
    hovertemplate='<b>ZIP Code</b>: %{label}<br><b>Count</b>: %{value}<br><b>Percent</b>: %{percent}<extra></extra>'
)

# -----------------------------------------------------------------------------

# Get the distinct values in column

# distinct_service = df['What service did/did not complete?'].unique()
# print('Distinct:\n', distinct_service)

# =============================== Folium ========================== #

# empty_strings = df[df['ZIP Code:'].str.strip() == ""]
# print("Empty strings: \n", empty_strings.iloc[:, 10:12])

# Filter df to exclued all rows where there is no value for "ZIP Code:"
# df = df[df['ZIP Code:'].str.strip() != ""]

mode_value = df['ZIP Code'].mode()[0]
df['ZIP Code'] = df['ZIP Code'].fillna(mode_value)

# print("ZIP value counts:", df['ZIP Code:'].value_counts())
# print("Zip Unique Before: \n", df['ZIP Code:'].unique().tolist())

# Check for non-numeric values in the 'ZIP Code:' column
# print("ZIP non-numeric values:", df[~df['ZIP Code:'].str.isnumeric()]['ZIP Code:'].unique())

# df['ZIP Code:'] = df['ZIP Code:'].astype(str).str.strip()

# df['ZIP Code:'] = (
#     df['ZIP Code:']
#     .astype(str).str.strip()
#         .replace({
#             'Texas': mode_value,
#             'Unhoused': mode_value,
#             'unknown': mode_value,
#             'Unknown': mode_value,
#             'UnKnown': mode_value,
#             'uknown': mode_value,
#             'NA': mode_value,
#             "": mode_value,
#             'nan': mode_value
# }))

# df['ZIP Code:'] = df['ZIP Code:'].where(df['ZIP Code:'].str.isdigit(), mode_value)
# df['ZIP Code:'] = df['ZIP Code:'].astype(int)

# df_zip = df['ZIP Code:'].value_counts().reset_index(name='Residents')
# # df_zip['ZIP Code:'] = df_zip['index'].astype(int)
# df_zip['Residents'] = df_zip['Residents'].astype(int)
# # df_zip.drop('index', axis=1, inplace=True)

# # print("Zip Unique After: \n", df['ZIP Code:'].unique().tolist())

# # print(df_zip.head())

# # Create a folium map
# m = folium.Map([30.2672, -97.7431], zoom_start=10)

# # Add different tile sets
# folium.TileLayer('OpenStreetMap', attr='© OpenStreetMap contributors').add_to(m)
# folium.TileLayer('Stamen Terrain', attr='Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL.').add_to(m)
# folium.TileLayer('Stamen Toner', attr='Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL.').add_to(m)
# folium.TileLayer('Stamen Watercolor', attr='Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL.').add_to(m)
# folium.TileLayer('CartoDB positron', attr='Map tiles by CartoDB, under CC BY 3.0. Data by OpenStreetMap, under ODbL.').add_to(m)
# folium.TileLayer('CartoDB dark_matter', attr='Map tiles by CartoDB, under CC BY 3.0. Data by OpenStreetMap, under ODbL.').add_to(m)

# # Available map styles
# map_styles = {
#     'OpenStreetMap': {
#         'tiles': 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
#         'attribution': '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
#     },
#     'Stamen Terrain': {
#         'tiles': 'https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.jpg',
#         'attribution': 'Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under ODbL.'
#     },
#     'Stamen Toner': {
#         'tiles': 'https://stamen-tiles.a.ssl.fastly.net/toner/{z}/{x}/{y}.png',
#         'attribution': 'Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under ODbL.'
#     },
#     'Stamen Watercolor': {
#         'tiles': 'https://stamen-tiles.a.ssl.fastly.net/watercolor/{z}/{x}/{y}.jpg',
#         'attribution': 'Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under ODbL.'
#     },
#     'CartoDB positron': {
#         'tiles': 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png',
#         'attribution': '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
#     },
#     'CartoDB dark_matter': {
#         'tiles': 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
#         'attribution': '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
#     },
#     'ESRI Imagery': {
#         'tiles': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
#         'attribution': 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
#     }
# }

# # Add tile layers to the map
# for style, info in map_styles.items():
#     folium.TileLayer(tiles=info['tiles'], attr=info['attribution'], name=style).add_to(m)

# # Select a style
# # selected_style = 'OpenStreetMap'
# # selected_style = 'Stamen Terrain'
# # selected_style = 'Stamen Toner'
# # selected_style = 'Stamen Watercolor'
# selected_style = 'CartoDB positron'
# # selected_style = 'CartoDB dark_matter'
# # selected_style = 'ESRI Imagery'

# # Apply the selected style
# if selected_style in map_styles:
#     style_info = map_styles[selected_style]
#     # print(f"Selected style: {selected_style}")
#     folium.TileLayer(
#         tiles=style_info['tiles'],
#         attr=style_info['attribution'],
#         name=selected_style
#     ).add_to(m)
# else:
#     print(f"Selected style '{selected_style}' is not in the map styles dictionary.")
#      # Fallback to a default style
#     folium.TileLayer('OpenStreetMap').add_to(m)
    
# geolocator = Nominatim(user_agent="your_app_name", timeout=10)

# # Function to get coordinates from zip code
# # def get_coordinates(zip_code):
# #     geolocator = Nominatim(user_agent="response_q4_2024.py", timeout=10) # Add a timeout parameter to prevent long waits
# #     location = geolocator.geocode({"postalcode": zip_code, "country": "USA"})
# #     if location:
# #         return location.latitude, location.longitude
# #     else:
# #         print(f"Could not find coordinates for zip code: {zip_code}")
# #         return None, None
    
# def get_coordinates(zip_code):
#     for _ in range(3):  # Retry up to 3 times
#         try:
#             location = geolocator.geocode({"postalcode": zip_code, "country": "USA"})
#             if location:
#                 return location.latitude, location.longitude
#         except GeocoderTimedOut:
#             time.sleep(2)  # Wait before retrying
#     return None, None  # Return None if all retries fail

# # Apply function to dataframe to get coordinates
# df_zip['Latitude'], df_zip['Longitude'] = zip(*df_zip['ZIP Code:'].apply(get_coordinates))

# # Filter out rows with NaN coordinates
# df_zip = df_zip.dropna(subset=['Latitude', 'Longitude'])
# # print(df_zip.head())
# # print(df_zip[['Zip Code', 'Latitude', 'Longitude']].head())
# # print(df_zip.isnull().sum())

# # instantiate a feature group for the incidents in the dataframe
# incidents = folium.map.FeatureGroup()

# for index, row in df_zip.iterrows():
#     lat, lng = row['Latitude'], row['Longitude']

#     if pd.notna(lat) and pd.notna(lng):  
#         incidents.add_child(# Check if both latitude and longitude are not NaN
#         folium.vector_layers.CircleMarker(
#             location=[lat, lng],
#             radius=row['Residents'] * 1.2,  # Adjust the multiplication factor to scale the circle size as needed,
#             color='blue',
#             fill=True,
#             fill_color='blue',
#             fill_opacity=0.4
#         ))

# # add pop-up text to each marker on the map
# latitudes = list(df_zip['Latitude'])
# longitudes = list(df_zip['Longitude'])

# # labels = list(df_zip[['Zip Code', 'Residents_In_Zip_Code']])
# labels = df_zip.apply(lambda row: f"ZIP Code: {row['ZIP Code:']}, Patients: {row['Residents']}", axis=1)

# for lat, lng, label in zip(latitudes, longitudes, labels):
#     if pd.notna(lat) and pd.notna(lng):
#         folium.Marker([lat, lng], popup=label).add_to(m)
 
# formatter = "function(num) {return L.Util.formatNum(num, 5);};"
# mouse_position = MousePosition(
#     position='topright',
#     separator=' Long: ',
#     empty_string='NaN',
#     lng_first=False,
#     num_digits=20,
#     prefix='Lat:',
#     lat_formatter=formatter,
#     lng_formatter=formatter,
# )

# m.add_child(mouse_position)

# # add incidents to map
# m.add_child(incidents)

# map_path = 'zip_code_map.html'
# map_file = os.path.join(script_dir, map_path)
# m.save(map_file)
# map_html = open(map_file, 'r').read()

# ========================== DataFrame Table ========================== #

df = df.sort_values('Date of Activity', ascending=True)

# create a display index column and prepare table data/columns
# reset index to ensure contiguous numbering after any filtering/sorting upstream
df_indexed = df.reset_index(drop=True).copy()
# Insert '#' as the first column (1-based row numbers)
df_indexed.insert(0, '#', df_indexed.index + 1)

# Convert to records for DataTable
data = df_indexed.to_dict('records')
columns = [{"name": col, "id": col} for col in df_indexed.columns]

df_table = go.Figure(data=[go.Table(
    columnwidth=[200] * len(df.columns),   # give each column 200px width
    header=dict(
        values=list(df.columns),
        fill_color='#34A853',
        align='center',
        height=30,
        font=dict(size=16, color='white', family='Calibri') 
    ),
    cells=dict(
        values=[df[col] for col in df.columns],
        fill_color='lavender',
        align='left',
        height=25,
        font=dict(size=12)
    )
)])

df_table.update_layout(
    autosize=False,
    width=len(df.columns) * 100,   # total width = #columns × col width
    height=900,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    shapes=[
        dict(
            type="rect",
            xref="paper", yref="paper",
            x0=0, y0=0, x1=1, y1=1,
            line=dict(color="black", width=2),
            fillcolor="rgba(0,0,0,0)"
        )
    ],
)

# ============================== Dash Application ========================== #

app = dash.Dash(__name__)
server= app.server

app.layout = html.Div(
    children=[ 
              
    # ============================ Header ========================== #
    html.Div(
        className='divv', 
        children=[ 
            html.H1(
                'Movement is Medicine Report', 
                className='title'
            ),
            html.H1(
                f'{report_month} {report_year}', 
                className='title2'
            ),
            html.Div(
                className='btn-box', 
                children=[
                    html.A(
                        'Repo',
                        href=f'https://github.com/CxLos/MIM_{report_month}_{report_year}',
                        className='btn'
                    ),
                ]
            ),
        ]
    ),  

    # ============================ Rollups ========================== #

    # ROW 1
    html.Div(
        className='rollup-row',
        children=[
            html.Div(
                className='rollup-box-tl',
                children=[
                    html.Div(
                        className='title-box',
                        children=[
                            html.H3(
                                className='rollup-title',
                                children=[f'{report_month} Participants']
                            ),
                        ]
                    ),

                    html.Div(
                        className='circle-box',
                        children=[
                            html.Div(
                                className='circle-1',
                                children=[
                                    html.H1(
                                    className='rollup-number',
                                    children=[df_len]
                                    ),
                                ]
                            )
                        ],
                    ),
                ]
            ),
            html.Div(
                className='rollup-box-tr',
                children=[
                    html.Div(
                        className='title-box',
                        children=[
                            html.H3(
                                className='rollup-title',
                                children=['Avg. Heart Rate']
                            ),
                        ]
                    ),
                    html.Div(
                        className='circle-box',
                        children=[
                            html.Div(
                                className='circle-2',
                                children=[
                                    html.H1(
                                    className='rollup-number',
                                    children=[hr_avg]
                                    ),
                                ]
                            )
                        ],
                    ),
                ]
            ),
        ]
    ),

    html.Div(
        className='rollup-row',
        children=[
            html.Div(
                className='rollup-box-bl',
                children=[
                    html.Div(
                        className='title-box',
                        children=[
                            html.H3(
                                className='rollup-title',
                                children=['Avg. Systolic']
                            ),
                        ]
                    ),

                    html.Div(
                        className='circle-box',
                        children=[
                            html.Div(
                                className='circle-3',
                                children=[
                                    html.H1(
                                    className='rollup-number',
                                    children=[systolic_avg]
                                    ),
                                ]
                            )
                        ],
                    ),
                ]
            ),
            html.Div(
                className='rollup-box-br',
                children=[
                    html.Div(
                        className='title-box',
                        children=[
                            html.H3(
                                className='rollup-title',
                                children=['Avg. Diastolic']
                            ),
                        ]
                    ),
                    html.Div(
                        className='circle-box',
                        children=[
                            html.Div(
                                className='circle-4',
                                children=[
                                    html.H1(
                                    className='rollup-number',
                                    children=[diastolic_avg]
                                    ),
                                ]
                            )
                        ],
                    ),
                ]
            ),
        ]
    ),

# ============================ Visuals ========================== #

html.Div(
    className='graph-container',
    children=[
        
        html.H1(
            className='visuals-text',
            children='Visuals'
        ),
        
        html.Div(
            className='graph-row',
            children=[
                
                html.Div(
                    className='graph-box',
                    children=[
                        dcc.Graph(
                            className='graph',
                            figure=age_bar
                        )
                    ]
                ),
                html.Div(
                    className='graph-box',
                    children=[
                        dcc.Graph(
                            className='graph',
                            figure=age_pie
                        )
                    ]
                ),
            ]
        ),

    html.Div(
        className='graph-container',
        children=[
            
            html.Div(
                className='graph-row',
                children=[
                    html.Div(
                        className='graph-box',
                        children=[
                            dcc.Graph(
                                className='graph',
                                figure=gender_bar
                            )
                        ]
                    ),
                    html.Div(
                        className='graph-box',
                        children=[
                            dcc.Graph(
                                className='graph',
                                figure=gender_pie
                            )
                        ]
                    ),
                ]
            ),
        ]
    ),

    html.Div(
        className='graph-container',
        children=[
            
            html.Div(
                className='graph-row',
                children=[
                    html.Div(
                        className='graph-box',
                        children=[
                            dcc.Graph(
                                className='graph',
                                figure=race_bar
                            )
                        ]
                    ),
                    html.Div(
                        className='graph-box',
                        children=[
                            dcc.Graph(
                                className='graph',
                                figure=race_pie
                            )
                        ]
                    ),
                ]
            ),
        ]
    ),

    html.Div(
        className='graph-container',
        children=[
            
            html.Div(
                className='graph-row',
                children=[
                    html.Div(
                        className='graph-box',
                        children=[
                            dcc.Graph(
                                className='graph',
                                figure=interest_bar
                            )
                        ]
                    ),
                    html.Div(
                        className='graph-box',
                        children=[
                            dcc.Graph(
                                className='graph',
                                figure=interest_pie
                            )
                        ]
                    ),
                ]
            ),
        ]
    ),

    html.Div(
        className='graph-container',
        children=[
            
            html.Div(
                className='graph-row',
                children=[
                    html.Div(
                        className='graph-box',
                        children=[
                            dcc.Graph(
                                className='graph',
                                figure=outcome_bar
                            )
                        ]
                    ),
                    html.Div(
                        className='graph-box',
                        children=[
                            dcc.Graph(
                                className='graph',
                                figure=outcome_pie
                            )
                        ]
                    ),
                ]
            ),
        ]
    ),

    html.Div(
        className='graph-container',
        children=[
            
            html.Div(
                className='graph-row',
                children=[
                    html.Div(
                        className='graph-box',
                        children=[
                            dcc.Graph(
                                className='graph',
                                figure=continue_bar
                            )
                        ]
                    ),
                    html.Div(
                        className='graph-box',
                        children=[
                            dcc.Graph(
                                className='graph',
                                figure=continue_pie
                            )
                        ]
                    ),
                ]
            ),
        ]
    ),

    html.Div(
        className='graph-container',
        children=[
            
            html.Div(
                className='graph-row',
                children=[
                    html.Div(
                        className='graph-box',
                        children=[
                            dcc.Graph(
                                className='graph',
                                figure=enroll_bar
                            )
                        ]
                    ),
                    html.Div(
                        className='graph-box',
                        children=[
                            dcc.Graph(
                                className='graph',
                                figure=enroll_pie
                            )
                        ]
                    ),
                ]
            ),
        ]
    ),

# ============================ Folium ========================== #

    html.Div(
        className='graph-row',
        children=[
            html.Div(
                className='wide-box',
                children=[
                    dcc.Graph(
                        className='zip-graph',
                        figure=zip_fig
                    )
                ]
            ),
        ]
    ),
    html.Div(
        className='folium-row',
        children=[
            html.Div(
                className='folium-box',
                children=[
                    html.H1(
                        'Visitors by Zip Code Map', 
                        className='zip'
                    ),
                    html.Iframe(
                        className='folium',
                        id='folium-map',
                        # srcDoc=map_html
                    )
                ]
            ),
        ]
    ),
]),

    # ============================ Data Table ========================== #

    html.Div(
        className='data-box',
        children=[
            html.H1(
                className='data-title',
                children='Movement is Medicine Table'
            ),
            dash_table.DataTable(
                id='applications-table',
                data=data,
                columns=columns,
                page_size=10,
                sort_action='native',
                filter_action='native',
                row_selectable='multi',
                style_table={
                    'overflowX': 'auto',
                    # 'border': '3px solid #000',
                    # 'borderRadius': '0px'
                },
                style_cell={
                    'textAlign': 'left',
                    'minWidth': '100px', 
                    'whiteSpace': 'normal'
                },
                style_header={
                    'textAlign': 'center', 
                    'fontWeight': 'bold',
                    'backgroundColor': '#34A853', 
                    'color': 'white'
                },
                style_data={
                    'whiteSpace': 'normal',
                    'height': 'auto',
                },
                style_cell_conditional=[
                    # make the index column narrow and centered
                    {'if': {'column_id': '#'},
                    'width': '20px', 'minWidth': '60px', 'maxWidth': '60px', 'textAlign': 'center'},
                    {'if': {'column_id': 'Timestamp'},
                    'width': '50px', 'minWidth': '100px', 'maxWidth': '200px', 'textAlign': 'center'},
                    {'if': {'column_id': 'Date of Activity'},
                    'width': '50px', 'minWidth': '100px', 'maxWidth': '200px', 'textAlign': 'center'},
                    {'if': {'column_id': 'Interest'},
                    'width': '200px', 'minWidth': '200px', 'maxWidth': '200px', 'textAlign': 'center'},
                ]
            ),
        ]
    ),
])

print(f"Serving Flask app '{current_file}'! 🚀")

if __name__ == '__main__':
    app.run(debug=
                   True)
                #    False)
                
# ---------------------------- Updated Database ---------------------------- #

# updated_path = f'data/Navigation(FH)_{report_month}_{report_year}.xlsx'
# data_path = os.path.join(script_dir, updated_path)
# sheet_name=f'{report_month} {report_year}'

# with pd.ExcelWriter(data_path, engine='xlsxwriter') as writer:
#     df.to_excel(
#             writer, 
#             sheet_name=sheet_name, 
#             startrow=1, 
#             index=False
#         )

#     # Access the workbook and each worksheet
#     workbook = writer.book
#     sheet1 = writer.sheets[sheet_name]
    
#     # Define the header format
#     header_format = workbook.add_format({
#         'bold': True, 
#         'font_size': 16, 
#         'align': 'center', 
#         'valign': 'vcenter',
#         'border': 1, 
#         'font_color': 'black', 
#         'bg_color': '#B7B7B7',
#     })
    
#     # Set column A (Name) to be left-aligned, and B-E to be right-aligned
#     left_align_format = workbook.add_format({
#         'align': 'left',  # Left-align for column A
#         'valign': 'vcenter',  # Vertically center
#         'border': 0  # No border for individual cells
#     })

#     right_align_format = workbook.add_format({
#         'align': 'right',  # Right-align for columns B-E
#         'valign': 'vcenter',  # Vertically center
#         'border': 0  # No border for individual cells
#     })
    
#     # Create border around the entire table
#     border_format = workbook.add_format({
#         'border': 1,  # Add border to all sides
#         'border_color': 'black',  # Set border color to black
#         'align': 'center',  # Center-align text
#         'valign': 'vcenter',  # Vertically center text
#         'font_size': 12,  # Set font size
#         'font_color': 'black',  # Set font color to black
#         'bg_color': '#FFFFFF'  # Set background color to white
#     })

#     # Merge and format the first row (A1:E1) for each sheet
#     sheet1.merge_range('A1:AB1', f'Client Navigation Report {report_month} {report_year}', header_format)

#     # Set column alignment and width
#     # sheet1.set_column('A:A', 20, left_align_format)  

#     print(f"Navigation Excel file saved to {data_path}")

# -------------------------------------------- KILL PORT ---------------------------------------------------

# netstat -ano | findstr :8050
# taskkill /PID 24772 /F
# npx kill-port 8050


# ---------------------------------------------- Host Application -------------------------------------------

# 1. pip freeze > requirements.txt
# 2. add this to procfile: 'web: gunicorn impact_11_2024:server'
# 3. heroku login
# 4. heroku create
# 5. git push heroku main

# Create venv 
# virtualenv venv 
# source venv/bin/activate # uses the virtualenv

# Update PIP Setup Tools:
# pip install --upgrade pip setuptools

# Install all dependencies in the requirements file:
# pip install -r requirements.txt

# Check dependency tree:
# pipdeptree
# pip show package-name

# Remove
# pypiwin32
# pywin32
# jupytercore

# ----------------------------------------------------

# Name must start with a letter, end with a letter or digit and can only contain lowercase letters, digits, and dashes.

# Heroku Setup:
# heroku login
# heroku create nav-jul-2025
# heroku git:remote -a nav-jul-2025
# git remote set-url heroku git@heroku.com:nav-jan-2025.git
# git push heroku main

# Clear Heroku Cache:
# heroku plugins:install heroku-repo
# heroku repo:purge_cache -a nav-nov-2024

# Set buildpack for heroku
# heroku buildpacks:set heroku/python

# Heatmap Colorscale colors -----------------------------------------------------------------------------

#   ['aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance',
            #  'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg',
            #  'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl',
            #  'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric',
            #  'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys',
            #  'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet',
            #  'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges',
            #  'orrd', 'oryel', 'oxy', 'peach', 'phase', 'picnic', 'pinkyl',
            #  'piyg', 'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn',
            #  'puor', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu',
            #  'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar',
            #  'spectral', 'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn',
            #  'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid',
            #  'turbo', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr',
            #  'ylorrd'].

# rm -rf ~$bmhc_data_2024_cleaned.xlsx
# rm -rf ~$bmhc_data_2024.xlsx
# rm -rf ~$bmhc_q4_2024_cleaned2.xlsx