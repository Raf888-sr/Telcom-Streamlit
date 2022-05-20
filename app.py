# Import necessary librariries
import streamlit as st
import pandas as pd
import numpy as np
import hydralit_components as hc
import requests
import inspect
from streamlit_lottie import st_lottie
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from numerize import numerize
from itertools import chain
import plotly.graph_objects as go
import plotly.express as px
import joblib
import sklearn
import statsmodels.api as sm

# Set Page Icon,Title, and Layout
st.set_page_config(layout="wide", page_icon = "https://cdn-icons-png.flaticon.com/512/2824/2824717.png", page_title = "Telecom Customer Churn")

# Load css style file from local disk
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)
# Load css style from url
def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">',unsafe_allow_html = True)

# Display lottie animations
def load_lottieurl(url):

    # get the url
    r = requests.get(url)
    # if error 200 raised return Nothing
    if r.status_code !=200:
        return None
    return r.json()


# Upload File Function
def upload():

    # Dispaly Upload File Widget
    uploaded_file = st.file_uploader(label="Upload your CSV File",type=["csv"])

    # Save the file in internal memory of streamlit

    if 'file' not in st.session_state:
        st.session_state['file'] = None


    st.session_state['file'] = uploaded_file

    # Save the table in internal memory of Streamlit
    if 'table' not in st.session_state:
            st.session_state['table'] = None

    if uploaded_file is not None:
            st.session_state['table'] = pd.read_csv(uploaded_file)
            return st.session_state['table']
    else:
        st.session_state['table'] = pd.read_csv("telcom.csv")
        return st.session_state['table']


# Navigation Bar Design
menu_data = [
{'label':"Home", 'icon': "bi bi-house"},
{'label':"EDA", 'icon': "bi bi-clipboard-data"},
{'label':'Overview', 'icon' : "bi bi-graph-up-arrow"},
{'label':'Profiling', 'icon' : "bi bi-file-person"},
{'label':'Application', 'icon' : "fa fa-brain"}]


# Set the Navigation Bar
menu_id = hc.nav_bar(menu_definition = menu_data,
                    sticky_mode = 'sticky',
                    sticky_nav = False,
                    hide_streamlit_markers = False,
                    override_theme = {'txc_inactive': 'white',
                                        'menu_background' : '#0178e4',
                                        'txc_active':'#0178e4',
                                        'option_active':'white'})


# Load css library
remote_css("https://unpkg.com/tachyons@4.12.0/css/tachyons.min.css")
# Load css style
local_css('style.css')


# Extract Lottie Animations
lottie_churn = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_7z8wtyb0.json")
lottie_upload = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_z7DhMX.json")
lottie_eda = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_ic37y4kv.json")
lottie_ml = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_q5qeoo3q.json")



# Home Page
if menu_id == "Home":

    # Display Introduction
    st.markdown("""
    <article>
  <header class="bg-gold sans-seSans Serif">
    <div class="mw9 center pa4 pt5-ns ph7-l">
      <time class="f6 mb2 dib ttu tracked"><small>18 May, 2022</small></time>
      <h3 class="f2 f1-m f-headline-l measure-narrow lh-title mv0">
        <span class="bg-black-90 lh-copy white pa1 tracked-tight">
          Machine Learning Project: Telco Customer Churn
        </span>
      </h3>
      <h4 class="f3 fw1 Sans Serif i">Analyzing IBM telecommunications data (Kaggle dataset)</h4>
      <h5 class="f6 ttu tracked black-80">By Rafic Srouji</h5>
      </div>
      </header>
      <div class="pa4 ph7-l mw9-l center">
      <p class="f5 f4-ns lh-copy measure mb4" style="text-align: justify;">
        Churn (customer migration) is affecting a growing number of companies. Preventing customer departures becomes crucial, especially when we have a relatively mature market and few new customers for a product or service, and it is easy for the customer to change supplier.
        Tackling customer migration is beneficial because the cost of acquiring new customers is usually higher than keeping existing customers. It's also worth emphasizing that long-term client collaboration has advantages in terms of growing income and company promotion.
      </p>
      <p class="f5 f4-ns lh-copy measure mb4">
        Apple Cofounder Steve Jobs once said:
      </p>
      <p class="f6 f5-ns lh-copy measure i pl4 bl bw1 b--gold mb4">
      Get closer than ever to your customers. So close that you tell them what they need
      well before they realize it themselves.
      </p>
      </div>
      </article>""",unsafe_allow_html=True)

    # Display customer churn animation
    st_lottie(lottie_churn, key = "churn", height = 300, width = 1000)


    # Show brief introduction about importnace of churn analysis
    st.markdown("""
    <article>
        <div class="pa4 ph7-l Sans Serif mw9-l center">
            <p class="f5 f4-ns lh-copy measure mb4" style="text-align: justify;">
            In this application, we will explore the churn rate in-depth and pinpoint the pitfalls behind the departure
            of customers and identify at risk customers using a machine learning model that act as an early warning system.
            </p>
            <p class="f5 f4-ns lh-copy measure mb4" style="text-align: justify;">
            Please Upload the File to Process Your Data
            </p>
            </div>
            </article>""",unsafe_allow_html=True)

    # 3 columns of same size layout
    col1,col2,col3 = st.columns([1,1,1])

    # First column - Show Upload Animation
    with col1:
        st_lottie(lottie_upload, key = "upload",height = 200, width = 600)
    # Second Column - Display Upload Widget
    with col2:
        upload()


# Retreive detailed report of the Exploratory Data Analysis
def profile(df):
    pr = ProfileReport(df, explorative=True)
    tbl = st_profile_report(pr)
    return  tbl

# Save the data in memory
df = st.session_state['table']
# Some Cleaning
df['PaymentMethod'] = df['PaymentMethod'].str.replace(' (automatic)', '', regex=False)


# EDA page

if menu_id == "EDA":

    # Drop unnecessary columns
    df1 = df.drop(['Latitude','Longitude','Churn_Index','First Name','Last Name','Full Name'],axis=1)
    # transform TotalCharges column to numeric
    df1['TotalCharges'] = pd.to_numeric(df1['TotalCharges'], errors='coerce')
    # remove (automatic) from payment method names
    df1['PaymentMethod'] = df1['PaymentMethod'].str.replace(' (automatic)', '', regex=False)


    # 2 Column Layouts of Same Size
    col4,col5 = st.columns([1,1])

    # First Column - Shows Description of EDA
    with col4:
        st.markdown("""
        <h3 class="f2 f1-m f-headline-l measure-narrow lh-title mv0">
         Know Your Data
         </h3>
         <p class="f5 f4-ns lh-copy measure mb4" style="text-align: justify;font-family: Sans Serif">
          Before implementing your machine learning model, it is important at the initial stage to explore your data.
          It is a good practice to understand the data first and try gather as many insights from it. EDA is all about
          making sense of data in hand,before getting them dirty with it.
         </p>
            """,unsafe_allow_html = True)
        global eda_button

        # Customize Button
        button = st.markdown("""
        <style>
        div.stButton > button{
        background-color: #0178e4;
        color:#ffffff;
        box-shadow: #094c66 4px 4px 0px;
        border-radius:8px 8px 8px 8px;
        transition : transform 200ms,
        box-shadow 200ms;
        }

         div.stButton > button:focus{
        background-color: #0178e4;
        color:#ffffff;
        box-shadow: #094c66 4px 4px 0px;
        border-radius:8px 8px 8px 8px;
        transition : transform 200ms,
        box-shadow 200ms;
        }


        div.stButton > button:active {

                transform : translateY(4px) translateX(4px);
                box-shadow : #0178e4 0px 0px 0px;

            }
        </style>""", unsafe_allow_html=True)
        # Display Button
        eda_button= st.button("Explore Your Data")


    # Second Column - Display EDA Animation
    with col5:
        st_lottie(lottie_eda, key = "eda",height = 300, width = 800)

    # User Clicks on Button, then profile report of the uplaoded or existing dataframe will be displayed
    if eda_button:
        profile(df1)


# Dashboard Page
if menu_id == "Overview":

    # Get Percentage of Gender
    df_gender = df.groupby(['gender'],as_index=False).size()

    # Change Total Charges to numeric
    df['TotalCharges']= pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Get unique list of Citites
    city = [['All'],df['City'].unique().tolist()]
    City = list(chain(*city))

    # Create 4 column layouts of same size for the filters
    filters = st.columns(4)

    # First Filter - Churn
    with filters[0]:
        churn = st.selectbox('Churn',options = ['All','Yes','No'])

    # Second Filter - Senior Citizen
    with filters[1]:
        senior = st.selectbox('Senior Citizen', options = ['All','Yes','No'])

    # Third Filter - Dependents
    with filters[2]:
        dependents = st.selectbox('Dependents', options = ['All','Yes','No'])

    # Foruth Filter - Partner
    with filters[3]:
        partner = st.selectbox('Partner', options = ['All','Yes','No'])


    # Since we have 4 filters and to avoid error,
    # we have to make a total of 16 combination of possibilites

    # Condition 1
    if (churn != 'All') & (senior != 'All') & (dependents != 'All') & (partner != 'All'):

        df_filtered = df[(df['Churn'] == churn) & (df['SeniorCitizen'] == senior) & (df['Dependents'] == dependents) & (df['Partner'] == partner)]

    # Condition 2
    elif (churn == 'All') & (senior != 'All') & (dependents != 'All') & (partner != 'All'):

        df_filtered = df[(df['SeniorCitizen'] == senior) & (df['Dependents'] == dependents) & (df['Partner'] == partner)]

    # Condition 3
    elif (churn != 'All') & (senior == 'All') & (dependents != 'All') & (partner != 'All'):

        df_filtered = df[(df['Churn'] == churn) & (df['Dependents'] == dependents) & (df['Partner'] == partner)]

    # Condition 4
    elif (churn != 'All') & (senior != 'All') & (dependents == 'All') & (partner != 'All'):

        df_filtered = df[(df['Churn'] == churn) & (df['SeniorCitizen'] == senior) & (df['Partner'] == partner)]

    # Condition 5
    elif (churn != 'All') & (senior != 'All') & (dependents != 'All') & (partner == 'All'):

        df_filtered = df[(df['Churn'] == churn) & (df['SeniorCitizen'] == senior) & (df['Dependents'] == dependents)]


    # Condition 6
    elif (churn == 'All') & (senior == 'All') & (dependents == 'All') & (partner != 'All'):

        df_filtered = df[df['Partner'] == partner]

    # Condition 7
    elif (churn == 'All') & (senior == 'All') & (dependents != 'All') & (partner == 'All'):

        df_filtered = df[df['Dependents'] == partner]

    # Condition 8
    elif (churn == 'All') & (senior != 'All') & (dependents == 'All') & (partner == 'All'):

        df_filtered = df[df['SeniorCitizen'] == senior]

    # Condition 9
    elif (churn != 'All') & (senior == 'All') & (dependents == 'All') & (partner == 'All'):

        df_filtered = df[df['Churn'] == churn]

    # Condition 10
    elif (churn == 'All') & (senior == 'All') & (dependents != 'All') & (partner != 'All'):

        df_filtered = df[(df['Dependents'] == dependents) & (df['Partner'] == partner)]

    # Condition 11
    elif (churn == 'All') & (senior != 'All') & (dependents == 'All') & (partner != 'All'):

        df_filtered = df[(df['SeniorCitizen'] == senior) & (df['Partner'] == partner)]

    # Condition 12
    elif (churn != 'All') & (senior == 'All') & (dependents == 'All') & (partner != 'All'):

        df_filtered = df[(df['Churn'] == churn) & (df['Partner'] == partner)]

    # Condition 13
    elif (churn == 'All') & (senior != 'All') & (dependents != 'All') & (partner == 'All'):

        df_filtered = df[(df['SeniorCitizen'] == senior) & (df['Dependents'] == dependents)]

    # Condition 14
    elif (churn != 'All') & (senior == 'All') & (dependents != 'All') & (partner == 'All'):

        df_filtered = df[(df['Churn'] == churn) & (df['Dependents'] == dependents)]

    # Condition 15
    elif (churn != 'All') & (senior != 'All') & (dependents == 'All') & (city == 'All'):

        df_filtered = df[(df['Churn'] == churn) & (df['SeniorCitizen'] == senior)]

    # Condition 16
    else:
        df_filtered = df



    #can apply customisation to almost all the properties of the card, including the progress bar
    theme_customers= {'bgcolor': '#f6f6f6','title_color': '#2A4657','content_color': '#0178e4','progress_color': '#0178e4','icon_color': '#0178e4', 'icon': 'fa fa-user-friends'}
    theme_churn = {'bgcolor': '#f6f6f6','title_color': '#2A4657','content_color': '#0178e4','progress_color': '#0178e4','icon_color': '#0178e4', 'icon': 'fa fa-running'}
    theme_charges = {'bgcolor': '#f6f6f6','title_color': '#2A4657','content_color': '#0178e4','progress_color': '#0178e4','icon_color': '#0178e4', 'icon': 'fa fa-hand-holding-usd'}
    theme_tenure = {'bgcolor': '#f6f6f6','title_color': '#2A4657','content_color': '#0178e4','progress_color': '#0178e4','icon_color': '#0178e4', 'icon': 'fa fa-business-time'}

    # Set 4 info cards
    info = st.columns(4)

    # First KPI - Number of Custoemrs
    with info[0]:
        hc.info_card(title='# of Customers', content=df_filtered.shape[0], bar_value = (df_filtered.shape[0]/df.shape[0])*100,sentiment='good', theme_override = theme_customers)
    # Second KPI - Number of Churened Customers
    with info[1]:
        hc.info_card(title='Churns', content=df_filtered[df_filtered['Churn']=='Yes'].shape[0], bar_value = (df_filtered[df_filtered['Churn']=='Yes'].shape[0]/df[df['Churn']=='Yes'].shape[0])*100,sentiment='good', theme_override = theme_churn)

    # Third KPI - Total Charges
    with info[2]:
        hc.info_card(title='Total Charges', content=numerize.numerize(df_filtered['TotalCharges'].sum(), 2)+'$', bar_value = (df_filtered['TotalCharges'].sum()/df['TotalCharges'].sum())*100,sentiment='good', theme_override = theme_charges)
    # Fourth KPI - Average Tenure
    with info[3]:
        hc.info_card(title='Average Tenure', content=str(np.round(df_filtered['tenure'].mean(),2)) + ' Months', bar_value = (np.round(df_filtered['tenure'].mean(),2)/df['tenure'].max())*100,sentiment='good', theme_override = theme_tenure)

    # Set first 3 sets of viusals
    viz1 = st.columns(3)

    # Donut Chart - Gender Percentage
    with viz1[0]:
            # Group By Gender and get the count
            df_gender = df_filtered.groupby(['gender'],as_index=False).size()

            # Calculate the Percentage Differnece between  genders
            gender_range = (abs(df_gender['size'][0]-df_gender['size'][1])/df_gender['size'].sum()) * 100

            # In case the Differnece was greater than 10%, then the customers are not balanced in terms of gender
            if gender_range <= 10:

                # Plot Donut Chart
                fig = go.Figure(data = [go.Pie(labels = df_gender['gender'], values = df_gender['size'], hole = 0.6)])
                # Assign Colors to Male and Female
                fig.update_traces(marker = dict(colors=['#0178e4','#00284C']))

                # Add image to the plot
                fig.add_layout_image(
                            dict(
                                source="https://cdn-icons-png.flaticon.com/512/28/28591.png",
                                  xref="paper", yref="paper",
                                  x=0.5, y=0.5,
                                  sizex=0.3, sizey=0.3,
                                  xanchor="center", yanchor="middle"
                                                                )
                                        )
                # Set Size and Title of the Plot
                fig.update_layout(width = 400,
                                height = 400,
                                autosize=False,
                                title = f"Gender Percentage <br><sup>Our customer base is almost balanced in terms of gender</sup>",
                                 title_font_family="Sans Serif",
                                  title_font_size = 18  )
                # Show Figure
                st.write(fig)

            # Incase the gender Percentage was less than or equal to 10%, then update the title

            else:
                # Plot Donut Chart
                fig = go.Figure(data = [go.Pie(labels = df_gender['gender'], values = df_gender['size'], hole = 0.6)])
                # Assign Colors to Male and Female
                fig.update_traces(marker = dict(colors=['#0178e4','#00284C']))
                # Add image to the plot
                fig.add_layout_image(
                            dict(
                                source="https://cdn-icons-png.flaticon.com/512/28/28591.png",
                                  xref="paper", yref="paper",
                                  x=0.5, y=0.5,
                                  sizex=0.3, sizey=0.3,
                                  xanchor="center", yanchor="middle"))
                # Set Size and Title of the Plot
                fig.update_layout(width = 400,
                                height = 400,
                                autosize=False,
                                title = f"Gender Percentage <br><sup>Our customer base is not balanced in terms of gender</sup>",
                                 title_font_family="Sans Serif",
                                  title_font_size = 18  )
                # Show Plott
                st.write(fig)





    # Second Plot - Display the Distribution of Tenure with bin size = 12
    with viz1[1]:
        # Histogram Chart
        fig = go.Figure(data=[go.Histogram(x=df_filtered['tenure'],
                                           xbins = dict(start = 0, end = 12 * round(df_filtered['tenure'].max()/12),size=12),
                                           marker_color = "#0178e4")])
        # Set theme and dimensions
        fig.update_layout(template = "simple_white",
                         width = 500,
                         height = 400,
                         title = "Tenure Distribution (Months)",
                          title_font_family="Sans Serif",
                           title_font_size = 18  ))
        # Show Plot
        st.write(fig)

    # Third Plot - Display RelationShip between Tenure and Total Charges
    with viz1[2]:
        # Scatter Plot
        fig = px.scatter(df_filtered, x="tenure",
                        y="TotalCharges",
                        color = 'Churn',
                        trendline = "ols",
                        color_discrete_map={
                            'Yes' : '#0178e4',
                            'No': '#01427e'})
        # Get the parameters of a fitted linear regression model
        results = px.get_trendline_results(fig)
        # Get R-squared
        r2 = results.px_fit_results.iloc[0].rsquared

        # In case r2 > 0.5, then we assume the relationship is linear between Tenure and TotalCharges
        if r2> 0.5:
            fig.update_layout(template = "simple_white",
                            title = f"Total Charges vs Tenure<br><sup>Total Charges seem to have a linear relationship with Tenure</sup>",
                            title_font_family="Sans Serif",
                            title_font_size = 18,
                            width = 500,
                            height = 400)

        # In case r2<=0.5, then the relationship between Tenure and Total Charges is not linear
        else:
            fig.update_layout(template = "simple_white",
                            title = f"Total Charges vs Tenure<br><sup>Total Charges does not seem to have a linear relationship with Tenure</sup>",
                            title_font_family="Sans Serif",
                            title_font_size = 18,
                            width = 500,
                            height = 400)

        st.write(fig)

    # Set 3 additonal columns
    viz2 = st.columns(3)

    # Foruth Plot - Scatter Map
    with viz2[0]:
        # Scatter Map
        fig = go.Figure(data=go.Scattergeo(
            lon = df_filtered['Longitude'],
            lat = df_filtered['Latitude'],
            text = df_filtered['TotalCharges'],
            mode = 'markers',
            marker = dict(
            colorscale = 'Blues',
             reversescale = False,
            autocolorscale = False,
            color = df_filtered['TotalCharges'],
            cmin = 0,
            cmax = df_filtered['TotalCharges'].max(),
            colorbar_title="Total Charges ($)"
            )))

        # Set the dimensions and style of map
        fig.update_layout(
        title = f'Total Charges in California',
        title_font_family="Sans Serif",
        title_font_size = 18,
        width = 400,
        height = 400,
        geo = dict(
            scope='usa',
            projection_type='albers usa',
            showland = True,
            landcolor = "rgb(244,238,224)",
            subunitcolor = "rgb(217, 217, 217)",
            countrycolor = "rgb(217, 217, 217)",
            countrywidth = 0.5,
            subunitwidth = 0.5
        ),
    )
        # Show Plot
        st.write(fig)

    # Fifth Plot - Donut Chart of Contract Terms
    with viz2[1]:
        # Get the count of each contract type
        df_contract = df_filtered.groupby(['Contract'],as_index=False).size()
        # Get name of most susbcribed contract
        contract = df_contract[df_contract['size']==df_contract['size'].max()].reset_index()['Contract'][0]

        # Set colors to classes
        colors = {'Month to Month': '#0178e4',
                  'One Year' : '#00284C',
                  'Two Year': '#00080F'}

        # Convert to Series
        s = pd.Series(colors)
        # Set Donut Chart
        fig = go.Figure(data = [go.Pie(labels = df_contract['Contract'], values = df_contract['size'], hole = 0.6)])
        # Update Colors
        fig.update_traces(marker = dict(colors = s))
        # Add Image to the Plot
        fig.add_layout_image(
                    dict(
                        source="https://cdn-icons-png.flaticon.com/512/684/684872.png",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5,
                          sizex=0.3, sizey=0.3,
                          xanchor="center", yanchor="middle"))

        # Set dimensions and style of plot
        fig.update_layout(width = 400,
                        height = 400,
                        autosize=False,

                        title = f"Contract Terms<br><sup>Most customers have subscribed with {contract} plan</sup>",
                         title_font_family="Sans Serif",
                          title_font_size = 18)


        # Show Plot
        st.write(fig)

    # Sixth Plot - Bar Chart displaying number of occurences of each Internet Service
    with viz2[2]:
        # Get count of each category of InternetService
        df_internet = df_filtered.groupby(['InternetService'],as_index = False).size().sort_values(by = 'size', ascending = False)
        # Get the category with highest occurences
        service = df_internet[df_internet['size']==df_internet['size'].max()].reset_index()['InternetService'][0]
        # If Most customers does not have access to any Internet Service, adjust tthe title
        if service != 'No':
            # Bat Chart
            fig = px.bar( df_internet,x = 'InternetService', y = 'size')
            # Set dimensions and size of plot
            fig.update_layout(template = "simple_white",
                              width = 500,
                              height = 400,
                              autosize = False,
                              title =f"Internet Services <br><sup>Most customers have access to {service} services</sup>",
                               title_font_family="Sans Serif",
                                title_font_size = 18)
            # Show Plot
            st.write(fig)

        # Same Procedure as before
        else:

                fig = px.bar( df_internet,x = 'InternetService', y = 'size')
                fig.update_layout(template = "simple_white",
                                  width = 500,
                                  height = 400,
                                  autosize = False,
                                  title =f"Internet Services <br><sup>Most customers have {service} acess to any of Internet services</sup>",
                               title_font_family="Sans Serif",
                                    title_font_size = 18)
                st.write(fig)

#
    cols = st.columns(6)

    with cols[0]:
        st.markdown(f"""
        <html
        <head>
            <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
            <title>Card Hover Effect</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link rel="stylesheet" type="text/css" media="screen" href="style.css" />
            <link href="https://stackpath.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css" rel="stylesheet" integrity="sha384-wvfXpqpZZVQGK6TAh5PVlGOfQNHSoD2xbE+QkPxCAFlNEevoEH3Sl0sibVcOQVnN"
            crossorigin="anonymous">
        </head>
        <body>
        <div class="container">
                <div class="card">
                <div class="slide slide1">
                    <div class="content">
                        <div class="icon">
                        <i class="material-icons" aria-hidden="true">lock</i>
                        </div>
                    </div>
                </div>
            <div class="slide slide2">
                <div class="content">
                <h3>
                Online Security
                </h3>
                <p>{round((df_filtered[df_filtered['OnlineSecurity'] == "Yes"]['OnlineSecurity'].count()/df.shape[0])*100,2)}%</p>
            </div>
            </div>
            </div>
            </div>
            </body>
            </html>""", unsafe_allow_html = True)


        with cols[1]:
            st.markdown(f"""
            <html
            <head>
                <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
                <title>Card Hover Effect</title>
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <link rel="stylesheet" type="text/css" media="screen" href="style.css" />
                <link href="https://stackpath.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css" rel="stylesheet" integrity="sha384-wvfXpqpZZVQGK6TAh5PVlGOfQNHSoD2xbE+QkPxCAFlNEevoEH3Sl0sibVcOQVnN"
                crossorigin="anonymous">
            </head>
            <body>
            <div class="container">
                    <div class="card">
                    <div class="slide slide1">
                        <div class="content">
                            <div class="icon">
                            <i class="material-icons" aria-hidden="true">backup</i>
                            </div>
                        </div>
                    </div>
                <div class="slide slide2">
                    <div class="content">
                    <h3>
                    Online Backup
                    </h3>
                    <p>{round((df_filtered[df_filtered['OnlineBackup'] == "Yes"]['OnlineBackup'].count()/df.shape[0])*100,2)}%</p>
                </div>
                </div>
                </div>
                </div>
                </body>
                </html>""", unsafe_allow_html = True)

            with cols[2]:
                st.markdown(f"""
                <html
                <head>
                    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
                    <title>Card Hover Effect</title>
                    <meta name="viewport" content="width=device-width, initial-scale=1">
                    <link rel="stylesheet" type="text/css" media="screen" href="style.css" />
                    <link href="https://stackpath.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css" rel="stylesheet" integrity="sha384-wvfXpqpZZVQGK6TAh5PVlGOfQNHSoD2xbE+QkPxCAFlNEevoEH3Sl0sibVcOQVnN"
                    crossorigin="anonymous">
                </head>
                <body>
                <div class="container">
                        <div class="card">
                        <div class="slide slide1">
                            <div class="content">
                                <div class="icon">
                                <i class="material-icons" aria-hidden="true">smartphone</i>
                                </div>
                            </div>
                        </div>
                    <div class="slide slide2">
                        <div class="content">
                        <h3>
                        Device Protection
                        </h3>
                        <p>{round((df_filtered[df_filtered['DeviceProtection'] == "Yes"]['DeviceProtection'].count()/df.shape[0])*100,2)}%</p>
                    </div>
                    </div>
                    </div>
                    </div>
                    </body>
                    </html>""", unsafe_allow_html = True)


            with cols[3]:
                st.markdown(f"""
                <html
                <head>
                    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
                    <title>Card Hover Effect</title>
                    <meta name="viewport" content="width=device-width, initial-scale=1">
                    <link rel="stylesheet" type="text/css" media="screen" href="style.css" />
                    <link href="https://stackpath.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css" rel="stylesheet" integrity="sha384-wvfXpqpZZVQGK6TAh5PVlGOfQNHSoD2xbE+QkPxCAFlNEevoEH3Sl0sibVcOQVnN"
                    crossorigin="anonymous">
                </head>
                <body>
                <div class="container">
                        <div class="card">
                        <div class="slide slide1">
                            <div class="content">
                                <div class="icon">
                                <i class="material-icons" aria-hidden="true">construction</i>
                                </div>
                            </div>
                        </div>
                    <div class="slide slide2">
                        <div class="content">
                        <h3>
                        Tech Support
                        </h3>
                        <p>{round((df_filtered[df_filtered['TechSupport'] == "Yes"]['TechSupport'].count()/df.shape[0])*100,2)}%</p>
                    </div>
                    </div>
                    </div>
                    </div>
                    </body>
                    </html>""", unsafe_allow_html = True)

                with cols[4]:
                    st.markdown(f"""
                    <html
                    <head>
                        <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
                        <title>Card Hover Effect</title>
                        <meta name="viewport" content="width=device-width, initial-scale=1">
                        <link rel="stylesheet" type="text/css" media="screen" href="style.css" />
                        <link href="https://stackpath.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css" rel="stylesheet" integrity="sha384-wvfXpqpZZVQGK6TAh5PVlGOfQNHSoD2xbE+QkPxCAFlNEevoEH3Sl0sibVcOQVnN"
                        crossorigin="anonymous">
                    </head>
                    <body>
                    <div class="container">
                            <div class="card">
                            <div class="slide slide1">
                                <div class="content">
                                    <div class="icon">
                                    <i class="material-icons" aria-hidden="true">tv</i>
                                    </div>
                                </div>
                            </div>
                        <div class="slide slide2">
                            <div class="content">
                            <h3>
                            Streaming TV
                            </h3>
                            <p>{round((df_filtered[df_filtered['StreamingTV'] == "Yes"]['StreamingTV'].count()/df.shape[0])*100,2)}%</p>
                        </div>
                        </div>
                        </div>
                        </div>
                        </body>
                        </html>""", unsafe_allow_html = True)


                with cols[5]:
                    st.markdown(f"""
                    <html
                    <head>
                        <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
                        <title>Card Hover Effect</title>
                        <meta name="viewport" content="width=device-width, initial-scale=1">
                        <link rel="stylesheet" type="text/css" media="screen" href="style.css" />
                        <link href="https://stackpath.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css" rel="stylesheet" integrity="sha384-wvfXpqpZZVQGK6TAh5PVlGOfQNHSoD2xbE+QkPxCAFlNEevoEH3Sl0sibVcOQVnN"
                        crossorigin="anonymous">
                    </head>
                    <body>
                    <div class="container">
                            <div class="card">
                            <div class="slide slide1">
                                <div class="content">
                                    <div class="icon">
                                    <i class="material-icons" aria-hidden="true">movie</i>
                                    </div>
                                </div>
                            </div>
                        <div class="slide slide2">
                            <div class="content">
                            <h3>
                            Streaming Movies
                            </h3>
                            <p>{round((df_filtered[df_filtered['StreamingMovies'] == "Yes"]['StreamingMovies'].count()/df.shape[0])*100,2)}%</p>
                        </div>
                        </div>
                        </div>
                        </div>
                        </body>
                        </html>""", unsafe_allow_html = True)

        # st.markdown("""
        # <style>
        # div.stButton > button{
        # background-color: #0178e4;
        # color:#ffffff;
        # box-shadow: #094c66 4px 4px 0px;
        # border-radius:8px 8px 8px 8px;
        # transition : transform 200ms,
        # box-shadow 200ms;
        # }
        # </style>""", unsafe_allow_html=True)


####


if menu_id == "Profiling":

    col1,col2,col3,col4,col5 = st.columns([1,0.5,2,0.5,1])
    cols = st.columns(5)
    # col6,col7,col8,col9,col10 = st.columns([1,0.5,2,0.5,1])
    #col2 = st.columns(3)



    with col3:

#
        selected_customer = st.selectbox("Select Customer ID", options = df['customerID'].unique())
        selected_id = df[df['customerID'] == selected_customer].reset_index()
        selected_churn = selected_id['Churn_Index'][0]

        if selected_churn > 0.5:
            risk_level = 'High Risk'
            color = '#00284c'
        elif selected_churn > 0.3 and selected_churn <=0.5:
            risk_level = 'Moderate Risk'
            color = '#1e73b7'
        else:
            risk_level = 'Low Risk'
            color = '#0178e4'
        # for i in selected_id['gender']:
        if selected_id['gender'][0] == 'Male':

              st.write(f"""
            <div>
                <div style="display:inline-block;horizontal-align:middle;padding-top:50px";">
                <img src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png" width=250/>
                <div style="vertical-align:center;font-size:25px;text-align:center;padding-top:30px;margin-left:1em";>
                 {selected_id['Full Name'][0]}
                 </div>
                <div style="vertical-align:center;color:{color};font-size:25px;text-align:center;padding-top:30px;margin-left:1em";>
                  {risk_level}
                </div>""",unsafe_allow_html = True)
#
        else:
             st.write(f"""
           <div>
               <div style="display:inline-block;horizontal-align:center;padding-top:50px";>
               <img src="https://cdn-icons-png.flaticon.com/512/3135/3135789.png" width=250/>
               <div style="vertical-align:center;color:{color};font-size:25px;text-align:center;padding-top:30px;margin-left:1em";>
                {selected_id['Full Name'][0]}
                </div>
                <div style="vertical-align:center;color:{color};font-size:25px;text-align:center;padding-top:30px;margin-left:1em";>
                    {risk_level}
                </div>""",unsafe_allow_html = True)


    with cols[0]:
                # hc.info_card(title='City', content=selected_id['City'][0],sentiment='good', theme_override = theme_location)
            st.markdown(f"""
            <html
            <head>
                <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
                <title>Card Hover Effect</title>
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <link rel="stylesheet" type="text/css" media="screen" href="style.css" />
                <link href="https://stackpath.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css" rel="stylesheet" integrity="sha384-wvfXpqpZZVQGK6TAh5PVlGOfQNHSoD2xbE+QkPxCAFlNEevoEH3Sl0sibVcOQVnN"
                crossorigin="anonymous">
            </head>
            <body>
            <div class="container">
                    <div class="card">
                    <div class="slide slide1">
                        <div class="content">
                            <div class="icon">
                            <i class="material-icons" aria-hidden="true">pin_drop</i>
                            </div>
                        </div>
                    </div>
                <div class="slide slide2">
                    <div class="content">
                    <h3>
                    {selected_id['City'][0]}
                    </h3>
                    <p>City</p>
                </div>
                </div>
                </div>
                </div>
                </body>
                </html>""", unsafe_allow_html = True)

    with cols[1]:
        if selected_id['tenure'][0] != 1:
            month = 'Months'
        else:
            month = 'Month'
        st.markdown(f"""
            <html
            <head>
                <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
                <title>Card Hover Effect</title>
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <link rel="stylesheet" type="text/css" media="screen" href="style.css" />
                <link href="https://stackpath.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css" rel="stylesheet" integrity="sha384-wvfXpqpZZVQGK6TAh5PVlGOfQNHSoD2xbE+QkPxCAFlNEevoEH3Sl0sibVcOQVnN"
                crossorigin="anonymous">
            </head>
            <body>
            <div class="container">
                    <div class="card">
                    <div class="slide slide1">
                        <div class="content">
                            <div class="icon">
                            <i class="material-icons" aria-hidden="true">loyalty</i>
                            </div>
                        </div>
                    </div>
                <div class="slide slide2">
                    <div class="content">
                    <h3>
                    {selected_id['tenure'][0]} {month}
                    </h3>
                    <p>Tenure</p>
                </div>
                </div>
                </div>
                </div>
                </body>
                </html>""", unsafe_allow_html = True)




    with cols[2]:
        st.markdown(f"""
            <html
            <head>
                <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
                <title>Card Hover Effect</title>
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <link rel="stylesheet" type="text/css" media="screen" href="style.css" />
                <link href="https://stackpath.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css" rel="stylesheet" integrity="sha384-wvfXpqpZZVQGK6TAh5PVlGOfQNHSoD2xbE+QkPxCAFlNEevoEH3Sl0sibVcOQVnN"
                crossorigin="anonymous">
            </head>
            <body>
            <div class="container">
                    <div class="card">
                    <div class="slide slide1">
                        <div class="content">
                            <div class="icon">
                            <i class="material-icons" aria-hidden="true">payments</i>
                            </div>
                        </div>
                    </div>
                <div class="slide slide2">
                    <div class="content">
                    <h3>
                    {selected_id['TotalCharges'][0]}$
                    </h3>
                    <p>Total Charges</p>
                </div>
                </div>
                </div>
                </div>
                </body>
                </html>""", unsafe_allow_html = True)


        with cols[3]:
            st.markdown(f"""
                <html
                <head>
                    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
                    <title>Card Hover Effect</title>
                    <meta name="viewport" content="width=device-width, initial-scale=1">
                    <link rel="stylesheet" type="text/css" media="screen" href="style.css" />
                    <link href="https://stackpath.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css" rel="stylesheet" integrity="sha384-wvfXpqpZZVQGK6TAh5PVlGOfQNHSoD2xbE+QkPxCAFlNEevoEH3Sl0sibVcOQVnN"
                    crossorigin="anonymous">
                </head>
                <body>
                <div class="container">
                        <div class="card">
                        <div class="slide slide1">
                            <div class="content">
                                <div class="icon">
                                <i class="material-icons" aria-hidden="true">handshake</i>
                                </div>
                            </div>
                        </div>
                    <div class="slide slide2">
                        <div class="content">
                        <h3>
                        {selected_id['Contract'][0]}
                        </h3>
                        <p>Contract Type</p>
                    </div>
                    </div>
                    </div>
                    </div>
                    </body>
                    </html>""", unsafe_allow_html = True)


        with cols[4]:
            st.markdown(f"""
                <html
                <head>
                    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
                    <title>Card Hover Effect</title>
                    <meta name="viewport" content="width=device-width, initial-scale=1">
                    <link rel="stylesheet" type="text/css" media="screen" href="style.css" />
                    <link href="https://stackpath.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css" rel="stylesheet" integrity="sha384-wvfXpqpZZVQGK6TAh5PVlGOfQNHSoD2xbE+QkPxCAFlNEevoEH3Sl0sibVcOQVnN"
                    crossorigin="anonymous">
                </head>
                <body>
                <div class="container">
                        <div class="card">
                        <div class="slide slide1">
                            <div class="content">
                                <div class="icon">
                                <i class="material-icons" aria-hidden="true">credit_card</i>
                                </div>
                            </div>
                        </div>
                    <div class="slide slide2">
                        <div class="content">
                        <h3>
                        {selected_id['PaymentMethod'][0]}
                        </h3>
                        <p>Payment Method</p>
                    </div>
                    </div>
                    </div>
                    </div>
                    </body>
                    </html>""", unsafe_allow_html = True)



    # col4,col5,col6 = st.columns([1,1,1])
    #
    # with col4:
    #
    #     st.markdown(f"""
    #                 <html
    #                 <head>
    #                     <title>Card Hover Effect</title>
    #                     <meta name="viewport" content="width=device-width, initial-scale=1">
    #                     <link rel="stylesheet" type="text/css" media="screen" href="style.css" />
    #                     <link href="https://stackpath.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css" rel="stylesheet" integrity="sha384-wvfXpqpZZVQGK6TAh5PVlGOfQNHSoD2xbE+QkPxCAFlNEevoEH3Sl0sibVcOQVnN"
    #                     crossorigin="anonymous">
    #                 </head>
    #                 <body>
    #                 <div class="container">
    #                     <div class="card">
    #                         <div class="slide slide1">
    #                             <div class="content">
    #                                 <div class="icon">
    #                                 <i class="fa fa-circle-check" aria-hidden="true"></i>
    #                                 </div>
    #                             </div>
    #                         </div>
    #                     <div class="slide slide2">
    #                         <div class="content">
    #                         <h3>
    #                         {selected_id['Contract'][0]}
    #                         </h3>
    #                     </div>
    #                     </div>
    #                     </div>
    #                     </div>
    #                     </body>
    #                     </html>""", unsafe_allow_html = True)


if menu_id == "Application":

    col = st.columns(2)
    with col[0]:
        st.markdown("""
        <h3 class="f2 f1-m f-headline-l measure-narrow lh-title mv0">
        Know The Risks
         </h3>
         <p class="f5 f4-ns lh-copy measure mb4" style="text-align: justify;font-family: Sans Serif">
         Now, it's time to detect whether any existing or upcoming customer has a risk to churn.
         Fill out the customers' demographic, account, and services information to see the result.
         </p>
            """,unsafe_allow_html = True)

    with col[1]:
        st_lottie(lottie_ml, key = "churn", height = 300, width = 800)

    numerical_features = ['TotalCharges','tenure']
    categorical_features = ['gender','SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',
                            'StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod']

    df_numerical = df[numerical_features]
    df_categorical = df[categorical_features]

    # Features

    X = pd.concat([df_categorical,df_numerical], axis = 1)

    # Target Variable

    y = df['Churn']

    def user_features():

        st.title('Demographic Information')

        cols1 = st.columns(4)
        with cols1[0]:
            gender = st.selectbox("Gender",('Female','Male'))
        with cols1[1]:
            senior_citizen = st.selectbox("Senior Citizen",('Yes','No'))
        with cols1[2]:
            partner = st.selectbox("Partner",("Yes","No"))
        with cols1[3]:
            dependents = st.selectbox("Dependents",("Yes","No"))

        st.write("------")

        st.title('Customer Account Information')
        cols2 = st.columns(5)
#
        with cols2[0]:
            tenure_months = st.number_input("Tenure",value = 12.00, min_value = 0.0)


        with cols2[1]:
            contracts = st.selectbox("Contract Term",("Month-to-month","One year","Two year"))

        with cols2[2]:
            paper = st.selectbox("Paperless Billing",("Yes","No"))

        with cols2[3]:
            payment_method = st.selectbox("Payment Method",("Electric check","Mailed Checked","Bank Transfer","Credit Card"))
            # phone_service = st.selectbox("Phone Services",("Yes","No"))

        with cols2[4]:
            total_charges = st.number_input("Total Charges", value = 100.00, min_value =0.00)


        st.write("------")
        st.title("Services Information")
        cols3 = st.columns(3)
#####
        with cols3[0]:
            phone_service = st.selectbox("Phone Services",("Yes","No"))

        with cols3[1]:
            multiple_lines = st.selectbox("Multiple Lines",("Yes","No","No phone service"))

        with cols3[2]:
            internet_service = st.selectbox("Internet Service",("DSL","Fiber optic","No"))

        cols4 = st.columns(3)

        with cols4[0]:
            online_security = st.selectbox("Online Security",("Yes","No","No internet service"))

        with cols4[1]:
            online_backup = st.selectbox("Online Backup",("Yes","No","No internet service"))

        with cols4[2]:
            device_protetcion = st.selectbox("Device Protection",("Yes","No","No internet serivce"))


        cols5 = st.columns(3)

        with cols5[0]:
            tech_support = st.selectbox("Tech Support",("Yes","No","No internet service"))

        with cols5[1]:
            streaming_tv = st.selectbox("Streaming TV",("Yes","No","No internet service"))

        with cols5[2]:
            streaming_movies = st.selectbox("Streaming Movies",("Yes","No","No internet service"))

        # monthly_charges = st.number_input("Monthly Charges", value = 20, min_value = 0)
        # with cols6[2]:
        #     tenure_months = st.number_input("Tenure",value = 12.00, min_value = 0.0)

        dataframe = {'gender':gender,
                    'SeniorCitizen':senior_citizen,
                    'Partner':partner,
                    'Dependents':dependents,
                    'tenure':tenure_months,
                    'PhoneService':phone_service,
                    'MultipleLines':multiple_lines,
                    'InternetService':internet_service,
                    'OnlineSecurity':online_security,
                    'OnlineBackup': online_backup,
                    'DeviceProtection':device_protetcion,
                    'TechSupport':tech_support,
                    'StreamingTV': streaming_tv,
                    'StreamingMovies': streaming_movies,
                    'Contract': contracts,
                    'PaperlessBilling': paper,
                    'PaymentMethod': payment_method,
                    # 'MonthlyCharges': monthly_charges,
                    'TotalCharges':total_charges}
        features= pd.DataFrame(dataframe, index=[0])
        return features

    df_input = user_features()

    model = joblib.load("pipe.joblib")
    # st.write(model)
    # st.write(df_input)#
    # st.write(model.predict(df_input))

    st.write("")
    st.write("")
    st.write("")

    # button,result= st.columns([1,2])
    # with button:
    predict = st.button('Predict')

    # with result:
#########
    if predict:
        res = model.predict(df_input)
        if res == 0:
            st.write("")
            st.write("")

            col1,col2 = st.columns([0.1,1])
            with col1:
                st.image("https://cdn-icons-png.flaticon.com/512/709/709510.png",width =80)
                st.write('''
               <style>
                   img, svg {
                    'vertical-align': 'center';
                                }
               </style>
                ''', unsafe_allow_html=True)
            with col2:
                st.markdown("""<h3 style="color:#0178e4;font-size:35px;">
                   The customer with above information is not subject to Churn
                    </h3>""",unsafe_allow_html = True)
        else:
                 st.write("")
                 st.write("")

                 col3,col4 = st.columns([0.1,1])
                 with col3:
                     st.image("https://cdn-icons-png.flaticon.com/512/159/159469.png",width =80)
                     st.write('''
                    <style>
                        img, svg {
                         'vertical-align': 'center';
                                     }
                    </style>
                     ''', unsafe_allow_html=True)
                 with col4:
                     st.markdown("""<h3 style="color:#00284c;font-size:35px;">
                        The customer with above information is subject to Churn
                         </h3>""",unsafe_allow_html = True)



            # st#.markdown("""
            # <div>
            #     <img src = "https://cdn-icons-png.flaticon.com/512/159/159469.png" width = 50 style = "display:inline-block;vertical-align:middle;"/>
            #     <h3 style="color:#0178e4;font-size:50px;vertical-align:middle;">
            #     Churn
            #     </h3>
            # </div>""",unsafe_allow_html = True)

#
    # with result:
        # if predict:
        #     st.markdown("""
#
#
#
    # if predict:
    #     st.
###


######

    # with cc[1]:
    #     st.markdown("""
    #          <p class="f5 f4-ns lh-copy measure mb4" style="text-align: center;font-size:25px;">
    #           Customer Details
    #          </p>""", unsafe_allow_html = True)
    #     st.markdown(f"""
    #     <div style="vertical-align:center;font-size:18px;padding-left:30px;padding-top:30px;margin-left:1em";>
    #     Senior Citizen:&nbsp;{selected_id['SeniorCitizen'][0]}
    #     </div>""", unsafe_allow_html = True)
    #
    #
    #     st.markdown(f"""
    #         <div style="vertical-align:center;font-size:18px;padding-left:30px;padding-top:30px;margin-left:1em";>
    #         Partner:&nbsp;{selected_id['Partner'][0]}
    #         </div>""", unsafe_allow_html = True)
    #
    #     st.markdown(f"""
    #         <div style="vertical-align:center;font-size:18px;padding-left:30px;padding-top:30px;margin-left:1em";>
    #         Dependents:&nbsp;{selected_id['Dependents'][0]}
    #         </div>""", unsafe_allow_html = True)
    #
    #     st.markdown(f"""
    #         <div style="vertical-align:center;font-size:18px;padding-left:30px;padding-top:30px;margin-left:1em";>
    #         Tenure:&nbsp;{selected_id['tenure'][0]} Months
    #         </div>""", unsafe_allow_html = True)
    #
    #     # st.markdown(f"""
    #     #     <div style="vertical-align:center;font-size:18px;padding-left:30px;padding-top:30px;margin-left:1em";>
    #     #     Phone Service:&nbsp;{selected_id['PhoneService'][0]}
    #     #     </div>""", unsafe_allow_html = True)
    #     #
    #     # st.markdown(f"""
    #     #     <div style="vertical-align:center;font-size:18px;padding-left:30px;padding-top:30px;margin-left:1em";>
    #     #     Multiple Lines:&nbsp;{selected_id['MultipleLines'][0]}
    #     #     </div>""", unsafe_allow_html = True)
    #     #
    #     # st.markdown(f"""
    #     #     <div style="vertical-align:center;font-size:18px;padding-left:30px;padding-top:30px;margin-left:1em";>
    #     #     Internet Service:&nbsp;{selected_id['InternetService'][0]}
    #     #     </div>""", unsafe_allow_html = True)
    #     #
    #     # st.markdown(f"""
    #     #     <div style="vertical-align:center;font-size:18px;padding-left:30px;padding-top:30px;margin-left:1em";>
    #     #     Online Security:&nbsp;{selected_id['OnlineSecurity'][0]}
    #     #     </div>""", unsafe_allow_html = True)
    #     #
    #     # st.markdown(f"""
    #     #     <div style="vertical-align:center;font-size:18px;padding-left:30px;padding-top:30px;margin-left:1em";>
    #     #     Online Backup:&nbsp;{selected_id['OnlineBackup'][0]}
    #     #     </div>""", unsafe_allow_html = True)
    #     #
    #     # st.markdown(f"""
    #     #     <div style="vertical-align:center;font-size:18px;padding-left:30px;padding-top:30px;margin-left:1em";>
    #     #     Online Security:&nbsp;{selected_id['OnlineSecurity'][0]}
    #     #     </div>""", unsafe_allow_html = True)
    #
    #     st.markdown(f"""
    #         <div style="vertical-align:center;font-size:18px;padding-left:30px;padding-top:30px;margin-left:1em";>
    #         Payment Method:&nbsp;{selected_id['PaymentMethod'][0]}
    #         </div>""", unsafe_allow_html = True)
    #
    #     st.markdown(f"""
    #         <div style="vertical-align:center;font-size:18px;padding-left:30px;padding-top:30px;margin-left:1em";>
    #         Contract:&nbsp;{selected_id['Contract'][0]}
    #         </div>""", unsafe_allow_html = True)
    #
    #     st.markdown(f"""
    #         <div style="vertical-align:center;font-size:18px;padding-left:30px;padding-top:30px;margin-left:1em";>
    #         City:&nbsp;{selected_id['City'][0]}
    #         </div>""", unsafe_allow_html = True)
    #
    #     st.markdown(f"""
    #         <div style="vertical-align:center;font-size:18px;padding-left:30px;padding-top:30px;margin-left:1em";>
    #         Total Charges:&nbsp;${selected_id['TotalCharges'][0]}
    #         </div>""", unsafe_allow_html = True)

                # st.image('https://cdn-icons-png.flaticon.com/512/3135/3135715.png',width = 300)






        # st.write(f)
         # st.write(f"""
         #        <div>
         #            <div style="display:inline-block;vertical-align:center;">
         #            <img src="https://cdn-icons-png.flaticon.com/512/3126/3126649.png" width=100/>
         #            </div>
         #            <div style="display:inline-block;vertical-align:center;font-size:20px;padding-left: 30px;margin-left:1em";>
         #            {0}
         #            </div>
         #            <div style="display:inline-block;vertical-align:center;font-size:20px;padding-left: 30px;margin-left:1em";>
         #            {0}
         #            </div>""", unsafe_allow_html=True)
    # st.write(df1)
    # st.write(df1.shape)
    # profile(df1)
    # st.header('**Input DataFrame**')
    # profile(df1)
    # pr = ProfileReport(df1, explorative=True)
    # st_profile_report(pr)
    # pr = profile(df1)
    # pr#
    # profile(
