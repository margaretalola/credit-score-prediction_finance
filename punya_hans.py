import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import requests
import gdown

DATA = "https://raw.githubusercontent.com/hng011/people-personality-analysis/refs/heads/main/datasets/people_personality_types.csv"
ENCODED_DATA = "https://raw.githubusercontent.com/hng011/people-personality-analysis/refs/heads/main/datasets/encoded_data.csv"

def fetch_data(end_point):
    res = requests.get(end_point)
    if res.status_code == 200:
        return res.content
    else:
        print("Something went wrong")

def distribution_of_interests(df):
    st.header("DISTRIBUTION OF PEOPLE'S INTEREST")
    colors = ["#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
    fig = plt.figure()
    sns.countplot(df["Interest"], order=df["Interest"].value_counts().index, palette=colors)
    st.pyplot(fig)

def home(df):
    st.header("DISTRIBUTION OF 16 PERSONALITIES 👀")
    
    with st.container():
        total_male = len(df[df["Gender"] == "Male"])
        total_female = len(df[df["Gender"] == "Female"])    
    
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"Total Data: {df.shape[0]}")
            st.download_button("Download Data 📃", 
                               data=fetch_data(DATA), 
                               file_name="people_personality_types.csv",
                               mime="text/csv") 
        with col2:
            st.subheader(f"Total Male\t: {total_male}")
            st.subheader(f"Total Female\t: {total_female}")
    
    fig = plt.figure()
    sns.barplot(data=df.groupby(["Personality", "Gender"])["Gender"].count().reset_index(name="Count").sort_values(by="Count", ascending=False), 
                  y="Count", 
                  x="Personality", 
                  hue="Gender",
                  palette=["#72BCD4", "#D3D3D3"])
    
    plt.xticks(rotation=45)
    st.pyplot(fig)    

def get_person_characteristic():

    with st.form(key="demographic"):
        age = st.number_input("Age:", min_value=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        education = st.selectbox("Education Level", ["Graduate-lever or Higher", "Undergraduate or Lower"])
        introversion_sc = st.selectbox("Introversion Score | Represent the individual's tendency toward introversion versus extraversion:", 
                                    [1, 2, 3, 4, 5])
        sensing_sc = st.selectbox("Sensing Score | Represent the individual's preference for sensing versus intuition", 
                                    [1, 2, 3, 4, 5])
        thinking_sc = st.selectbox("Thinking Score | Indicate the individual's preference for thinking versus feeling", 
                                    [1, 2, 3, 4, 5])
        judging_sc = st.selectbox("Judging Score | Represent the individual's preference for judging versus perceiving",
                                    [1, 2, 3, 4, 5])
        interest = st.selectbox("Interest", ["Arts", "Sports", "Technology", "Other", "Unknown"])
        
        submit_btn = st.form_submit_button(label="NGAK🦅")

        success = True
        list_data = [age, gender, education, introversion_sc, sensing_sc, thinking_sc, judging_sc, interest]
        
    if submit_btn:
        return list_data, success
    else:
        list_data, success = [], False
        return list_data, success
        
@st.cache_data
def load_model(data_model):
    url = data_model
    out = "model.pkl"
    model = None

    try:
        gdown.download(url, out, quiet=False)
        model = joblib.load(out)
    except Exception as e:
        st.write(e)

    return model


def train_model(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
    model = RandomForestClassifier(n_estimators=132)
    model.fit(X_train, y_train)
    return model
    

def predict(df):
    """
    AGE                 : num 
    GENDER              : categorical (Male || Female)
    EDUCATION           : binary (1: graduate-level edu or higher | 0: undergraduate and lower)
    INTROVERSION SCORE  : 0 - 10 | Represent the individual's tendency toward introversion versus extraversion
    SENSING SCORE       : 0 - 10 | Represent the individual's preference for sensing versus intuition
    THINKING SCORE      : 0 - 10 | Indicate the individual's preference for thinking versus feeling
    JUDGING SCORE       : 0 - 10 | Represent the individual's preference for judging versus perceiving
    INTEREST            : 5 sectors
    """
    
    # DATA ENCODING
    GENDER = {'Male': 0.548, 'Female': 0.452}
    
    PERSONALITIES = {'ENFP': 34,
                    'INTP': 30,
                    'INFJ': 19,
                    'INFP': 87,
                    'ESTP': 79,
                    'ENFJ': 88,
                    'ENTP': 41,
                    'ENTJ': 35,
                    'INTJ': 46,
                    'ESFP': 24,
                    'ISTP': 48,
                    'ESFJ': 17,
                    'ISFP': 71,
                    'ISTJ': 78,
                    'ESTJ': 68,
                    'ISFJ': 14}
    
    reversed_pers_dict = {val: key for key, val in PERSONALITIES.items()}
    
    INTERESTS = {'Unknown': 0.381,
                'Arts': 0.2,
                'Others': 0.17,
                'Technology': 0.148,
                'Sports': 0.101}

    scores = {1: 1.0, 
              2: 2.5, 
              3: 5.0, 
              4: 7.5, 
              5:9}

    # GET INPUT
    list_data, success = get_person_characteristic()

    if success:
        age, gender, education, introversion_sc, sensing_sc, thinking_sc, judging_sc, interest = list_data
        encoded_df = pd.read_csv(ENCODED_DATA)
        age = float(age)
        gender = GENDER[gender]
        education = 1 if education == "Graduate-lever or Higher" else 0
        introversion_sc = scores[introversion_sc]
        sensing_sc = scores[sensing_sc]
        thinking_sc = scores[thinking_sc]
        judging_sc = scores[judging_sc]
        interest = INTERESTS[interest]
        
        # st.write(age, gender, education, introversion_sc, sensing_sc, thinking_sc, judging_sc, interest)
        
        data_input = {
            "Age":age, "Gender":gender, "Education":education, "Introversion Score":introversion_sc,
            "Sensing Score": sensing_sc, "Thinking Score": thinking_sc, "Judging Score": judging_sc,
            "Interest": interest 
        }

        try:
            # Scaling
            scaler = RobustScaler()
            encoded_df.iloc[:, :8] = scaler.fit_transform(encoded_df.iloc[:, :8])
            
            inputted_data = pd.DataFrame(data_input, index=[0]) # Ensure that the values in your dictionary are lists (even if they contain a single element) when creating a DataFrame from multiple rows.
            inputted_data.iloc[:, :] = scaler.transform(inputted_data.iloc[:,:])

            st.dataframe(inputted_data)
           
            # load model
            file_model = "https://drive.google.com/uc?id=18vwrvGCkXfUfXpfBBJDQ1wB8fxUXV5LP"  
            # clf = load_model(file_model)

            try:
                clf = train_model(encoded_df.iloc[:, :8], encoded_df["Personality"], .3) 
                pred = clf.predict(inputted_data)
            except Exception as e:
                st.write(e)

            if clf:
                st.write("Personality:",reversed_pers_dict[pred[0]])
            else:
                st.write("Unable to load the model")
            
        except Exception as e:
            st.write(e)
    else:
        st.write("🤸🤸🤸")

    
def main():
    df = pd.read_csv(DATA)
    options = ["Home", "Distribution of Interests", "Predict"]

    with st.sidebar:
        try:
            selected = option_menu(menu_title="Dashboard Menu",
                                   options=options,
                                   default_index=0)

        except:
            st.write("streamlit_option_menu was not found")
            st.write("Try to install the module using the following command")
            st.write("`pip install streamlit-option-menu`")

    if selected == options[0]:
        home(df)

    if selected == options[1]:
        distribution_of_interests(df)
    
    if selected == options[2]:
        predict(df)

if __name__ == "__main__":
    main()