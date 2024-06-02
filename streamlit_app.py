import os, tempfile
from dotenv import load_dotenv
import streamlit as st
import pandas as pd


light = '''
<style>
    .stApp {
    background-color: white;
    }
</style>
'''

dark = '''
<style>
    .stApp {
    background-color: black;
    }
</style>
'''

def make_false():
    # When toggle 1 is True, set toggle 2 to False
    if st.session_state.t2:
        st.session_state.t1 = False


# Template Configuration
st.markdown(dark, unsafe_allow_html=True)

if __name__ == "__main__":
    
    load_dotenv()

    # Streamlit app
    st.subheader("Chat with Your Dataset")

    # Get OpenAI API key, PROXYCURL API key and SERPAPI KEY, and source document input
    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API key", type="password")

    os.environ["OPENAI_API_KEY"] = openai_api_key

    iris_toggle, uploaded_file = st.columns(2)

    # Iris DataSet
    iris_dataset_flag = iris_toggle.toggle("If you don't have any .csv file to upload, you can chat & analyze the Iris dataset.", 
                                  value=True,
                                  key="t1",
                                  on_change=make_false)

    # File uploader
    uploaded_file_check = uploaded_file.file_uploader("Upload your csv file to chat & analyze!", key="t2", label_visibility="visible")

    if uploaded_file_check is None and iris_dataset_flag: 
        st.write("You can now chat & analyze the Iris dataset!")
        df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
        st.dataframe(df) # Display the DataFrame

    elif uploaded_file_check is not None: 
        # Save uploaded file temporarily to disk, load and split the file into pages, delete temp file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file_check.read())

        # Read the data file into a DataFrame
        df = pd.read_csv(tmp_file.name, header=None) # Assuming there's no header in the file
        os.remove(tmp_file.name)

        # Display the DataFrame
        st.dataframe(df)




    if st.button("Submit"):

        # Validate inputs
        try:
            pass
            # I will continue from here!
            # Ask questions using OpenAI

        except Exception as e:
            st.error(f"An error occurred: {e}")