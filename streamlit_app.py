import os, tempfile
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from pandasai import SmartDataframe
from pandasai.responses.response_parser import ResponseParser
from pandasai.llm import OpenAI
from pandasai.callbacks import BaseCallback
import json


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

# Template Configuration
st.markdown(dark, unsafe_allow_html=True)

class StreamlitResponse(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        st.dataframe(result["value"])
        return

    def format_plot(self, result):
        st.image(result["value"])
        return

    def format_other(self, result):
        st.write(result["value"])
        return


class StreamlitCallback(BaseCallback):
    def __init__(self, container) -> None:
        """Initialize callback handler"""
        self.container = container

    def on_code(self, response: str):
        self.container.code(response)


def make_false():
    # When toggle 2 is True, set toggle 1 to False
    if st.session_state.t2:
        st.session_state.t1 = False


def extract_input_output(result):
    input_cmds = [step[0].tool_input for step in result['intermediate_steps']]
    output = result['output']
    return input_cmds, output


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
    uploaded_file_check = uploaded_file.file_uploader("Upload your csv file to chat & analyze!",
                                                       key="t2",
                                                       label_visibility="visible",
                                                       on_change=make_false)

    if uploaded_file_check is None and iris_dataset_flag: 
        st.write("You can now chat & analyze the Iris dataset!")
        df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
        st.dataframe(df) # Display the DataFrame

        default_text = """Do some analysis with at least 3 plots, use a subplot for each graph so they can be shown at the same time, use matplotlib for the graphs."""

        query = st.text_area("Pose a question you're eager to delve into regarding the iris dataset!", default_text)

    elif uploaded_file_check is not None: 
        # Save uploaded file temporarily to disk, load and split the file into pages, delete temp file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file_check.read())

        # Read the data file into a DataFrame
        df = pd.read_csv(tmp_file.name, header=None) # Assuming there's no header in the file
        os.remove(tmp_file.name)

        # Display the DataFrame
        st.dataframe(df)

        default_text = """Do some analysis with at least 3 plots, use a subplot for each graph so they can be shown at the same time, use matplotlib for the graphs."""

        query = st.text_area("Pose a question you're eager to delve into regarding the dataset!", default_text)
 
    container = st.container()

    if st.button("Submit"):
        try:
            if query == default_text:

                intermediate_steps = ["```python\nimport matplotlib.pyplot as plt\n\nfig, axs = plt.subplots(3, figsize=(10,15))\n\n# Histogram of sepal length\naxs[0].hist(df['sepal_length'], bins=10, color='blue', alpha=0.7)\naxs[0].set_title('Histogram of Sepal Length')\n\n# Scatter plot of sepal width vs petal length\naxs[1].scatter(df['sepal_width'], df['petal_length'], color='green', alpha=0.7)\naxs[1].set_title('Scatter Plot of Sepal Width vs Petal Length')\naxs[1].set_xlabel('Sepal Width')\naxs[1].set_ylabel('Petal Length')\n\n# Bar plot of species count\nspecies_count = df['species'].value_counts()\naxs[2].bar(species_count.index, species_count.values, color='red', alpha=0.7)\naxs[2].set_title('Bar Plot of Species Count')\n\nplt.tight_layout()\nplt.show()\n```"]

                st.write("Intermediate Steps: ", intermediate_steps)

                st.write("""The code successfully creates three plots: a histogram of sepal length, a scatter plot of sepal width 
                        vs petal length, and a bar plot of the count of each species. These plots are displayed in a single figure 
                        with three subplots.""")

            elif query != default_text and openai_api_key:
                
                #llm = OpenAI(model_name='gpt-4', temperature=0, verbose=True)
                #query_engine = SmartDataframe(
                #    df,
                #    config={
                #       "llm": llm,
                #        "response_parser": StreamlitResponse,
                #        "callback": StreamlitCallback(container),
                #    },
                #)
                #answer = query_engine.chat(query)
                #st.write(answer) # Return the result

                llm = ChatOpenAI(model_name='gpt-4', temperature=0, verbose=True)
                agent = create_pandas_dataframe_agent(llm, df, verbose=True, return_intermediate_steps=True)
                answer = agent.invoke(query)
                input_cmds, output = extract_input_output(answer)
                st.write("Intermediate Steps: ", input_cmds)
                st.write(output) # Return the final answer

            else:
                st.error(f"""
                        Sorry we cannot provide a summary for your prompt. \n
                        Please provide your OpenAI API key. \n
                        You can only search for the default prompt for free!
                        """)

        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Unset environment variables
    if 'OPENAI_API_KEY' in os.environ:
        del os.environ['OPENAI_API_KEY']
    
