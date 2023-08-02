import streamlit as st
import pandas as pd
import openai
from datetime import datetime
import os

# Define your API key for OpenAI
openai.api_key = os.environ.get("OPENAI_API_KEY")

@st.cache(allow_output_mutation=True, show_spinner=False)
def load_data(csv_file):
    if csv_file is not None:
        chunksize = 5  # Choose a value that fits your needs
        df = pd.concat(pd.read_csv(csv_file, chunksize=chunksize))
        return df

def save_query(df, query):
    if df is None:
        df = pd.DataFrame(columns=["Query"])
        
    new_row = {"Query": query}
    df = df.append(new_row, ignore_index=True)
    return df


def get_gpt3_response(query):
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
            {"role": "system", "content": "You are a friendly assistant."},
            {"role": "user", "content": query},
        ]
    )
    return response['choices'][0]['message']['content']

def run_chat():
    st.title('Streamlit Chat App')

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    chat_history = load_data(uploaded_file)

    if chat_history is not None:
        st.subheader('Enter your queries')
        
        query = st.text_input('Query')

        if st.button('Send'):
            chat_history = save_query(chat_history, query)
            gpt3_response = get_gpt3_response(query)
            chat_history = save_query(chat_history, gpt3_response)

        # Display chat history
        st.table(chat_history)

if __name__ == "__main__":
    run_chat()
