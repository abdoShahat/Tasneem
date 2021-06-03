#!/usr/bin/env python
# coding: utf-8

# In[9]:


import streamlit as st  ## streamlit
import pandas as pd  ## for data manipulation

from PIL import Image   ## For image
from io import StringIO  ## for text input and output from the web app


# In[10]:


from sklearn.externals import joblib


# # Loading models

# In[11]:



new_model = joblib.load('token_logesticreg.pkl')
new_token = joblib.load('token_tfidf.pkl')


# In[12]:


#new prediction
def predict(tweet):
    
    tweet = new_token.transform([tweet])
    result = new_model.predict(tweet)
    final = 'sincere' if result[0]==0 else 'insincere'

    return final


# In[13]:


predict("trump")


# In[ ]:





# In[14]:


predict('How did Quebec nationalists see their province as a nation in the 1960s?')


# In[15]:


def run():
    st.sidebar.info('You can either enter the news item online in the textbox or upload a txt file')
    st.set_option('deprecation.showfileUploaderEncoding', False)
    add_selectbox = st.sidebar.selectbox("How would you like to predict?", ("Online", "Txt file"))
    st.title("Predicting sincerety ")
    st.header('This app is created to predict if a question is sincere or not')
    if add_selectbox == "Online":
        text1 = st.text_area('Enter text')
        output = ""
        if st.button("Predict"):
            output = predict(text1)
            output = str(output) # since its a list, get the 1st item
            st.success(f"The news item is {output}")
            st.balloons()
    elif add_selectbox == "Txt file":
        output = ""
        file_buffer = st.file_uploader("Upload text file for new item", type=["txt"])
        if st.button("Predict"):
             text_news = file_buffer.read()

# in the latest stream-lit version ie. 68, we need to explicitly convert bytes to text
        st_version = st.__version__ # eg 0.67.0
        versions = st_version.split('.')
        if int(versions[1]) > 67:
                text_news = text_news.decode('utf-8')
                print(text_news)
        output = predict(text_news)
        output = str(output)
        st.success(f"The news item is {output}")
        st.balloons()


# In[16]:


if __name__ == "__main__":

    run()


# In[ ]:





# In[ ]:




