# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 17:54:08 2023

@author: hp
"""

from textblob import TextBlob
import streamlit as st
import pandas as pd
import numpy as np
#import plotly.graph_objects as go
#np.random.randn
st.title('NLP')
st.header('Text Analysis: AMAZON REVIEW DATA ')
df=pd.read_csv(r"_newfile (2).csv")
st.subheader('DATA POST EDA')
st.dataframe(df)



import pandas as pd
import streamlit as st


st.header('Sentiment Analysis:  USE PROJECT FILE{Product_details.csv}')
st.subheader('Enter a CSV with columns having Text in COLUMN NAME: Product_Description.')
with st.expander('Analyze Text'):
    text=st.text_input('Enter any Text here to check SUBJECTIVITY AND POLARITY: ')
    if text:
        blob = TextBlob(text)
        
        st.write('Polarity: ', round(blob.sentiment.polarity,2))
        st.write('Subjectivity: ', round(blob.sentiment.subjectivity,2))
       
        


with st.expander('Analyze CSV'):
    upl = st.file_uploader('Upload file')
    
    def score(x):
        blob1 = TextBlob(x)
        return blob1.sentiment.polarity
    
    def analyze(y):
        if y >= 0.5:
           return 'Positive'
        elif y <= -0.5:
            return 'Negative'
        else:
            return 'Neutral'
    
    def analyzee(z):
        if z >= 0.5:
           return 3
        elif z <= 0.5 and z >= 0:
           return 2
        elif z <= 0 and z >= -0.5:
            return 1
        elif z <= -0.5:
            return 0

    if upl:
         df4 = pd.read_csv(upl) 
         #del df4['Unnamed: 0']
         df4['Score']=df4['Product_Description'].apply(score)
         df4['Analysis']=df4['Score'].apply(analyze)
         df4['Sentiments']=df4['Score'].apply(analyzee)
         df4['Products'] = df4['Product_Description'].astype(str) +"-"+ df4["Product_Type"].astype(str)
         st.write(df4.head())
         st.write(df4.tail())
         st.write(df4.head(100))
         st.write(df4.tail(10))
         
         
         
         @st.cache
         def convert_df(df4):
         #IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df4.to_csv().encode('utf-8')
            
         csv = convert_df(df4)
         
         st.download_button(
             label="Download new data as CSV",
             data=csv,
             file_name='Sentiment.csv',
             mime='text/csv'
             )
             

         st.write(df4['Products'])

         text2=st.text_input('Checker: if Texts and ProductType is availabale in the Product_Details.csv: ') 
         st.write('The - is the divison between Reviews and the Reviewed ProductType')
         sx=df4['Products']
         
         def score(x):
             blob1 = TextBlob(x)
             return blob1.sentiment.polarity
         
         for i in sx:
             while text2 == i :
                 d=i
                 st.write(d)
                 break
         text4=st.text_input('Enter your Reviews text here to Predict SENTIMENT of the Review: ')
         if text4:
             blob6 = TextBlob(text4)
             xs=blob6.sentiment.polarity
                 
             st.write('Polarity: ', round(blob6.sentiment.polarity),2)
             st.write('Subjectivity: ', round(blob6.sentiment.subjectivity),2)
             if xs >= 0.5:
                st.write('Sentiment is 3')
             elif xs <= 0.5 and xs >= 0:
                st.write('Sentiment is 2')
             elif xs <= 0 and xs >= -0.5:
                st.write('Sentiment is 1')
             elif xs <= -0.5:
                st.write('Sentiment is 0')
             
              
             
         @st.cache
         def convert_df(df4):
         #IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
            
         csv = convert_df(df)
         
         st.download_button(
             label="Download data as CSV",
             data=csv,
             file_name='Sentiment1.csv',
             mime='text/csv'
             )
         
         
         st.subheader('MAXIMUM VALUES')
         g5=df4['Sentiments'].value_counts()
         st.write(g5)


         st.subheader('MAXIMUM VALUES')
         g6=df4['Analysis'].value_counts()
         st.write(g6)
         
         st.subheader('PRODUCT CROSS SENTIMENT ANALYSIS')
         g7=pd.crosstab(df4.Sentiments,df4.Product_Type)
         st.write(g7)
        
from textblob import TextBlob
import pandas as pd
import streamlit as st


st.header('Sentiment Analysis: USE PROJECT FILE{_newfile_.csv}')
st.subheader('Enter a CSV with columns having COLUMN NAME: Sentence')
    
with st.expander('Analyze CSV'):
    upl1 = st.file_uploader('Upload file2')
    
    def scoreq(x1):
        blob2 = TextBlob(x1)
        return blob2.sentiment.polarity
    
    def analyzeq(y1):
        if y1 >= 0.5:
           return 'Positive'
        elif y1 <= -0.5:
            return 'Negative'
        else:
            return 'Neutral'
    
    def analyzeeq(z1):
        if z1 >= 0.5:
           return 3
        elif z1 <= 0.5 and z1 >= 0:
           return 2
        elif z1 <= 0 and z1 >= -0.5:
            return 1
        elif z1 <= -0.5:
            return 0

    if upl1:
         df5 = pd.read_csv(upl1) 
         del df5['Unnamed: 0']
         df5['Score']=df5['sentence'].apply(scoreq)
         df5['Analysis']=df5['Score'].apply(analyzeq)
         df5['Sentiments']=df5['Score'].apply(analyzeeq)
         st.write(df5.head())
         st.write(df5.tail())
         st.write(df5.head(100))
         st.write(df5.tail(100))
         
         @st.cache
         def convert_df(df5):
         #IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df5.to_csv().encode('utf-8')
            
         csv = convert_df(df5)
         
         st.download_button(
             label="Download data as CSV",
             data=csv,
             file_name='Sentiments.csv',
             mime='text/csv'
             )
             






st.subheader('AFFINITY MODEL PROCESSED SENTIMENT SCORE')
st.area_chart(data=df, x=['SentimentScore'], y=['sentiment_value'], width=0, height=0)
st.subheader('FILTERED AFFINITY MODEL PROCESSED SENTIMENT SCORE')
st.area_chart(data=df, x=['Product_Type'], y=['SentimentScore'], width=0, height=0)
st.subheader('COMPARISON OF POSITIVE AND NEGATIVE EMOTIONS')
st.title('What does polarity mean in NLP? Polarity is float which lies in the range of [-1,1] where 1 means positive statement and -1 means a negative statement.')
st.area_chart(data=df, x=['SentimentScore'], y=['Polarity'], width=0, height=0)
st.subheader('FILTERED COMPARISON OF POSITIVE AND NEGATIVE EMOTIONS')
st.area_chart(data=df, x=['Product_Type'], y=['Polarity_rating'], width=0, height=0)
st.subheader('FILTERED COMPARISON OF FACTUAL AND NON-FACTUAL EMOTIONS')
st.area_chart(data=df, x=['SentimentScore'], y=['Subjectivity_rating'], width=0, height=0)
st.subheader('COMPARISON OF FACTUAL AND NON-FACTUAL EMOTIONS')
st.area_chart(data=df, x=['Product_Type'], y=['Subjectivity'], width=0, height=0)

st.subheader('NEGATIVE TEXT CLASSIFICATION SCORES')
st.area_chart(data=df, x=['SentimentScore'], y=['neg'], width=0, height=0)
st.subheader('NEGATIVE TEXT CLASSIFICATION SCORES HISTOGRAM')
st.set_option('deprecation.showPyplotGlobalUse', False)
#histogram
import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
df1 = pd.DataFrame(df, columns = ['neg'])
df1.hist()
plt.show()
st.pyplot()

st.subheader('NEUTRAL TEXT CLASSIFICATION SCORES')
st.area_chart(data=df, x=['Product_Type'], y=['neu'], width=0, height=0)
st.subheader('NEUTRAL TEXT CLASSIFICATION SCORES HISTOGRAM')
st.set_option('deprecation.showPyplotGlobalUse', False)
#histogram
import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
df1 = pd.DataFrame(df, columns = ['neu'])
df1.hist()
plt.show()
st.pyplot()

st.subheader('POSITIVE TEXT CLASSIFICATION SCORES')
st.area_chart(data=df, x=['SentimentScore'], y=['pos'], width=0, height=0)
st.subheader('POSITIVE TEXT CLASSIFICATION SCORES HISTOGRAM')
st.set_option('deprecation.showPyplotGlobalUse', False)
#histogram
import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
df1 = pd.DataFrame(df, columns = ['pos'])
df1.hist()
plt.show()
st.pyplot()

st.subheader('COMPOUND/OVERALLSUM TEXT CLASSIFICATION SCORES')
st.area_chart(data=df, x=['Product_Type'], y=['compound'], width=0, height=0)
st.subheader('COMPOUND/OVERALLSUM TEXT CLASSIFICATION SCORES HISTOGRAM')
st.set_option('deprecation.showPyplotGlobalUse', False)
#histogram
import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
df1 = pd.DataFrame(df, columns = ['compound'])
df1.hist()
plt.show()
st.pyplot()

st.subheader('PRODUCT_TYPE AREA GRAPH')
st.area_chart(data=df, x=['Product_Type'], y=['Product_Type'], width=0, height=0)

st.set_option('deprecation.showPyplotGlobalUse', False)

st.subheader('HISTOGRAM OF PRODUCT TYPE')
#histogram
import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
df1 = pd.DataFrame(df, columns = ['Product_Type'])
df1.hist()
plt.show()
st.pyplot()



st.subheader('HISTOGRAM OF SENTIMENT')
#histogram
import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
df1 = pd.DataFrame(df, columns = ['Sentiment'])
df1.hist()
plt.show()
st.pyplot()


st.subheader('HISTOGRAM OF SENTIMENTSCORE')
#histogram
import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
df1 = pd.DataFrame(df, columns = ['SentimentScore'])
df1.hist()
plt.show()
st.pyplot()


st.subheader('HISTOGRAM OF SENTIMENT_VALUE')
#histogram
import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
df1 = pd.DataFrame(df, columns = ['sentiment_value'])
df1.hist()
plt.show()
st.pyplot()


st.subheader('HISTOGRAM OF POLARITY')
#histogram
import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
df1 = pd.DataFrame(df, columns = ['Polarity_rating'])
df1.hist()
plt.show()
st.pyplot()


st.subheader('HISTOGRAM OF SUBJECTIVITY')
#histogram
import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
df1 = pd.DataFrame(df, columns = ['Subjectivity_rating'])
df1.hist()
plt.show()
st.pyplot()





st.subheader('ADJUSTED SENTIMENT ACCORDING TO PROVIDED DATA')
st.area_chart(data=df, x=['Product_Type'], y=['rating'], width=0, height=0)
st.subheader('SENTIMENT ACCORDING TO DATA')
st.area_chart(data=df, x=None, y=['Sentiment'], width=0, height=0)


st.subheader('BARPLOT OF SENTIMENT AND COMPOUND')
import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
sns.barplot(data=df,x='Sentiment', y = 'compound')
plt.show()
st.pyplot()

import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
df.Product_Type.plot.density(color='green')
plt.title('Density plot for Product_Type')
plt.show()
st.pyplot()


df.Sentiment.plot.density(color='green')
plt.title('Density plot for Sentiment')
plt.show()
st.pyplot()



st.subheader('SENTIMENT WORDCLOUD')
import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from wordcloud import WordCloud

wordcloud = WordCloud(min_font_size=10, \
                      max_font_size=50, max_words=50, \
                      background_color="white", colormap = "Oranges")

zero_star_text = " ".join(df[df["Sentiment"]==0]["Product_Description"].values).lower()
one_star_text = " ".join(df[df["Sentiment"]==1]["Product_Description"].values).lower()
two_star_text = " ".join(df[df["Sentiment"]==2]["Product_Description"].values).lower()
three_star_text = " ".join(df[df["Sentiment"]==3]["Product_Description"].values).lower()

text_list = [zero_star_text, one_star_text, two_star_text, \
             three_star_text]

for index, text in enumerate(text_list):
    f, axes = plt.subplots(figsize=(10,7))
    wordcloud.generate(text)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title(f"Word Cloud for {index+1}-Star Ratings")
    plt.axis("off")
    plt.show()
    st.pyplot()




st.subheader('POLARITY WORDCLOUD')
import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from wordcloud import WordCloud

wordcloud = WordCloud(min_font_size=10, \
                      max_font_size=50, max_words=50, \
                      background_color="white", colormap = "Oranges")

zero_star_text = " ".join(df[df["Polarity_rating"]==-1]["Product_Description"].values).lower()
one_star_text = " ".join(df[df["Polarity_rating"]==0]["Product_Description"].values).lower()
two_star_text = " ".join(df[df["Polarity_rating"]==1]["Product_Description"].values).lower()
#three_star_text = " ".join(df[df["Sentiment"]==3]["Product_Description"].values).lower()

text_list = [zero_star_text, one_star_text, two_star_text, \
             ]

for index, text in enumerate(text_list):
    f, axes = plt.subplots(figsize=(10,7))
    wordcloud.generate(text)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title(f"Word Cloud for {index+1}-Star Ratings")
    plt.axis("off")
    plt.show()
    st.pyplot()



import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from wordcloud import WordCloud

st.subheader('MAXIMUM VALUES')
g=df['Product_Type'].value_counts()
st.write(g)

st.subheader('MAXIMUM VALUES')
g1=df['SentimentScore'].value_counts()
st.write(g1)

st.subheader('MAXIMUM VALUES')
g2=df['sentiment_value'].value_counts()
st.write(g2)

st.subheader('MAXIMUM VALUES')
g3=df['Sentiment'].value_counts()
st.write(g3)

st.subheader('MAXIMUM VALUES')
g4=df['compound'].value_counts()
st.write(g4)







gk = df.groupby('Product_Type')
  
# Let's print the first entries
# in all the groups formed.

st.subheader('PRODUCT TYPE SENTIMENT SCORES')
ttt=gk.first()
st.write(ttt)
st.subheader('PRODUCT TYPE 0 SENTIMENT SCORES')
hhh=gk.get_group(0)['Sentiment'].value_counts()
st.write(hhh)
st.subheader('PRODUCT TYPE 1 SENTIMENT SCORES')
hhhh=gk.get_group(1)['Sentiment'].value_counts()
st.write(hhhh)
st.subheader('PRODUCT TYPE 2 SENTIMENT SCORES')
hhhhh=gk.get_group(2)['Sentiment'].value_counts()
st.write(hhhhh)
st.subheader('PRODUCT TYPE 3 SENTIMENT SCORES')
zzz=gk.get_group(3)['Sentiment'].value_counts()
st.write(zzz)
st.subheader('PRODUCT TYPE 4 SENTIMENT SCORES')
zzzz=gk.get_group(4)['Sentiment'].value_counts()
st.write(zzzz)
st.subheader('PRODUCT TYPE 5 SENTIMENT SCORES')
zzzzzz=gk.get_group(5)['Sentiment'].value_counts()
st.write(zzzzzz)
st.subheader('PRODUCT TYPE 6 SENTIMENT SCORES')
uuu=gk.get_group(6)['Sentiment'].value_counts()
st.write(uuu)
st.subheader('PRODUCT TYPE 7 SENTIMENT SCORES')
uuuu=gk.get_group(7)['Sentiment'].value_counts()
st.write(uuuu)
st.subheader('PRODUCT TYPE 8 SENTIMENT SCORES')
uuuuu=gk.get_group(8)['Sentiment'].value_counts()
st.write(uuuuu)
st.subheader('PRODUCT TYPE 9 SENTIMENT SCORES')
wwwwwwww=gk.get_group(9)['Sentiment'].value_counts()
st.write(wwwwwwww)




def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://media.istockphoto.com/id/1352614032/vector/wave-3d-wave-of-particles-abstract-white-geometric-background-big-data-technology.jpg?s=612x612&w=0&k=20&c=59qnEW3AfJrAlx5Hhic0-hLzrIARBCzgv7d-DL7Bu-0=");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 

st.subheader('HEATMAP OF CORRELATION IN DATA')
import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
sns.heatmap(df.corr(), ax=ax)
st.write(fig)

