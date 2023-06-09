import streamlit as st
from PIL import Image
import pickle
import string
import re
import nltk
from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

# from copy import transform_text
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
import time

hide_menu = """
<style>
#MainMenu{
    visibility:hidden;
}
footer{
    visibility:hidden;
}
</style>
"""

showWarningOnDirectExecution = False
ps = PorterStemmer()
image = Image.open('icons/logo.png')


st.set_page_config(page_title = "Cyberbullying Detection", page_icon = image)

st.markdown(hide_menu, unsafe_allow_html=True)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.image(image , use_column_width=True, output_format='auto')


st.sidebar.markdown("---")


st.sidebar.markdown("<br> <br> <br> <br> <br> <br> <h1 style='text-align: center; font-size: 18px; color: #0080FF;'>© 2023 | Pranav Naik</h1>", unsafe_allow_html=True)


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

# def strip_emoji(text):
#     return emoji.replace_emoji(text,replace="")

def strip_all_entities(text):
    stop_words = set(stopwords.words('english')) 
    text = text.replace('\r', '').replace('\n', ' ').lower()
    text = re.sub(r"(?:\@|https?\://)\S+", "", text)
    text = re.sub(r'[^\x00-\x7f]',r'', text)
    text = re.sub(r'(.)1+', r'1', text)
    text = re.sub('[0-9]+', '', text)
    stopchars= string.punctuation
    table = str.maketrans('', '', stopchars)
    text = text.translate(table)
    text = [word for word in text.split() if word not in stop_words]
    text = ' '.join(text)
    return text

def decontract(text):
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    return text

def clean_hashtags(text):
    tweet = " ".join(word.strip() for word in re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', text))
    clean_tweet = " ".join(word.strip() for word in re.split('#|_', tweet))
    return clean_tweet

def filter_chars(text):
    sent = []
    for word in text.split(' '):
        if ('$' in word) | ('&' in word):
            sent.append('')
        else:
            sent.append(word)
    return ' '.join(sent)

def remove_mult_spaces(text):
    return re.sub("\s\s+" , " ", text)

def stemmer(text):
    tokenized = nltk.word_tokenize(text)
    ps = PorterStemmer()
    return ' '.join([ps.stem(words) for words in tokenized])

def lemmatize(text):
    tokenized = nltk.word_tokenize(text)
    lm = WordNetLemmatizer()
    return ' '.join([lm.lemmatize(words) for words in tokenized])

def preprocess(text):
    # text = strip_emoji(text)
    text = decontract(text)
    text = strip_all_entities(text)
    text = clean_hashtags(text)
    text = filter_chars(text)
    text = remove_mult_spaces(text)
    text = stemmer(text)
    text = lemmatize(text)
    return text

tfidf = pickle.load(open('pickle/TFIDFvectorizer.pkl','rb'))
model = pickle.load(open('pickle/model.pkl','rb'))

vectoriser = pickle.load(open("pickle/vectorizer.pkl", "rb"))

st.title("Cyber-Bullying Detection🔍")
st.markdown("---")
st.markdown("<br>", unsafe_allow_html=True)
input_text = st.text_area("**_Enter the text to analyze_**", key="**_Enter the text to analyze_**")
col1, col2 = st.columns([1,6])
with col1:
    button_predict = st.button('Predict')
with col2:

    def clear_text():
        st.session_state["**_Enter the text to analyze_**"] = ""

    # clear button
    button_clear = st.button("Clear", on_click=clear_text)

st.markdown("---")
    # predict button animations
if button_predict:
    if input_text == "":
     st.snow()
     st.warning("Please provide some text!")
    else:
        with st.spinner("**_Prediction_** in progress. Please wait 🙏"):
            time.sleep(3)
    # 1. preprocess

        # cleanText = clean_text(input_text)

        # transformText = transform_text(cleanText)

    # 2. vectorize

        # vector_input = tfidf.transform([transformText])
    # 3. predict
        text = vectoriser.transform([input_text])
        result = model.predict(text)


        # result2 = model.predict_proba(vector_input)[0] 
        #clf=svm.SVC(probability=True)

        clean_text = strip_all_entities(input_text)
        decontract_text = decontract(clean_text)
        hastags_removed = clean_hashtags(decontract_text)
        filtered_text = filter_chars(hastags_removed)
        mult_text = remove_mult_spaces(filtered_text)
        stemmer_text = stemmer(mult_text)
        lemmatize_text = lemmatize(stemmer_text)
        

    # 4. display

         
        if result == 1 :
            st.subheader("Result")
            st.error(":red[**_Religion_**]")
            # st.markdown(result2)
        elif result == 2 :
            st.subheader("Result")
            st.error(":red[**_Age_**]")
        elif result == 3 :
            st.subheader("Result")
            st.error(":red[**_Ethnicity_**]")
        elif result == 4 :
            st.subheader("Result")
            st.error(":red[**_Gender_**]")
        elif result == 5 :
            st.subheader("Result")
            st.error(":red[**_Other Cyberbullying_**]")
        
        else:
            st.subheader("Result")
            st.success(":green[**_Not Cyberbullying_**]")
            # st.markdown(result2)
        st.markdown("---")
        st.subheader("Original Text")
        expander_original = st.expander("Information", expanded=False)
        with expander_original:
            st.info("The text that the user provided!")
        st.text(input_text)
        st.markdown("---")
        st.subheader("Cleaned Text")
        expander_clean = st.expander("Information", expanded=False)
        with expander_clean:
            st.info("From original text has removed punctuation and special characters. Also it has removed hashtags, tags and emoji's!")
        st.text(clean_text)
        st.markdown("---")
        st.subheader("Decontracted Text")
        expander_decontract = st.expander("Information", expanded=False)
        with expander_decontract:
            st.info("From Cleaned text has removed stopwords and transformed to lowercase. Also, it has be used Stemming!")
        st.text(decontract_text)
        st.markdown("---")
        st.subheader("Hashtag removed Text")
        expander_hastag = st.expander("Information", expanded=False)
        with expander_hastag:
            st.info("From Cleaned text has removed stopwords and transformed to lowercase. Also, it has be used Stemming!")
        st.text(hastags_removed)
        st.markdown("---")
        st.subheader("Filtered Text")
        expander_filter = st.expander("Information", expanded=False)
        with expander_filter:
            st.info("From Cleaned text has removed stopwords and transformed to lowercase. Also, it has be used Stemming!")
        st.text(filtered_text)
        st.markdown("---")
        st.subheader("Multispaces removed Text")
        expander_mul = st.expander("Information", expanded=False)
        with expander_mul:
            st.info("From Cleaned text has removed stopwords and transformed to lowercase. Also, it has be used Stemming!")
        st.text(mult_text)
        # st.markdown("---")
        # st.subheader("Hashtag removed Text")
        # expander_hastag = st.expander("Information", expanded=False)
        # with expander_hastag:
        #     st.info("From Cleaned text has removed stopwords and transformed to lowercase. Also, it has be used Stemming!")
        # st.text(hastags_removed)
        # st.markdown("---")
        # st.subheader("Binary Prediction")
        # expander_binary = st.expander("Information", expanded=False)
        # with expander_binary:
        #     st.info("Binary Prediction from the Model!")
        # if result == 1:
        #     st.markdown(":red["+ str(result) +"]")
        # else:
        #     st.markdown(":green["+ str(result) +"]")
        # st.markdown("---")
        # st.subheader("Model Accuracy")
        # expander_accuracy = st.expander("Information", expanded=False)
        # with expander_accuracy:
        #     st.info("Model Accuracy using Random Forest (RF) Classifier!")
        # st.warning("Accuracy:  **_91.70 %_**")
        # st.markdown("---")