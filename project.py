#%% IMPORT1
from selenium import webdriver
import pandas as pd
from time import sleep
import pathlib
import glob
import os
from pygooglenews import GoogleNews
from leia import SentimentIntensityAnalyzer
import yfinance as yf
import datetime
from datetime import date

#%% IMPORT2

import seaborn as sns
import matplotlib.pylab as plt
from unidecode import unidecode
import re
import spacy
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


#%% Text Mining

gn = GoogleNews(lang = 'pt', country = 'BR')

options = webdriver.ChromeOptions()
options.add_argument("--headless=new")


wd = webdriver.Chrome(options=options)

def busca_carteira_teorica(indice, espera=8):
  url = f'https://sistemaswebb3-listados.b3.com.br/indexPage/day/{indice.upper()}?language=pt-br'  
  wd.get(url)
  sleep(espera)

  wd.find_element_by_id('segment').send_keys("Setor de Atuação")
  sleep(espera)

  wd.find_element_by_link_text("Download").click()
  sleep(espera)
  
  list_of_files = glob.glob(str(pathlib.Path.home())+'/Downloads/*.csv')
  latest_file = max(list_of_files, key=os.path.getctime)
  
  return pd.read_csv(latest_file, sep=';', encoding='ISO-8859-1',skipfooter=2, engine='python', thousands='.', decimal=',', header=1, index_col=False)



ibov = busca_carteira_teorica('ibov',5)


tickers = ibov['Código']


#%% Searching

df = pd.DataFrame(columns=['Code', 'Title','Score_LeIA','LeIA'])

end = date.today()

start = end-datetime.timedelta(days=1)




for i in range(0,len(tickers)):
    search = gn.search(f'"{tickers[i]}" -varzea', from_=start.strftime('%Y-%m-%d'), to_=end.strftime('%Y-%m-%d'))
    for item in search['entries']:
        if tickers[i] in item.title:
            df = df.append(pd.Series([tickers[i], item.title], index = ["Code","Title",]), ignore_index=True)
            
#%% Sentiment Analysis- LeIA

sia = SentimentIntensityAnalyzer()
    

for i in range(0,len(df)):
    scores = sia.polarity_scores(df['Title'].iloc[i])
    
    if scores['compound'] >= 0.05:
        sentiment = 'positive'
    elif scores['compound'] <= -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
        
    df['Score_LeIA'].iloc[i] = scores['compound']
    df['LeIA'].iloc[i] = sentiment
       
    
#%% Sentiment Analysis ML - A partir de um trabalho no Kaggle: https://www.kaggle.com/code/zeneto11/ml-nlp/notebook (Spicy RF foi o melhor caso)
    
df_sent = pd.read_csv('financial_phrase_bank_pt_br.csv')

df_train = df_sent[:-400]
df_test = df_sent[-400:]

df_negative = df_train[df_train['y'] == 'negative']
df_neutral = df_train[df_train['y'] == 'neutral'].sample(n=len(df_negative) // 3, random_state=42)
df_positive = df_train[df_train['y'] == 'positive'].sample(n=len(df_negative) // 3, random_state=42)


# Concatenar os DataFrames resampleados
df_balanced = pd.concat([df_negative, df_neutral, df_positive])

# Juntar com as ocorrências preservadas de teste
df_con = pd.concat([df_balanced, df_test])

# Dataset somente com as notícias em português
df_pt = df_con.drop('text', axis=1)

# Removendo acentos e pontuações
df_pt['text_pt'] = df_pt['text_pt'].apply(lambda x: unidecode(re.sub(r'[^\w\s]', '', x)))


# Utilizar POS tagging do spaCy - remove todas as palavras que não são substantivos, verbos ou adjetivos 
df_pt_spc = df_pt.copy()

nlp = spacy.load('pt_core_news_sm')
df_pt_spc['text_pt'] = df_pt_spc['text_pt'].apply(lambda x: ' '.join([token.lemma_ for token in nlp(x) if not token.is_stop and token.pos_ in 
                                                          ['NOUN', 'VERB','ADJ', 'AUX', 'PROP']]))
# Separando variáveis de entrada (X) e saída (y) - spaCy
X_pt_spc = df_pt_spc['text_pt']
y_pt_spc = df_pt_spc['y']

# Convertendo as sentenças em vetores TF-IDF usando o TfidfVectorizer:
vectorizer = TfidfVectorizer()
X_tfidf_pt_spc = vectorizer.fit_transform(X_pt_spc)

# Divisão treino e teste - spaCy
def split_data(X, y, split_point=-400):
    X_train = X[:split_point]
    y_train = y[:split_point]
    X_test = X[split_point:]
    y_test = y[split_point:]
    return X_train, y_train, X_test, y_test

X_train_pt_spc, y_train_pt_spc, X_test_pt_spc, y_test_pt_spc = split_data(X_tfidf_pt_spc, y_pt_spc)

def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    model_rf = (RandomForestClassifier(n_estimators=400, 
                                    min_samples_split= 2, 
                                    min_samples_leaf= 1, 
                                    max_features= 10, 
                                    max_depth = 10,
                                    random_state = 122,
                                    class_weight='balanced'))
    model_rf.fit(X_train, y_train)
    y_pred = model_rf.predict(X_test)
    
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    return model_rf, f1, precision, recall, y_pred

# Treinamento e teste do modelo - spaCy
model_rf_spc, f1_rf_spc, precision_rf_spc, recall_rf_spc, y_pred_rf_spc = train_and_evaluate_model(X_train_pt_spc, y_train_pt_spc, X_test_pt_spc, y_test_pt_spc)

data = {
    'F1 Score': [f1_rf_spc],
    'Precisão': [precision_rf_spc],
    'Recall': [recall_rf_spc]
}

df_resultados = pd.DataFrame(data)


# Matrizes de confusão para as melhores configurações de cada modelo

# Definir os nomes das classes
nomes_classes = ['negative', 'positive', 'neutral']

# Criar a matriz de confusão
conf_mat_rf = confusion_matrix(y_test_pt_spc, y_pred_rf_spc)

# Criar DataFrame para a matriz de confusão com os nomes das classes
conf_mat_rf_df = pd.DataFrame(conf_mat_rf, index=nomes_classes, columns=nomes_classes)


# Plotar a matriz de confusão usando Seaborn
title = "Random Forest"
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_mat_rf_df, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title(title)
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')

plt.show()
    
    
#%% Previsão ML - Achei meio ruim

df_pred = pd.DataFrame(columns=['Title'])

# Removendo acentos e pontuações
df_pred['Title'] = df['Title'].apply(lambda x: unidecode(re.sub(r'[^\w\s]', '', x)))


# Utilizar POS tagging do spaCy - remove todas as palavras que não são substantivos, verbos ou adjetivos 
df_pt_spc_pred = df_pred.copy()

nlp = spacy.load('pt_core_news_sm')
df_pt_spc_pred['Title'] = df_pt_spc_pred['Title'].apply(lambda x: ' '.join([token.lemma_ for token in nlp(x) if not token.is_stop and token.pos_ in 
                                                          ['NOUN', 'VERB','ADJ', 'AUX', 'PROP']]))


X_pred= df_pt_spc_pred['Title']

# Convertendo as sentenças em vetores TF-IDF usando o TfidfVectorizer:
X_tfidf_pt_spc_pred = vectorizer.transform(X_pred)

df['ML'] = model_rf_spc.predict(X_tfidf_pt_spc_pred)


#%% Recomendação


rec1 = df[['Code','Score_LeIA']].groupby(['Code']).mean()

rec2 = df[df['ML'] != 'neutral'].groupby('Code')['ML'].apply(lambda x: x.mode().iloc[0])

rec = pd.merge(rec1, rec2, on='Code', how='left')

rec['Rec_LeIA'] = ' '


for i in range(0,len(rec)):

    if rec['Score_LeIA'].iloc[i] >= 0.05:
        rec['Rec_LeIA'].iloc[i]  = 'positive'
    elif rec['Score_LeIA'].iloc[i]  <= -0.05:
        rec['Rec_LeIA'].iloc[i]  = 'negative'
    else:
        rec['Rec_LeIA'].iloc[i]  = 'neutral'
        
rec['Rec_LeIA'] = rec[rec['Rec_LeIA'] != 'neutral']['Rec_LeIA']
rec = rec.reset_index()

#%% Variação Ação

rec['Var'] = ' '
rec['Var_sent'] = ' '

for i in range(0,len(rec)):

    # Substitua 'PETR4' pelo símbolo da ação desejada
    ticker_symbol = rec['Code'].iloc[i]+'.SA'
    
    # Obtenha dados até o momento atual do dia
    today = date.today().strftime('%Y-%m-%d')
    
    try:
        # Tente obter os dados históricos
        data = yf.Ticker(ticker_symbol).history(period='1d')
    
        if not data.empty:
            # Use o último preço de fechamento disponível ou o preço atual, se disponível
            close_price = data['Close'].iloc[-1] if not pd.isna(data['Close'].iloc[-1]) else data['Open'].iloc[-1]
    
            # Calcule a variação percentual
            variation_percent = ((close_price - data['Open'].iloc[0]) / data['Open'].iloc[0]) * 100
            rec.loc[i, 'Var'] = variation_percent
            
            if variation_percent >= 0:
                rec.loc[i, 'Var_sent']  = 'positive'
            else:
                rec.loc[i, 'Var_sent']  = 'negative'

            
        else:
            print(f'Não há dados disponíveis para {ticker_symbol} no dia {today}.')
    except Exception as e:
        print(f'Falha ao obter dados para {ticker_symbol}. Motivo: {str(e)}')



#%% Correlações

rec_leia = rec[rec['Rec_LeIA'].notna()][['Code','Rec_LeIA','Var_sent']]
rec_ml = rec[rec['ML'].notna()][['Code','ML','Var_sent']]

correlation_matrix = rec['Score_LeIA'].astype(float).corr(rec['Var'].astype(float))

# Matrizes de confusão

# Definir os nomes das classes
nomes_classes_rec = ['negative', 'positive']

# Criar a matriz de confusão
conf_mat_leia = confusion_matrix(rec_leia['Var_sent'], rec_leia['Rec_LeIA'])
conf_mat_ml = confusion_matrix(rec_ml['Var_sent'], rec_ml['ML'])

# Criar DataFrame para a matriz de confusão com os nomes das classes
conf_df_leia = pd.DataFrame(conf_mat_leia, index=nomes_classes_rec, columns=nomes_classes_rec)
conf_df_ml = pd.DataFrame(conf_mat_ml, index=nomes_classes_rec, columns=nomes_classes_rec)

# Plotar a matriz de confusão usando Seaborn
title = "Recomendações via LeIA"
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_df_leia, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title(title)
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')

plt.show()


# Plotar a matriz de confusão usando Seaborn
title = "Recomendações via ML"
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_df_ml, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title(title)
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')

plt.show()

f1_leia = f1_score(rec_leia['Var_sent'], rec_leia['Rec_LeIA'], average='weighted')
precision_leia = precision_score(rec_leia['Var_sent'], rec_leia['Rec_LeIA'], average='weighted')
recall_leia = recall_score(rec_leia['Var_sent'], rec_leia['Rec_LeIA'], average='weighted')

rec_leia_pos = rec_leia.copy()
rec_leia_pos['Rec_LeIA'] = 'positive'

f1_leia_pos = f1_score(rec_leia_pos['Var_sent'], rec_leia_pos['Rec_LeIA'], average='weighted')
precision_leia_pos = precision_score(rec_leia_pos['Var_sent'], rec_leia_pos['Rec_LeIA'], average='weighted')
recall_leia_pos = recall_score(rec_leia_pos['Var_sent'], rec_leia_pos['Rec_LeIA'], average='weighted')

f1_ml = f1_score(rec_ml['Var_sent'], rec_ml['ML'], average='weighted')
precision_ml = precision_score(rec_ml['Var_sent'], rec_ml['ML'], average='weighted')
recall_ml = recall_score(rec_ml['Var_sent'], rec_ml['ML'], average='weighted')


rec_ml_pos = rec_ml.copy()
rec_ml_pos['ML'] = 'positive'

f1_ml_pos = f1_score(rec_ml_pos['Var_sent'], rec_ml_pos['ML'], average='weighted')
precision_ml_pos = precision_score(rec_ml_pos['Var_sent'], rec_ml_pos['ML'], average='weighted')
recall_ml_pos = recall_score(rec_ml_pos['Var_sent'], rec_ml_pos['ML'], average='weighted')


data_fim = {
    'Modelo': ['LeIA','LeIA POS','ML','ML POS'],
    'F1 Score': [f1_leia, f1_leia_pos, f1_ml, f1_ml_pos],
    'Precisão': [precision_leia, precision_leia_pos, precision_ml, precision_ml_pos],
    'Recall': [recall_leia, recall_leia_pos, recall_ml, recall_ml_pos]
}

df_data_fim = pd.DataFrame(data_fim)