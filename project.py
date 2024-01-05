#%% Imports
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pylab as plt

import re
import spacy
import datetime
import pathlib
import glob
import os

from selenium import webdriver
from selenium.webdriver.common.by import By
from time import sleep
from pygooglenews import GoogleNews
from leia import SentimentIntensityAnalyzer
from datetime import date
from unidecode import unidecode
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


#%%Presets

#Dados das ações
today = (date.today()-datetime.timedelta(days=0)).strftime('%Y-%m-%d')

#Notícias
end = date.today()-datetime.timedelta(days=0)
start = end-datetime.timedelta(days=1)

#True ou False para se o arquivo da carteira já foi baixado anteriomente
download = False

#%% Busca da carteira que compõe o índice desejado no dia
def busca_carteira_teorica(indice, espera=8):
    
    #Inicialização do chromedriver
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    wd = webdriver.Chrome(options=options)
    
    #Downlaod do arquivo CSV
    url = f'https://sistemaswebb3-listados.b3.com.br/indexPage/day/{indice.upper()}?language=pt-br'  
    wd.get(url)
    sleep(espera)
    wd.find_element('id','segment').send_keys("Setor de Atuação")
    sleep(espera)
    wd.find_element(By.LINK_TEXT,"Download").click()
    sleep(espera)

    return 

if download:
    #Escolha do índice e do tempo de espera
    busca_carteira_teorica('ibov',5)

#Ler arquivo que foi baixado
list_of_files = glob.glob(str(pathlib.Path.home())+'/Downloads/*.csv')
latest_file = max(list_of_files, key=os.path.getctime)
carteira = pd.read_csv(latest_file, sep=';', encoding='ISO-8859-1',skipfooter=2, engine='python', thousands='.', decimal=',', header=1, index_col=False)

#Criar dataframe com os tickers da carteira
tickers = carteira['Código']

#%% Buscar notícias
            
# Inicialização e configuração
gn = GoogleNews(lang='pt', country='BR')

# Criação do dataframe
df_list = []

# Loop para buscar notícias que contêm os tickers desejados nas datas desejadas
for ticker in tickers:
    # Eliminar resultados com "varzea" para evitar links indesejados
    search = gn.search(f'"{ticker}" -varzea', from_=start.strftime('%Y-%m-%d'), to_=end.strftime('%Y-%m-%d'))
    
    # Create a DataFrame for the current ticker and append it to df_list
    ticker_data = [[ticker, item.title, None, None] for item in search['entries'] if ticker in item.title]
    df_ticker = pd.DataFrame(ticker_data, columns=["Code", "Title", "Score_LeIA", "LeIA"])
    df_list.append(df_ticker)

# Concatenate the list of DataFrames into a single DataFrame
df = pd.concat(df_list, ignore_index=True)
            
#%% Sentiment Analysis - LeIA (https://github.com/rafjaa/LeIA)

#Inicialização
sia = SentimentIntensityAnalyzer()

#Cálculo do score com base no título da notícia
for i in range(0,len(df)):
    scores = sia.polarity_scores(df['Title'].iloc[i])
    
    #Definição de sentimento positivo, negativo ou neutro (https://akladyous.medium.com/sentiment-analysis-using-vader-c56bcffe6f24)
    if scores['compound'] >= 0.05:
        sentiment = 'positive'
    elif scores['compound'] <= -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
        
    df['Score_LeIA'].iloc[i] = scores['compound']
    df['LeIA'].iloc[i] = sentiment
       
    
#%% Criação e teste do modelo ML (a partir de um trabalho no Kaggle: https://www.kaggle.com/code/zeneto11/ml-nlp/notebook - spaCy Random Forest foi o melhor caso)
    
#Dataset do Kaggle (https://www.kaggle.com/datasets/mateuspicanco/financial-phrase-bank-portuguese-translation)
df_sent = pd.read_csv('financial_phrase_bank_pt_br.csv')

#Dividir dataframe em dados de teste e treinamento
df_train = df_sent[:-400]
df_test = df_sent[-400:]

#Resampling para balancear classes
df_negative = df_train[df_train['y'] == 'negative']
df_neutral = df_train[df_train['y'] == 'neutral'].sample(n=len(df_negative) // 3, random_state=42)
df_positive = df_train[df_train['y'] == 'positive'].sample(n=len(df_negative) // 3, random_state=42)

#Concatenar os DataFrames resampleados
df_balanced = pd.concat([df_negative, df_neutral, df_positive])

#Juntar com as ocorrências preservadas de teste
df_con = pd.concat([df_balanced, df_test])

#Dataset somente com as notícias em português
df_pt = df_con.drop('text', axis=1)

#Removeracentos e pontuações
df_pt['text_pt'] = df_pt['text_pt'].apply(lambda x: unidecode(re.sub(r'[^\w\s]', '', x)))

#Utilizar POS tagging do spaCy: remove todas as palavras que não são substantivos, verbos ou adjetivos 
df_pt_spc = df_pt.copy()
nlp = spacy.load('pt_core_news_sm')
df_pt_spc['text_pt'] = df_pt_spc['text_pt'].apply(lambda x: ' '.join([token.lemma_ for token in nlp(x) if not token.is_stop and token.pos_ in 
                                                          ['NOUN', 'VERB','ADJ', 'AUX', 'PROP']]))
#Separar variáveis de entrada (X) e saída (y)
X_pt_spc = df_pt_spc['text_pt']
y_pt_spc = df_pt_spc['y']

#Converter as sentenças em vetores TF-IDF usando o TfidfVectorizer
vectorizer = TfidfVectorizer()
X_tfidf_pt_spc = vectorizer.fit_transform(X_pt_spc)

#Divisão treino e teste
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

#Treinamento e teste do modelo
model_rf_spc, f1_rf_spc, precision_rf_spc, recall_rf_spc, y_pred_rf_spc = train_and_evaluate_model(X_train_pt_spc, y_train_pt_spc, X_test_pt_spc, y_test_pt_spc)

#Display dos resultados
data = {'F1 Score': [f1_rf_spc],
        'Precisão': [precision_rf_spc],
        'Recall': [recall_rf_spc]}
df_resultados = pd.DataFrame(data)


#Definir os nomes das classes
nomes_classes = ['negative', 'positive', 'neutral']

#Criar a matriz de confusão
conf_mat_rf = confusion_matrix(y_test_pt_spc, y_pred_rf_spc)

#Criar DataFrame para a matriz de confusão com os nomes das classes
conf_mat_rf_df = pd.DataFrame(conf_mat_rf, index=nomes_classes, columns=nomes_classes)

#Plotar a matriz de confusão usando Seaborn
title = "Modelo com Random Forest"
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_mat_rf_df, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title(title)
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
plt.show()
    
#%% Previsão ML

#Criação do dataframe com os títulos para previsão
df_pred = pd.DataFrame(columns=['Title'])

#Remover acentos e pontuações
df_pred['Title'] = df['Title'].apply(lambda x: unidecode(re.sub(r'[^\w\s]', '', x)))

#Utilizar POS tagging do spaCy: remove todas as palavras que não são substantivos, verbos ou adjetivos 
df_pt_spc_pred = df_pred.copy()
nlp = spacy.load('pt_core_news_sm')
df_pt_spc_pred['Title'] = df_pt_spc_pred['Title'].apply(lambda x: ' '.join([token.lemma_ for token in nlp(x) if not token.is_stop and token.pos_ in 
                                                          ['NOUN', 'VERB','ADJ', 'AUX', 'PROP']]))
#Variável de teste
X_pred= df_pt_spc_pred['Title']

#Converter as sentenças em vetores TF-IDF usando o TfidfVectorizer:
X_tfidf_pt_spc_pred = vectorizer.transform(X_pred)

#Previsão do modelo
df['ML'] = model_rf_spc.predict(X_tfidf_pt_spc_pred)


#%% Recomendações: verificar ações que tendem a subir ou cair

#Fazer uma média dos scores de todas as notícias com determinado ticker para LeIA
rec1 = df[['Code','Score_LeIA']].groupby(['Code']).mean()

#Remover sentimentos neutros para ML
rec2 = df[df['ML'] != 'neutral'].groupby('Code')['ML'].apply(lambda x: x.mode().iloc[0])

#Juntar os dataframes
rec = pd.merge(rec1, rec2, on='Code', how='left')

#Criar coluna para recomendação LeIA
rec['Rec_LeIA'] = ' '


#Loop para recomendações LeIA com base no threshold de +-0.05
for i in range(0,len(rec)):

    if rec['Score_LeIA'].iloc[i] >= 0.05:
        rec['Rec_LeIA'].iloc[i]  = 'positive'
    elif rec['Score_LeIA'].iloc[i]  <= -0.05:
        rec['Rec_LeIA'].iloc[i]  = 'negative'
    else:
        rec['Rec_LeIA'].iloc[i]  = 'neutral'

#Remover sentimentos neutros para LeIA
rec['Rec_LeIA'] = rec[rec['Rec_LeIA'] != 'neutral']['Rec_LeIA']

#Resetar o índice do dataframe
rec = rec.reset_index()

#%% Obter variação da ação no dia

#Criar coluna para variações
rec['Var'] = ' '
rec['Var_sent'] = ' '

#Loop para os tickers desejados
for i in range(0,len(rec)):
    ticker_symbol = rec['Code'].iloc[i]+'.SA'
    
    
    try:
        #Tentar obter os dados históricos
        data = yf.Ticker(ticker_symbol).history(period='1d')
    
        if not data.empty:
            #Usar o último preço de fechamento disponível ou o preço atual, se disponível
            close_price = data['Close'].iloc[-1] if not pd.isna(data['Close'].iloc[-1]) else data['Open'].iloc[-1]
    
            #Calcular a variação percentual
            variation_percent = ((close_price - data['Open'].iloc[0]) / data['Open'].iloc[0]) * 100
            rec.loc[i, 'Var'] = variation_percent
            
            #Determinar se a variação foi positiva ou negativa
            if variation_percent >= 0:
                rec.loc[i, 'Var_sent']  = 'positive'
            else:
                rec.loc[i, 'Var_sent']  = 'negative'
           
        else:
            print(f'Não há dados disponíveis para {ticker_symbol} no dia {today}.')
    except Exception as e:
        print(f'Falha ao obter dados para {ticker_symbol}. Motivo: {str(e)}')



#%% Correlação e matrizes de confusão

#Criação dos dataframes de recomendação separados
rec_leia = rec[rec['Rec_LeIA'].notna()][['Code','Rec_LeIA','Var_sent']]
rec_ml = rec[rec['ML'].notna()][['Code','ML','Var_sent']]

#Matriz de correlação para scores LeIA e porcentagem de variação
correlation_matrix = rec['Score_LeIA'].astype(float).corr(rec['Var'].astype(float))


#Definir os nomes das classes
nomes_classes_rec = ['negative', 'positive']

#Criar as matrizez de confusão
conf_mat_leia = confusion_matrix(rec_leia['Var_sent'], rec_leia['Rec_LeIA'])
conf_mat_ml = confusion_matrix(rec_ml['Var_sent'], rec_ml['ML'])

#Criar dataframe para a matriz de confusão com os nomes das classes
conf_df_leia = pd.DataFrame(conf_mat_leia, index=nomes_classes_rec, columns=nomes_classes_rec)
conf_df_ml = pd.DataFrame(conf_mat_ml, index=nomes_classes_rec, columns=nomes_classes_rec)

#Plotar a matriz de confusão LeIA usando Seaborn
title = "Recomendações via LeIA"
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_df_leia, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title(title)
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
plt.show()


# Plotar a matriz de confusão ML usando Seaborn
title = "Recomendações via ML"
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_df_ml, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title(title)
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
plt.show()


#Cálculo dos indicadores 
#Os dados POS são uma suposição para se considerassemos que todas as variações como positivas (base de comparação_)
#LeIA e ML têm quantidades diferentes devido à exclusão de neutros
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

#Display dos resultados
data_fim = {'Modelo': ['LeIA','LeIA POS','ML','ML POS'],
            'F1 Score': [f1_leia, f1_leia_pos, f1_ml, f1_ml_pos],
            'Precisão': [precision_leia, precision_leia_pos, precision_ml, precision_ml_pos],
            'Recall': [recall_leia, recall_leia_pos, recall_ml, recall_ml_pos]}
df_data_fim = pd.DataFrame(data_fim)