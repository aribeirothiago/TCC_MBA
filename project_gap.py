#%% Imports
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pylab as plt
import numpy as np

import re
import spacy
import pathlib
import glob
import os

from selenium import webdriver
from selenium.webdriver.common.by import By
from time import sleep
from pygooglenews import GoogleNews
from leia import SentimentIntensityAnalyzer
from datetime import date, time, datetime, timedelta
from unidecode import unidecode
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


#%% Presets

#Simulação de dinheiro investido
investimento = 1000

#Thresholds
sup_leia = 0.3
inf_leia = -0.5
pos_ml = 0.337
neg_ml = 0.343

#Delta para teste de dias anteriores
delta = 22

#True ou False para baixar o arquivo da carteira
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

#Data das notícias
end =  date.today()-timedelta(days=delta)
endtime = datetime.combine(date.today()-timedelta(days=delta),time(10, 0))
start = end-timedelta(days=1)
            
#Inicialização e configuração
gn = GoogleNews(lang='pt', country='BR')

#Criação da lista do dataframe
df_list = []

#Loop para buscar notícias que contêm os tickers desejados nas datas desejadas
for ticker in tickers:
    # Eliminar resultados com "varzea" para evitar links indesejados
    search = gn.search(f'"{ticker}" -varzea', from_=start.strftime('%Y-%m-%d'), to_=(end+timedelta(days=1)).strftime('%Y-%m-%d'))
    
    #Criar um dataframe para o ticker atual e juntar com a lista
    ticker_data = [[ticker, item.title, item.published, None] for item in search['entries'] if ticker in item.title]
    df_ticker = pd.DataFrame(ticker_data, columns=["Code", "Title", "Datetime","Score_LeIA"])
    df_list.append(df_ticker)

#Concatenar e transformar em dataframe
df = pd.concat(df_list, ignore_index=True)
df = df[pd.to_datetime(df['Datetime'])<=endtime]
df = df.reset_index()
            
#%% Sentiment Analysis - LeIA (https://github.com/rafjaa/LeIA)

#Inicialização
sia = SentimentIntensityAnalyzer()

#Cálculo do score com base no título da notícia
for i in range(0,len(df)):
    scores = sia.polarity_scores(df.loc[i,'Title'])      
    df.loc[i,'Score_LeIA'] = scores['compound']
       
    
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
    
    metrics = classification_report(y_test, y_pred, output_dict=True)
    return model_rf, metrics, y_pred

#Treinamento e teste do modelo
model_rf_spc, metrics_spc, y_pred_rf_spc = train_and_evaluate_model(X_train_pt_spc, y_train_pt_spc, X_test_pt_spc, y_test_pt_spc)

df_resultados = pd.DataFrame(metrics_spc)
df_resultados = df_resultados.transpose()


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
df['ML_proba_neg'] = model_rf_spc.predict_proba(X_tfidf_pt_spc_pred)[:,0]
df['ML_proba_neu'] = model_rf_spc.predict_proba(X_tfidf_pt_spc_pred)[:,1]
df['ML_proba_pos'] = model_rf_spc.predict_proba(X_tfidf_pt_spc_pred)[:,2]

#Calcular a classe prevista com base nas probabilidades
df['ML_predicted_class'] = np.argmax(df[['ML_proba_neg', 'ML_proba_neu', 'ML_proba_pos']].values, axis=1)

#%% Recomendações: verificar ações que tendem a subir ou cair

#Fazer uma média dos scores de todas as notícias com determinado ticker para LeIA
rec1 = df[['Code','Score_LeIA']].groupby(['Code']).mean()

#Fazer uma média dos scores de todas as notícias com determinado ticker para ML
rec2 = df[['Code','ML_proba_neg','ML_proba_neu','ML_proba_pos','ML_predicted_class']].groupby('Code').mean()

#Juntar os dataframes
rec = pd.merge(rec1, rec2, on='Code', how='outer')

#Resetar o índice do dataframe
rec = rec.reset_index()

rec['Rec_LeIA'] = ' '
#Loop para recomendações LeIA
for i in range(0,len(rec)):

    if rec.loc[i,'Score_LeIA'] >= sup_leia:
        rec.loc[i,'Rec_LeIA']  = 'positive'
    elif rec.loc[i,'Score_LeIA']  <= inf_leia:
        rec.loc[i,'Rec_LeIA'] = 'negative'
    else:
        rec.loc[i,'Rec_LeIA'] = 'neutral'
        
rec['ML']=''
#Loop para recomendações ML
for i in range(0,len(rec)):
    
    if rec.loc[i,'ML_proba_neu'] < rec.loc[i,'ML_proba_neg'] or rec.loc[i,'ML_proba_neu'] < rec.loc[i,'ML_proba_pos']:       
        if rec.loc[i,'ML_proba_neg'] > neg_ml and rec.loc[i,'ML_proba_neg'] > rec.loc[i,'ML_proba_pos']:
            rec.loc[i,'ML'] = 'negative'
        elif rec.loc[i,'ML_proba_pos'] > pos_ml:
            rec.loc[i,'ML'] = 'positive'
        else:
            rec.loc[i,'ML'] = 'neutral'
    else:
        rec.loc[i,'ML'] = 'neutral'


#%% Obter variação da ação no dia

#Criar coluna para variações
rec['fechou'] = ' '
rec['pfechar'] = ' '
rec['previsao_ml'] = ' '
rec['previsao_LeIA'] = ' '
rec['open'] = ' '
rec['close_previous'] = ' '
rec['close_today'] = ' '

#Loop para os tickers desejados
for i in range(0,len(rec)):
    ticker_symbol = rec.loc[i,'Code']+'.SA'
    
    #Obter dados intradiários (1 minuto) para o dia atual
    data_hoje = yf.download(ticker_symbol, start=end,end=end+timedelta(days=1), interval='1m', progress=False)
    data_hoje_close = yf.download(ticker_symbol, start=end,end=end+timedelta(days=1), progress=False)
    data_ontem = yf.download(ticker_symbol, start=start, end=end, progress=False)
    
    #Extrair preços de abertura e fechamento
    opening_prices = round(data_hoje['Open'],2)
    closing_price = round(data_ontem['Close'][0],2)
    closing_price_today = round(data_hoje_close['Close'][0],2)
    
    rec.loc[i,'open'] = opening_prices[0]
    rec.loc[i,'close_previous'] = closing_price
    rec.loc[i,'close_today'] = closing_price_today
    
    #Verificar se o "gap" foi fechado em algum momento
    gap_closed_at_some_point = closing_price in opening_prices.values
    
    #Exibir o resultado
    if gap_closed_at_some_point:
        rec.loc[i,'fechou'] = 'Y'
    else:
        rec.loc[i,'fechou'] = 'N'
        
    if opening_prices.iloc[0] > closing_price:
        rec.loc[i,'pfechar'] = 'negative'
    elif opening_prices.iloc[0] < closing_price:
        rec.loc[i,'pfechar'] = 'positive'
    else: 
        rec.loc[i,'pfechar'] = 'null'
    
    if rec.loc[i,'ML'] not in ('positive','negative'):
        pass
    elif rec.loc[i,'pfechar'] == rec.loc[i,'ML']:
        rec.loc[i,'previsao_ml'] = 'Y'
    else:
        rec.loc[i,'previsao_ml'] = 'N'
        
    if rec.loc[i,'Rec_LeIA'] not in ('positive','negative'):
        pass
    elif rec.loc[i,'pfechar'] == rec.loc[i,'Rec_LeIA']:
        rec.loc[i,'previsao_LeIA'] = 'Y'
    else:
        rec.loc[i,'previsao_LeIA'] = 'N'
        
rec=rec[rec['pfechar']!='null']
rec=rec.reset_index()        
        
#%% Correlação e matrizes de confusão

#Criação dos dataframes de recomendação separados
rec_leia = rec[rec['Rec_LeIA'] != 'neutral'][['Code','previsao_LeIA','fechou']]
rec_ml = rec[rec['ML'] != 'neutral'][['Code','previsao_ml','fechou']]

#Definir os nomes das classes
nomes_classes_rec = ['negative', 'positive']

#Criar as matrizez de confusão
conf_mat_leia = confusion_matrix(rec_leia['fechou'], rec_leia['previsao_LeIA'])
conf_mat_ml = confusion_matrix(rec_ml['fechou'], rec_ml['previsao_ml'])

#Criar dataframe para a matriz de confusão com os nomes das classes
conf_df_leia = pd.DataFrame(conf_mat_leia)
conf_df_ml = pd.DataFrame(conf_mat_ml)

#Plotar a matriz de confusão LeIA usando Seaborn
title = "Recomendações via LeIA"
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_df_leia, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title(title)
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
plt.show()

#Plotar a matriz de confusão ML usando Seaborn
title = "Recomendações via ML"
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_df_ml, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title(title)
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
plt.show()

#Cálculo dos indicadores 
#Os dados POS são uma suposição para se considerassemos que todas as ações fecharão o GAP (base de comparação)
#LeIA e ML podem ter quantidades diferentes devido à exclusão de neutros
metrics_leia_dict = classification_report(rec_leia['fechou'], rec_leia['previsao_LeIA'], output_dict=True, zero_division=1)
metrics_leia = pd.DataFrame(metrics_leia_dict)
metrics_leia = metrics_leia.transpose()

rec_leia_pos = rec_leia.copy()
rec_leia_pos['previsao_LeIA'] = 'Y'

metrics_leia_pos_dict = classification_report(rec_leia_pos['fechou'], rec_leia_pos['previsao_LeIA'], output_dict=True, zero_division=1)
metrics_leia_pos = pd.DataFrame(metrics_leia_pos_dict)
metrics_leia_pos = metrics_leia_pos.transpose()

metrics_ml_dict = classification_report(rec_ml['fechou'], rec_ml['previsao_ml'], output_dict=True, zero_division=1)
metrics_ml = pd.DataFrame(metrics_ml_dict)
metrics_ml = metrics_ml.transpose()

rec_ml_pos = rec_ml.copy()
rec_ml_pos['previsao_ml'] = 'Y'

metrics_ml_pos_dict = classification_report(rec_ml_pos['fechou'], rec_ml_pos['previsao_ml'], output_dict=True, zero_division=1)
metrics_ml_pos = pd.DataFrame(metrics_ml_pos_dict)
metrics_ml_pos = metrics_ml_pos.transpose()

#Display dos resultados
data_fim = {'Modelo': ['LeIA','LeIA POS','ML','ML POS'],
            'Precisão': [metrics_leia['precision']['weighted avg'], metrics_leia_pos['precision']['weighted avg'], metrics_ml['precision']['weighted avg'], metrics_ml_pos['precision']['weighted avg']],
            'Suport': [metrics_leia['support']['weighted avg'], metrics_leia_pos['support']['weighted avg'], metrics_ml['support']['weighted avg'], metrics_ml_pos['support']['weighted avg']]}
df_data_fim = pd.DataFrame(data_fim)
print(df_data_fim)


#%% Simulação de compra

rec['comprar_leia'] = 0
rec['comprar_ml'] = 0

qtd_leia = 0
qtd_ml = 0

for i in range(0,len(rec)):    
    #Se pra fechar o gap precisamos de uma variação positiva e a previsão é que o gap será fechado
    if rec.loc[i,'pfechar'] == 'positive' and rec.loc[i,'previsao_LeIA'] == 'Y':
        qtd_leia = qtd_leia + 1
        rec.loc[i,'comprar_leia'] = 1
    if rec.loc[i,'pfechar'] == 'positive' and rec.loc[i,'previsao_ml'] == 'Y':
        qtd_ml = qtd_ml + 1
        rec.loc[i,'comprar_ml'] = 1
        
compra_leia = rec[rec['comprar_leia'] == 1]
compra_ml = rec[rec['comprar_ml'] == 1]

compra_leia = compra_leia.reset_index()
compra_ml = compra_ml.reset_index()

compra_leia['var'] = ''
compra_ml['var'] = ''
compra_leia['lucro'] = ''
compra_ml['lucro'] = ''

#variação e lucro comprando ações que precisam subir para fechar gap e recomendadas por leia
for i in range(0,len(compra_leia)):
    if compra_leia.loc[i,'fechou'] == 'Y':
        compra_leia.loc[i,'var'] = compra_leia.loc[i,'close_previous']/compra_leia.loc[i,'open']-1
        compra_leia.loc[i,'lucro'] = investimento/qtd_leia*compra_leia.loc[i,'var']
    else: 
        compra_leia.loc[i,'var'] = compra_leia.loc[i,'close_today']/compra_leia.loc[i,'open']-1
        compra_leia.loc[i,'lucro'] = investimento/qtd_leia*compra_leia.loc[i,'var']
        
 #variação e lucro comprando ações que precisam subir para fechar gap e recomendadas por ML       
for i in range(0,len(compra_ml)):
    if compra_ml.loc[i,'fechou'] == 'Y':
        compra_ml.loc[i,'var'] = compra_ml.loc[i,'close_previous']/compra_ml.loc[i,'open']-1
        compra_ml.loc[i,'lucro'] = investimento/qtd_ml*compra_ml.loc[i,'var']
    else: 
        compra_ml.loc[i,'var'] = compra_ml.loc[i,'close_today']/compra_ml.loc[i,'open']-1
        compra_ml.loc[i,'lucro'] = investimento/qtd_ml*compra_ml.loc[i,'var']
        
        
lucro_leia = compra_leia['lucro'].sum()
lucro_ml = compra_ml['lucro'].sum()

print('Lucro Leia: ' + str(lucro_leia) + ' / Lucro ML: '+ str(lucro_ml))
