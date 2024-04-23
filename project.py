#%% IMPORTS
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
from datetime import time, datetime, timedelta
from unidecode import unidecode
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

#%% ENTRADAS

# Simulação de dinheiro investido
investimento = 100
investimento_acumulado_LEIA = investimento
investimento_acumulado_RF = investimento
investimento_acumulado_NB = investimento
investimento_acumulado_GAP = investimento

# Thresholds
sup_LEIA = 0.35
inf_LEIA = -0.55
pos_RF = 0.34
neg_RF = 0.345
pos_NB = 0.34
neg_NB = 0.345

# Perda aceitável
pa = 0.01

# True ou False para baixar o arquivo da carteira 
download = False

# Datas de início e fim (o primeiro dia não é considerado para previsões)
start_date = '2024-03-27'
end_date = '2024-03-28'

# Intervalo para busca de preço das ações (Opções: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
intervalo1='1m'
intervalo2='5m'

# Feriados
feriados = ['2023-02-20','2023-02-21','2023-04-07','2023-04-21','2023-05-01','2023-06-08','2023-09-07','2023-10-12','2023-11-02','2023-11-15','2023-12-25','2023-12-29','2024-01-01', '2024-02-12','2024-02-13','2024-03-29','2024-05-01','2024-05-30','2024-11-15','2024-12-24','2024-12-25','2024-12-31']

#%% DIAS ÚTEIS

# Cria um array de datas entre a data de início e fim
date_range = pd.date_range(start=start_date, end=end_date)

# Usa a função isin() para verificar se cada data é um dia útil (segunda a sexta)
business_days = date_range[date_range.to_series().dt.dayofweek < 5]
business_days = business_days[~business_days.isin(feriados)]

#%% DATASETS PARA SIMULAÇÃO DE LUCRO

compra_LEIA_total = pd.DataFrame(columns=['Dia','Acerto','Quantidade','Lucro','Lucro Acumulado'])
compra_RF_total = pd.DataFrame(columns=['Dia','Acerto','Quantidade','Lucro','Lucro Acumulado'])
compra_NB_total = pd.DataFrame(columns=['Dia','Acerto','Quantidade','Lucro','Lucro Acumulado'])
compra_GAP_total = pd.DataFrame(columns=['Dia','Acerto','Quantidade','Lucro','Lucro Acumulado'])
    
#%% FUNÇÃO PARA SAÍDA EM ARQUIVO E CONSOLE

def pw(message):
    #Console
    print(message)
    
    #Arquivo
    with open('./outputs/output.txt', 'a') as file:
        file.write(str(message) + '\n')

#%% CRIAÇÃO E TESTE DOS MODELOS ML (FONTE: https://www.kaggle.com/code/zeneto11/ml-nlp/notebook)
     
# Dataset (Fonte: https://www.kaggle.com/datasets/mateuspicanco/financial-phrase-bank-portuguese-translation)
df_sent = pd.read_csv('financial_phrase_bank_pt_br.csv')

# Dividir dataframe em dados de teste e treinamento
df_train = df_sent[:-400]
df_test = df_sent[-400:]

# Resampling para balancear classes
df_negative = df_train[df_train['y'] == 'negative']
df_neutral = df_train[df_train['y'] == 'neutral'].sample(n=len(df_negative) // 3, random_state=42)
df_positive = df_train[df_train['y'] == 'positive'].sample(n=len(df_negative) // 3, random_state=42)

# Concatenar os DataFrames resampleados
df_balanced = pd.concat([df_negative, df_neutral, df_positive])

# Juntar com as ocorrências preservadas de teste
df_con = pd.concat([df_balanced, df_test])

# Dataset somente com as notícias em português
df_pt = df_con.drop('text', axis=1)

# Remover acentos e pontuações
df_pt['text_pt'] = df_pt['text_pt'].apply(lambda x: unidecode(re.sub(r'[^\w\s]', '', x)))

# Utilizar POS tagging do spaCy: remove todas as palavras que não são substantivos, verbos ou adjetivos 
df_pt_spc = df_pt.copy()
nlp = spacy.load('pt_core_news_sm')
df_pt_spc['text_pt'] = df_pt_spc['text_pt'].apply(lambda x: ' '.join([token.lemma_ for token in nlp(x) if not token.is_stop and token.pos_ in 
                                                          ['NOUN', 'VERB','ADJ', 'AUX', 'PROP']]))
# Separar variáveis de entrada (X) e saída (y)
X_pt_spc = df_pt_spc['text_pt']
y_pt_spc = df_pt_spc['y']

# Converter as sentenças em vetores TF-IDF usando o TfidfVectorizer
vectorizer = TfidfVectorizer()
X_tfidf_pt_spc = vectorizer.fit_transform(X_pt_spc)

# Divisão treino e teste
def split_data(X, y, split_point=-400):
    X_train = X[:split_point]
    y_train = y[:split_point]
    X_test = X[split_point:]
    y_test = y[split_point:]
    return X_train, y_train, X_test, y_test

X_train_pt_spc, y_train_pt_spc, X_test_pt_spc, y_test_pt_spc = split_data(X_tfidf_pt_spc, y_pt_spc)

def train_and_evaluate_model_NB(X_train, y_train, X_test, y_test):
    model_NB = MultinomialNB(alpha = 0.1)
    model_NB.fit(X_train, y_train)
    y_pred = model_NB.predict(X_test)
    
    metrics = classification_report(y_test, y_pred, output_dict=True)
    return model_NB, metrics, y_pred

# Treinamento e teste do modelo
model_NB_spc, metrics_NB_spc, y_pred_NB_spc = train_and_evaluate_model_NB(X_train_pt_spc, y_train_pt_spc, X_test_pt_spc, y_test_pt_spc)

def train_and_evaluate_model_RF(X_train, y_train, X_test, y_test):
    model_RF = (RandomForestClassifier(n_estimators=400, 
                                    min_samples_split= 2, 
                                    min_samples_leaf= 1, 
                                    max_features= 10, 
                                    max_depth = 10,
                                    random_state = 122,
                                    class_weight='balanced'))
    model_RF.fit(X_train, y_train)
    y_pred = model_RF.predict(X_test)
    
    metrics = classification_report(y_test, y_pred, output_dict=True)
    return model_RF, metrics, y_pred

# Treinamento e teste do modelo
model_RF_spc, metrics_RF_spc, y_pred_RF_spc = train_and_evaluate_model_RF(X_train_pt_spc, y_train_pt_spc, X_test_pt_spc, y_test_pt_spc)

df_resultados_RF = pd.DataFrame(metrics_RF_spc).transpose()
df_resultados_NB = pd.DataFrame(metrics_NB_spc).transpose()

# Definir os nomes das classes
nomes_classes = ['negativo', 'positivo', 'neutro']

# Criar as matrizes de confusão
conf_mat_RF = confusion_matrix(y_test_pt_spc, y_pred_RF_spc)
conf_mat_NB = confusion_matrix(y_test_pt_spc, y_pred_NB_spc)

# Criar DataFrame para a matriz de confusão com os nomes das classes
conf_mat_RF_df = pd.DataFrame(conf_mat_RF, index=nomes_classes, columns=nomes_classes)
conf_mat_NB_df = pd.DataFrame(conf_mat_NB, index=nomes_classes, columns=nomes_classes)

# Plotar as matrizes de confusão usando Seaborn
title = "Modelo com Random Forest"
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_mat_RF_df, annot=True, fmt='d', cmap='OrRd', ax=ax)
ax.set_title(title)
ax.set_ylabel('Real')
ax.set_xlabel('Previsto')
plt.show()

title = "Modelo com Naive Bayes"
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_mat_NB_df, annot=True, fmt='d', cmap='OrRd', ax=ax)
ax.set_title(title)
ax.set_ylabel('Real')
ax.set_xlabel('Previsto')
plt.show()

#%% BUSCA DA CARTEIRA QUE COMPÕE O ÍNDICE DESEJADO NO DIA
     
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
    print("Fazendo download do arquivo da carteira...\n")
    #Escolha do índice e do tempo de espera
    busca_carteira_teorica('ibov',5)

# Ler arquivo que foi baixado
list_of_files = glob.glob(str(pathlib.Path.home())+'/Downloads/*.csv')
latest_file = max(list_of_files, key=os.path.getctime)
carteira = pd.read_csv(latest_file, sep=';', encoding='ISO-8859-1',skipfooter=2, engine='python', thousands='.', decimal=',', header=1, index_col=False)

# Criar dataframe com os tickers da carteira
tickers = carteira['Código']

#%% INÍCIO DO LOOP

for k in range(1,len(business_days)):
    
    try:
        
        # Data de hoje
        hj = business_days[k].strftime('%Y-%m-%d')
        
        pw('\n' + hj +': \n')
        
        #%% BUSCAR NOTÍCIAS
        
        # Data das notícias
        end =  business_days[k]
        endtime = datetime.combine(business_days[k],time(10, 0))
        start = end-timedelta(days=1)
                    
        # Inicialização e configuração
        gn = GoogleNews(lang='pt', country='BR')
        
        # Criação da lista do dataframe
        df_list = []
        
        print("Buscando notícias...\n")
        
        # Loop para buscar notícias que contêm os tickers desejados nas datas desejadas
        for ticker in tickers:
            # Eliminar resultados com "varzea" para evitar links indesejados
            search = gn.search(f'"{ticker}" -varzea', from_=start.strftime('%Y-%m-%d'), to_=(end+timedelta(days=1)).strftime('%Y-%m-%d'))
            
            # Criar um dataframe para o ticker atual e juntar com a lista
            ticker_data = [[ticker, item.title, item.published, None] for item in search['entries'] if ticker in item.title]
            df_ticker = pd.DataFrame(ticker_data, columns=["Code", "Title", "Datetime","Score_LEIA"])
            df_list.append(df_ticker)
        
        # Concatenar e transformar em dataframe
        df = pd.concat(df_list, ignore_index=True)
        df = df[pd.to_datetime(df['Datetime'])<=endtime]
        df = df.reset_index(drop=True)
                    
        #%% SENTIMENT ANALYSIS LEIA (Fonte: https://github.com/rafjaa/LEIA)
        
        # Inicialização
        sia = SentimentIntensityAnalyzer()
        
        # Cálculo do score com base no título da notícia
        for i in range(0,len(df)):
            scores = sia.polarity_scores(df.loc[i,'Title'])      
            df.loc[i,'Score_LEIA'] = scores['compound']
            
        #%% PREVISÃO ML
        
        # Criação do dataframe com os títulos para previsão
        df_pred = pd.DataFrame(columns=['Title'])
        
        # Remover acentos e pontuações
        df_pred['Title'] = df['Title'].apply(lambda x: unidecode(re.sub(r'[^\w\s]', '', x)))
        
        # Utilizar POS tagging do spaCy: remove todas as palavras que não são substantivos, verbos ou adjetivos 
        df_pt_spc_pred = df_pred.copy()
        nlp = spacy.load('pt_core_news_sm')
        df_pt_spc_pred['Title'] = df_pt_spc_pred['Title'].apply(lambda x: ' '.join([token.lemma_ for token in nlp(x) if not token.is_stop and token.pos_ in 
                                                                  ['NOUN','VERB','ADJ', 'AUX', 'PROP']]))
        # Variável de teste
        X_pred= df_pt_spc_pred['Title']
        
        # Converter as sentenças em vetores TF-IDF usando o TfidfVectorizer:
        X_tfidf_pt_spc_pred = vectorizer.transform(X_pred)
        
        # Previsão dos modelo
        df['RF_proba_neg'] = model_RF_spc.predict_proba(X_tfidf_pt_spc_pred)[:,0]
        df['RF_proba_neu'] = model_RF_spc.predict_proba(X_tfidf_pt_spc_pred)[:,1]
        df['RF_proba_pos'] = model_RF_spc.predict_proba(X_tfidf_pt_spc_pred)[:,2]
        df['NB_proba_neg'] = model_NB_spc.predict_proba(X_tfidf_pt_spc_pred)[:,0]
        df['NB_proba_neu'] = model_NB_spc.predict_proba(X_tfidf_pt_spc_pred)[:,1]
        df['NB_proba_pos'] = model_NB_spc.predict_proba(X_tfidf_pt_spc_pred)[:,2]
        
        # Calcular as classe prevista com base nas probabilidades
        df['RF_predicted_class'] = np.argmax(df[['RF_proba_neg', 'RF_proba_neu', 'RF_proba_pos']].values, axis=1)
        df['NB_predicted_class'] = np.argmax(df[['NB_proba_neg', 'NB_proba_neu', 'NB_proba_pos']].values, axis=1)
        
        #%% RECOMENDAÇÕES: VERIFICAR AÇÕES QUE TENDEM A SUBIR OU CAIR
        
        # Fazer uma média dos scores de todas as notícias com determinado ticker para LEIA
        rec1 = df[['Code','Score_LEIA']].groupby(['Code']).mean()
        
        # Fazer uma média dos scores de todas as notícias com determinado ticker para RF
        rec2 = df[['Code','RF_proba_neg','RF_proba_neu','RF_proba_pos','RF_predicted_class']].groupby('Code').mean()
        
        # Fazer uma média dos scores de todas as notícias com determinado ticker para RF
        rec3 = df[['Code','NB_proba_neg','NB_proba_neu','NB_proba_pos','NB_predicted_class']].groupby('Code').mean()
        
        # Juntar os dataframes
        rec = pd.merge(rec1, rec2, on='Code', how='outer')
        rec = pd.merge(rec, rec3, on='Code', how='outer')
        
        # Resetar o índice do dataframe
        rec = rec.reset_index()
        
        rec['Rec_LEIA'] = ' '
        # Loop para recomendações LEIA
        for i in range(0,len(rec)):
        
            if rec.loc[i,'Score_LEIA'] >= sup_LEIA:
                rec.loc[i,'Rec_LEIA']  = 'positive'
            elif rec.loc[i,'Score_LEIA']  <= inf_LEIA:
                rec.loc[i,'Rec_LEIA'] = 'negative'
            else:
                rec.loc[i,'Rec_LEIA'] = 'neutral'
                
        rec['RF']=''
        # Loop para recomendações RF
        for i in range(0,len(rec)):
            
            if rec.loc[i,'RF_proba_neu'] < rec.loc[i,'RF_proba_neg'] or rec.loc[i,'RF_proba_neu'] < rec.loc[i,'RF_proba_pos']:       
                if rec.loc[i,'RF_proba_neg'] > neg_RF and rec.loc[i,'RF_proba_neg'] > rec.loc[i,'RF_proba_pos']:
                    rec.loc[i,'RF'] = 'negative'
                elif rec.loc[i,'RF_proba_pos'] > pos_RF:
                    rec.loc[i,'RF'] = 'positive'
                else:
                    rec.loc[i,'RF'] = 'neutral'
            else:
                rec.loc[i,'RF'] = 'neutral'
                
        rec['NB']=''
        # Loop para recomendações NB
        for i in range(0,len(rec)):
            
            if rec.loc[i,'NB_proba_neu'] < rec.loc[i,'NB_proba_neg'] or rec.loc[i,'NB_proba_neu'] < rec.loc[i,'NB_proba_pos']:       
                if rec.loc[i,'NB_proba_neg'] > neg_NB and rec.loc[i,'NB_proba_neg'] > rec.loc[i,'NB_proba_pos']:
                    rec.loc[i,'NB'] = 'negative'
                elif rec.loc[i,'NB_proba_pos'] > pos_NB:
                    rec.loc[i,'NB'] = 'positive'
                else:
                    rec.loc[i,'NB'] = 'neutral'
            else:
                rec.loc[i,'NB'] = 'neutral'
        
        #%% OBTER VARIAÇÃO DA AÇÃO NO DIA
        
        # Criar coluna para variações
        rec['fechou'] = ' '
        rec['pfechar'] = ' '
        rec['previsao_RF'] = ' '
        rec['previsao_NB'] = ' '
        rec['previsao_LEIA'] = ' '
        rec['open'] = ' '
        rec['close_previous'] = ' '
        rec['close_today'] = ' '
        opens = pd.DataFrame()
        
        print("Buscando ações...\n")
        
        # Loop para os tickers desejados
        todrop = np.zeros(len(rec))
        for i in range(0,len(rec)):
            ticker_symbol = rec.loc[i,'Code']+'.SA'
            
            # Obter dados intradiários para o dia atual
            data_hoje = yf.download(ticker_symbol, start=business_days[k],end=business_days[k]+timedelta(days=1), interval=intervalo1, progress=False)
            if data_hoje.empty:
               data_hoje = yf.download(ticker_symbol, start=business_days[k],end=business_days[k]+timedelta(days=1), interval=intervalo2, progress=False)
            data_hoje_close = yf.download(ticker_symbol, start=business_days[k],end=business_days[k]+timedelta(days=1), progress=False)        
            if data_hoje_close.empty:
                data_hoje_close = data_hoje.tail(1)
            data_ontem = yf.download(ticker_symbol, start=business_days[k-1], end=business_days[k-1]+timedelta(days=1), progress=False)
            if data_ontem.empty:
                data_ontem = yf.download(ticker_symbol, start=business_days[k-1], end=business_days[k-1]+timedelta(days=1), interval='1m', progress=False).tail(1)
                
            # Extrair preços de abertura e fechamento
            opening_prices = round(data_hoje['Open'],2)
            closing_price = round(data_ontem['Close'][0],2)
            closing_price_today = round(data_hoje_close['Close'][0],2)
                         
            opens = pd.concat([opens, opening_prices.to_frame().transpose()], ignore_index=True)
            
            rec.loc[i,'open'] = opening_prices[0]
            rec.loc[i,'close_previous'] = closing_price
            rec.loc[i,'close_today'] = closing_price_today
            
            # Se o preços de abertura for muito menor ou maior (2 vezes) que o preço de fechamento, descartar ticker (evita erro causado por bugs na biblioteca)
            if rec.loc[i,'open'] > 2*rec.loc[i,'close_previous'] or 2*rec.loc[i,'open'] < rec.loc[i,'close_previous']:
                todrop[i] = 1
                continue
            
            # Verificar se o "GAP" foi fechado em algum momento
            for j in range(0,len(opening_prices.values)):
                if opening_prices.values[j] >= closing_price:
                    rec.loc[i,'fechou'] = 'Y'
                    break
                else:
                    rec.loc[i,'fechou'] = 'N'
                
            if opening_prices.iloc[0] > closing_price:
                rec.loc[i,'pfechar'] = 'negative'
            elif opening_prices.iloc[0] < closing_price:
                rec.loc[i,'pfechar'] = 'positive'
            else: 
                rec.loc[i,'pfechar'] = 'null'
            
            if rec.loc[i,'RF'] not in ('positive','negative'):
                pass
            elif rec.loc[i,'pfechar'] == rec.loc[i,'RF']:
                rec.loc[i,'previsao_RF'] = 'Y'
            else:
                rec.loc[i,'previsao_RF'] = 'N'
                
                
            if rec.loc[i,'NB'] not in ('positive','negative'):
                pass
            elif rec.loc[i,'pfechar'] == rec.loc[i,'NB']:
                rec.loc[i,'previsao_NB'] = 'Y'
            else:
                rec.loc[i,'previsao_NB'] = 'N'
                    
                
            if rec.loc[i,'Rec_LEIA'] not in ('positive','negative'):
                pass
            elif rec.loc[i,'pfechar'] == rec.loc[i,'Rec_LEIA']:
                rec.loc[i,'previsao_LEIA'] = 'Y'
            else:
                rec.loc[i,'previsao_LEIA'] = 'N'
                
        # Ajuste e limpeza do dataframe    
        rec=pd.concat([rec,opens],axis=1)
        rec=rec[rec['pfechar']!='null'] 
        for i in range(0,len(todrop)):
            if todrop[i] == 1:
                rec.drop(i, inplace=True)
        rec=rec.reset_index(drop=True)      
                
        #%% CORRELAÇÃO E MATRIZES DE CONFUSÃO
        
        # Criação dos dataframes de recomendação separados
        rec_LEIA = rec[rec['Rec_LEIA'] != 'neutral'][['Code','previsao_LEIA','fechou']]
        rec_RF = rec[rec['RF'] != 'neutral'][['Code','previsao_RF','fechou']]
        rec_NB = rec[rec['NB'] != 'neutral'][['Code','previsao_NB','fechou']]
        
        # Definir os nomes das classes
        nomes_classes_rec = ['negative', 'positive']
        
        # Criar as matrizez de confusão
        conf_mat_LEIA = confusion_matrix(rec_LEIA['fechou'], rec_LEIA['previsao_LEIA'])
        conf_mat_RF = confusion_matrix(rec_RF['fechou'], rec_RF['previsao_RF'])
        conf_mat_NB = confusion_matrix(rec_NB['fechou'], rec_NB['previsao_NB'])
        
        # Criar dataframe para a matriz de confusão com os nomes das classes
        conf_df_LEIA = pd.DataFrame(conf_mat_LEIA)
        conf_df_RF = pd.DataFrame(conf_mat_RF)
        conf_df_NB = pd.DataFrame(conf_mat_NB)
        
        try:
            # Plotar a matriz de confusão LEIA usando Seaborn
            title = "Recomendações via LEIA"
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(conf_df_LEIA, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(title)
            ax.set_ylabel('Real')
            ax.set_xlabel('Previsto')
            plt.savefig('./outputs/'+hj+'_calor_LEIA.png')
            plt.show()
        except:
            pass
    
        try:
            # Plotar a matriz de confusão RF usando Seaborn
            title = "Recomendações via RF"
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(conf_df_RF, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(title)
            ax.set_ylabel('Real')
            ax.set_xlabel('Previsto')
            plt.savefig('./outputs/'+hj+'_calor_RF.png')
            plt.show()
        except:
            pass
        
        try:
            # Plotar a matriz de confusão NB usando Seaborn
            title = "Recomendações via NB"
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(conf_df_NB, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(title)
            ax.set_ylabel('Real')
            ax.set_xlabel('Previsto')
            plt.savefig('./outputs/'+hj+'_calor_NB.png')
            plt.show()
        except:
            pass
        
        # Cálculo dos indicadores 
        # Os dados POS são uma suposição para se considerassemos que todas as ações fecharão o GAP (base de comparação)
        # Podem ter quantidades diferentes devido à exclusão de neutros
        metrics_LEIA_dict = classification_report(rec_LEIA['fechou'], rec_LEIA['previsao_LEIA'], output_dict=True, zero_division=1)
        metrics_LEIA = pd.DataFrame(metrics_LEIA_dict)
        metrics_LEIA = metrics_LEIA.transpose()
        
        rec_LEIA_pos = rec_LEIA.copy()
        rec_LEIA_pos['previsao_LEIA'] = 'Y'
        
        metrics_LEIA_pos_dict = classification_report(rec_LEIA_pos['fechou'], rec_LEIA_pos['previsao_LEIA'], output_dict=True, zero_division=1)
        metrics_LEIA_pos = pd.DataFrame(metrics_LEIA_pos_dict)
        metrics_LEIA_pos = metrics_LEIA_pos.transpose()
        
        metrics_RF_dict = classification_report(rec_RF['fechou'], rec_RF['previsao_RF'], output_dict=True, zero_division=1)
        metrics_RF = pd.DataFrame(metrics_RF_dict)
        metrics_RF = metrics_RF.transpose()
        
        rec_RF_pos = rec_RF.copy()
        rec_RF_pos['previsao_RF'] = 'Y'
        
        metrics_RF_pos_dict = classification_report(rec_RF_pos['fechou'], rec_RF_pos['previsao_RF'], output_dict=True, zero_division=1)
        metrics_RF_pos = pd.DataFrame(metrics_RF_pos_dict)
        metrics_RF_pos = metrics_RF_pos.transpose()
        
        metrics_NB_dict = classification_report(rec_NB['fechou'], rec_NB['previsao_NB'], output_dict=True, zero_division=1)
        metrics_NB = pd.DataFrame(metrics_NB_dict)
        metrics_NB = metrics_NB.transpose()
        
        rec_NB_pos = rec_NB.copy()
        rec_NB_pos['previsao_NB'] = 'Y'
        
        metrics_NB_pos_dict = classification_report(rec_NB_pos['fechou'], rec_NB_pos['previsao_NB'], output_dict=True, zero_division=1)
        metrics_NB_pos = pd.DataFrame(metrics_NB_pos_dict)
        metrics_NB_pos = metrics_NB_pos.transpose()
        
        # Display dos resultados
        data_fim = {'Modelo': ['LEIA','LEIA POS','RF','RF POS','NB','NB POS'],
                    'Precisão': [metrics_LEIA['precision']['weighted avg'], metrics_LEIA_pos['precision']['weighted avg'], metrics_RF['precision']['weighted avg'], metrics_RF_pos['precision']['weighted avg'], metrics_NB['precision']['weighted avg'], metrics_NB_pos['precision']['weighted avg']],
                    'Suport': [metrics_LEIA['support']['weighted avg'], metrics_LEIA_pos['support']['weighted avg'], metrics_RF['support']['weighted avg'], metrics_RF_pos['support']['weighted avg'], metrics_NB['support']['weighted avg'], metrics_NB_pos['support']['weighted avg']]}
        df_data_fim = pd.DataFrame(data_fim)
        pw(df_data_fim)
        
        
        #%% SIMULAÇÃO DE COMPRA
    
        compra_LEIA_total.loc[k,'Dia'] = hj
        compra_RF_total.loc[k,'Dia'] = hj
        compra_NB_total.loc[k,'Dia'] = hj
        compra_GAP_total.loc[k,'Dia'] = hj
        
        rec['comprar_LEIA'] = 0
        rec['comprar_RF'] = 0
        rec['comprar_NB'] = 0
        rec['comprar_GAP'] = 0
        
        qtd_LEIA = 0
        qtd_RF = 0
        qtd_NB = 0
        qtd_GAP = 0
        
        for i in range(0,len(rec)):
            
            # Se para fechar o GAP precisamos de uma variação positiva e a previsão é que o GAP será fechado
            if rec.loc[i,'pfechar'] == 'positive' and rec.loc[i,'previsao_LEIA'] == 'Y':
                qtd_LEIA = qtd_LEIA + 1
                rec.loc[i,'comprar_LEIA'] = 1
            if rec.loc[i,'pfechar'] == 'positive' and rec.loc[i,'previsao_RF'] == 'Y':
                qtd_RF = qtd_RF + 1
                rec.loc[i,'comprar_RF'] = 1
            if rec.loc[i,'pfechar'] == 'positive' and rec.loc[i,'previsao_NB'] == 'Y':
                qtd_NB = qtd_NB + 1
                rec.loc[i,'comprar_NB'] = 1
                
            # Se para fechar o GAP precisamos de uma variação positiva 
            if rec.loc[i,'pfechar'] == 'positive':
                qtd_GAP = qtd_GAP + 1
                rec.loc[i,'comprar_GAP'] = 1
                
        compra_LEIA = rec[rec['comprar_LEIA'] == 1]
        compra_RF = rec[rec['comprar_RF'] == 1]
        compra_NB = rec[rec['comprar_NB'] == 1]
        compra_GAP = rec[rec['comprar_GAP'] == 1]
        
        compra_LEIA = compra_LEIA.reset_index(drop=True)
        compra_RF = compra_RF.reset_index(drop=True)
        compra_NB = compra_NB.reset_index(drop=True)
        compra_GAP = compra_GAP.reset_index(drop=True)
        
        compra_LEIA['var'] = ''
        compra_RF['var'] = ''
        compra_NB['var'] = ''
        compra_GAP['var'] = ''
        compra_LEIA['lucro'] = ''
        compra_RF['lucro'] = ''
        compra_NB['lucro'] = ''
        compra_GAP['lucro'] = ''
        compra_LEIA['lucro_acumulado'] = ''
        compra_RF['lucro_acumulado'] = ''
        compra_NB['lucro_acumulado'] = ''
        compra_GAP['lucro_acumulado'] = ''
        
        prev_LEIA = 0
        prev_RF = 0
        prev_NB = 0
        prev_GAP = 0
        
        # Variação e lucro comprando ações que precisam subir para fechar GAP e recomendadas por LEIA
        for i in range(0,len(compra_LEIA)):
            for j in range(21,compra_LEIA.shape[1]-7):
                if compra_LEIA.iloc[i,j] <= (1-pa)*compra_LEIA.loc[i,'open']:
                    compra_LEIA.loc[i,'var'] = compra_LEIA.iloc[i,j]/compra_LEIA.loc[i,'open']-1
                    compra_LEIA.loc[i,'lucro'] = investimento/qtd_LEIA*compra_LEIA.loc[i,'var']
                    compra_LEIA.loc[i,'lucro_acumulado'] = investimento_acumulado_LEIA/qtd_LEIA*compra_LEIA.loc[i,'var']
                    break
                elif compra_LEIA.iloc[i,j] >= compra_LEIA.loc[i,'close_previous']:
                    compra_LEIA.loc[i,'var'] = compra_LEIA.iloc[i,j]/compra_LEIA.loc[i,'open']-1
                    compra_LEIA.loc[i,'lucro'] = investimento/qtd_LEIA*compra_LEIA.loc[i,'var']
                    compra_LEIA.loc[i,'lucro_acumulado'] = investimento_acumulado_LEIA/qtd_LEIA*compra_LEIA.loc[i,'var']
                    break
                else:
                    compra_LEIA.loc[i,'var'] = compra_LEIA.loc[i,'close_today']/compra_LEIA.loc[i,'open']-1
                    compra_LEIA.loc[i,'lucro'] = investimento/qtd_LEIA*compra_LEIA.loc[i,'var']
                    compra_LEIA.loc[i,'lucro_acumulado'] = investimento_acumulado_LEIA/qtd_LEIA*compra_LEIA.loc[i,'var']
             
            if compra_LEIA.loc[i,'previsao_LEIA'] == compra_LEIA.loc[i,'fechou']:
                prev_LEIA = prev_LEIA + 1
        
        # Variação e lucro comprando ações que precisam subir para fechar GAP e recomendadas por RF
        for i in range(0,len(compra_RF)):      
            for j in range(21,compra_RF.shape[1]-7):
                if compra_RF.iloc[i,j] <= (1-pa)*compra_RF.loc[i,'open']:
                    compra_RF.loc[i,'var'] = compra_RF.iloc[i,j]/compra_RF.loc[i,'open']-1
                    compra_RF.loc[i,'lucro'] = investimento/qtd_RF*compra_RF.loc[i,'var']
                    compra_RF.loc[i,'lucro_acumulado'] = investimento_acumulado_RF/qtd_RF*compra_RF.loc[i,'var']
                    break
                elif compra_RF.iloc[i,j] >= compra_RF.loc[i,'close_previous']:
                    compra_RF.loc[i,'var'] = compra_RF.iloc[i,j]/compra_RF.loc[i,'open']-1
                    compra_RF.loc[i,'lucro'] = investimento/qtd_RF*compra_RF.loc[i,'var']
                    compra_RF.loc[i,'lucro_acumulado'] = investimento_acumulado_RF/qtd_RF*compra_RF.loc[i,'var']
                    break
                else:
                    compra_RF.loc[i,'var'] = compra_RF.loc[i,'close_today']/compra_RF.loc[i,'open']-1
                    compra_RF.loc[i,'lucro'] = investimento/qtd_RF*compra_RF.loc[i,'var']
                    compra_RF.loc[i,'lucro_acumulado'] = investimento_acumulado_RF/qtd_RF*compra_RF.loc[i,'var']
             
            if compra_RF.loc[i,'previsao_RF'] == compra_RF.loc[i,'fechou']:
                prev_RF = prev_RF + 1
                

        # Variação e lucro comprando ações que precisam subir para fechar GAP e recomendadas por NB
        for i in range(0,len(compra_NB)):      
            for j in range(21,compra_NB.shape[1]-7):
                if compra_NB.iloc[i,j] <= (1-pa)*compra_NB.loc[i,'open']:
                    compra_NB.loc[i,'var'] = compra_NB.iloc[i,j]/compra_NB.loc[i,'open']-1
                    compra_NB.loc[i,'lucro'] = investimento/qtd_NB*compra_NB.loc[i,'var']
                    compra_NB.loc[i,'lucro_acumulado'] = investimento_acumulado_NB/qtd_NB*compra_NB.loc[i,'var']
                    break
                elif compra_NB.iloc[i,j] >= compra_NB.loc[i,'close_previous']:
                    compra_NB.loc[i,'var'] = compra_NB.iloc[i,j]/compra_NB.loc[i,'open']-1
                    compra_NB.loc[i,'lucro'] = investimento/qtd_NB*compra_NB.loc[i,'var']
                    compra_NB.loc[i,'lucro_acumulado'] = investimento_acumulado_NB/qtd_NB*compra_NB.loc[i,'var']
                    break
                else:
                    compra_NB.loc[i,'var'] = compra_NB.loc[i,'close_today']/compra_NB.loc[i,'open']-1
                    compra_NB.loc[i,'lucro'] = investimento/qtd_NB*compra_NB.loc[i,'var']
                    compra_NB.loc[i,'lucro_acumulado'] = investimento_acumulado_NB/qtd_NB*compra_NB.loc[i,'var']
             
            if compra_NB.loc[i,'previsao_NB'] == compra_NB.loc[i,'fechou']:
                prev_NB = prev_NB + 1                   
                
                
        # Variação e lucro comprando ações que precisam subir para fechar GAP
        for i in range(0,len(compra_GAP)):      
             for j in range(21,compra_GAP.shape[1]-7):
                 if compra_GAP.iloc[i,j] <= (1-pa)*compra_GAP.loc[i,'open']:
                     compra_GAP.loc[i,'var'] = compra_GAP.iloc[i,j]/compra_GAP.loc[i,'open']-1
                     compra_GAP.loc[i,'lucro'] = investimento/qtd_GAP*compra_GAP.loc[i,'var']
                     compra_GAP.loc[i,'lucro_acumulado'] = investimento_acumulado_GAP/qtd_GAP*compra_GAP.loc[i,'var']
                     break
                 elif compra_GAP.iloc[i,j] >= compra_GAP.loc[i,'close_previous']:
                     compra_GAP.loc[i,'var'] = compra_GAP.iloc[i,j]/compra_GAP.loc[i,'open']-1
                     compra_GAP.loc[i,'lucro'] = investimento/qtd_GAP*compra_GAP.loc[i,'var']
                     compra_GAP.loc[i,'lucro_acumulado'] = investimento_acumulado_GAP/qtd_GAP*compra_GAP.loc[i,'var']
                     break
                 else:
                     compra_GAP.loc[i,'var'] = compra_GAP.loc[i,'close_today']/compra_GAP.loc[i,'open']-1
                     compra_GAP.loc[i,'lucro'] = investimento/qtd_GAP*compra_GAP.loc[i,'var']
                     compra_GAP.loc[i,'lucro_acumulado'] = investimento_acumulado_GAP/qtd_GAP*compra_GAP.loc[i,'var']
              
             if compra_GAP.loc[i,'fechou'] == 'Y':
                 prev_GAP = prev_GAP + 1
                    
        lucro_LEIA = compra_LEIA['lucro'].sum()
        lucro_RF = compra_RF['lucro'].sum()
        lucro_NB = compra_NB['lucro'].sum()
        lucro_GAP = compra_GAP['lucro'].sum()
        lucro_acumulado_LEIA = compra_LEIA['lucro_acumulado'].sum()
        lucro_acumulado_RF = compra_RF['lucro_acumulado'].sum()
        lucro_acumulado_NB = compra_NB['lucro_acumulado'].sum()
        lucro_acumulado_GAP = compra_GAP['lucro_acumulado'].sum()
        
        compra_RF_total.loc[k,'Lucro'] = lucro_RF
        compra_NB_total.loc[k,'Lucro'] = lucro_NB
        compra_LEIA_total.loc[k,'Lucro'] = lucro_LEIA
        compra_GAP_total.loc[k,'Lucro'] = lucro_GAP
        compra_RF_total.loc[k,'Lucro Acumulado'] = lucro_acumulado_RF
        compra_NB_total.loc[k,'Lucro Acumulado'] = lucro_acumulado_NB
        compra_LEIA_total.loc[k,'Lucro Acumulado'] = lucro_acumulado_LEIA
        compra_GAP_total.loc[k,'Lucro Acumulado'] = lucro_acumulado_GAP
        
        compra_LEIA_total.loc[k,'Quantidade'] = str(len(compra_LEIA))
        compra_RF_total.loc[k,'Quantidade'] = str(len(compra_RF))
        compra_NB_total.loc[k,'Quantidade'] = str(len(compra_NB))
        compra_GAP_total.loc[k,'Quantidade'] = str(len(compra_GAP))
    
        # Saídas
        pw('\n'+'Lucro LEIA: ' + str(lucro_LEIA) + ' / Lucro RF: '+ str(lucro_RF) + ' / Lucro NB: '+ str(lucro_NB) +  ' / Lucro GAP: '+ str(lucro_GAP))
        pw('Lucro Acumulado LEIA: ' + str(lucro_acumulado_LEIA) + ' / Lucro Acumulado RF: '+ str(lucro_acumulado_RF) + ' / Lucro Acumulado NB: '+ str(lucro_acumulado_NB) + ' / Lucro Acumulado GAP: '+ str(lucro_acumulado_GAP))
        
        if len(compra_LEIA) != 0:
            acerto_LEIA = prev_LEIA/len(compra_LEIA)*100 
            pw('Acerto LEIA: ' + str(acerto_LEIA) + ' / Quantidade LEIA: ' + str(len(compra_LEIA)))
            compra_LEIA_total.loc[k,'Acerto'] = acerto_LEIA
        else:
            pw('Compra LEIA está vazio')
            compra_LEIA_total.loc[k,'Acerto'] = np.nan
         
        if len(compra_RF) != 0:
            acerto_RF = prev_RF/len(compra_RF)*100 
            pw('Acerto RF: '+ str(acerto_RF) + ' / Quantidade RF: '+ str(len(compra_RF)))
            compra_RF_total.loc[k,'Acerto'] = acerto_RF
        else:
            pw('Compra RF está vazio')
            compra_RF_total.loc[k,'Acerto'] = np.nan
            
        if len(compra_NB) != 0:
             acerto_NB = prev_NB/len(compra_NB)*100 
             pw('Acerto NB: '+ str(acerto_NB) + ' / Quantidade NB: '+ str(len(compra_NB)))
             compra_NB_total.loc[k,'Acerto'] = acerto_NB
        else:
             pw('Compra NB está vazio')
             compra_NB_total.loc[k,'Acerto'] = np.nan
            
        if len(compra_GAP) != 0:
            acerto_GAP = prev_GAP/len(compra_GAP)*100 
            pw('Acerto GAP: '+ str(acerto_GAP) + ' / Quantidade GAP: '+ str(len(compra_GAP)))
            compra_GAP_total.loc[k,'Acerto'] = acerto_GAP
        else:
            pw('Compra GAP está vazio')
            compra_GAP_total.loc[k,'Acerto'] = np.nan
            
        # Ajuste investimento acumulado
        investimento_acumulado_LEIA = investimento_acumulado_LEIA + lucro_LEIA
        investimento_acumulado_RF = investimento_acumulado_RF + lucro_RF
        investimento_acumulado_NB = investimento_acumulado_NB + lucro_NB
        investimento_acumulado_GAP = investimento_acumulado_GAP + lucro_GAP
            
        # Exportar dataframes diários em CSV 
        df.to_csv('./outputs/'+hj+'_df.csv',sep=';')
        metrics_LEIA.to_csv('./outputs/'+hj+'_metrics_LEIA.csv',sep=';')
        metrics_RF.to_csv('./outputs/'+hj+'_metrics_RF.csv',sep=';')
        metrics_NB.to_csv('./outputs/'+hj+'_metrics_NB.csv',sep=';')
        metrics_LEIA_pos.to_csv('./outputs/'+hj+'_metrics_LEIA_pos.csv',sep=';')
        metrics_RF_pos.to_csv('./outputs/'+hj+'_metrics_RF_pos.csv',sep=';')
        metrics_NB_pos.to_csv('./outputs/'+hj+'_metrics_NB_pos.csv',sep=';')
        rec.to_csv('./outputs/'+hj+'_rec.csv',sep=';')
        rec_LEIA.to_csv('./outputs/'+hj+'_rec_LEIA.csv',sep=';')
        rec_RF.to_csv('./outputs/'+hj+'_rec_RF.csv',sep=';')
        rec_NB.to_csv('./outputs/'+hj+'_rec_NB.csv',sep=';')
        compra_LEIA.to_csv('./outputs/'+hj+'_compra_LEIA.csv',sep=';')
        compra_RF.to_csv('./outputs/'+hj+'_compra_RF.csv',sep=';')
        compra_NB.to_csv('./outputs/'+hj+'_compra_NB.csv',sep=';')
        compra_GAP.to_csv('./outputs/'+hj+'_compra_GAP.csv',sep=';')
        
    except Exception as e:
        
        pw('Erro no dia '+ hj)
        pw(f"Exception type: {type(e).__name__}")
        
# Exportar dataframes totais em CSV     
compra_LEIA_total.to_csv('./outputs/compra_LEIA_total.csv',sep=';')
compra_RF_total.to_csv('./outputs/compra_RF_total.csv',sep=';')
compra_NB_total.to_csv('./outputs/compra_NB_total.csv',sep=';')
compra_GAP_total.to_csv('./outputs/compra_GAP_total.csv',sep=';')