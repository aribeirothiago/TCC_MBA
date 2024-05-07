#%% IMPORTS
import pandas as pd
import yfinance as yf
import numpy as np

import pathlib
import glob
import os

from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from time import sleep
from pygooglenews import GoogleNews
from leia import SentimentIntensityAnalyzer
from datetime import time, datetime, timedelta

#%% ENTRADAS

# Thresholds
# sup_LEIA = 0.35
# inf_LEIA = -0.55
sup_LEIA = 0.1
inf_LEIA = -0.1

# Perda aceitável
pa = 0.01

# True ou False para baixar o arquivo da carteira 
download = False

# Feriados
feriados = ['2024-01-01', '2024-02-12','2024-02-13','2024-03-29','2024-05-01','2024-05-30','2024-11-15','2024-12-24','2024-12-25','2024-12-31']

# Datas
ontem = (datetime.today()-timedelta(days=1)).strftime('%Y-%m-%d')
hoje = datetime.today().strftime('%Y-%m-%d')
amanha = (datetime.today()+timedelta(days=1)).strftime('%Y-%m-%d')

# Se ontem não for um dia útil
ontem_util = ontem
i=1
while ontem_util in feriados or datetime.weekday(datetime.strptime(ontem_util, '%Y-%m-%d')) in [5,6]:
    ontem_util = (datetime.strptime(ontem, '%Y-%m-%d')-timedelta(days=i)).strftime('%Y-%m-%d')
    i=i+1
    
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

#%% BUSCAR NOTÍCIAS

# Data das notícias
endtime = datetime.combine(datetime.strptime(hoje, '%Y-%m-%d'),time(10, 0))
            
# Inicialização e configuração
gn = GoogleNews(lang='pt', country='BR')

# Criação da lista do dataframe
df_list = []

print("Buscando notícias...\n")

# Loop para buscar notícias que contêm os tickers desejados
for ticker in tickers:
    # Eliminar resultados com "varzea" para evitar links indesejados
    search = gn.search(f'"{ticker}" -varzea', from_=ontem, to_=amanha)
    
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
    

#%% RECOMENDAÇÕES: VERIFICAR AÇÕES QUE TENDEM A SUBIR OU CAIR

# Fazer uma média dos scores de todas as notícias com determinado ticker para LEIA
rec = df[['Code','Score_LEIA']].groupby(['Code']).mean().reset_index()

rec['Rec_LEIA'] = ''
# Loop para recomendações LEIA
for i in range(0,len(rec)):

    if rec.loc[i,'Score_LEIA'] >= sup_LEIA:
        rec.loc[i,'Rec_LEIA']  = 'positive'
    elif rec.loc[i,'Score_LEIA']  <= inf_LEIA:
        rec.loc[i,'Rec_LEIA'] = 'negative'
    else:
        rec.loc[i,'Rec_LEIA'] = 'neutral'
        
rec = rec[rec['Rec_LEIA'] == 'positive'].reset_index(drop=True)

#%% OBTER VARIAÇÃO DA AÇÃO NO DIA

print('Selecionando ações...')

precos_abertura = [0]

def generator():
  while any(item == 0 for item in precos_abertura): # Enquanto os preços forem iguais a 0
    yield

for _ in tqdm(generator()):

    compra = []
    precos_abertura = []
    precos_fechar = []
    
    # Loop para os tickers desejados
    todrop = np.zeros(len(rec))
    for i in range(0,len(rec)):
        ticker_symbol = rec.loc[i,'Code']+'.SA'
        
        # Obter dados para hoje e ontem
        data_hoje = yf.download(ticker_symbol, start=hoje,end=amanha, progress=False)
        data_ontem = yf.download(ticker_symbol, start=ontem, end=hoje, progress=False)
      
        # Verificar qual a condição necessária para fechar o gap
        if float(data_hoje['Open'].iloc[0]) > float(data_ontem['Close'].iloc[0]):
            pfechar = 'negative'
        elif float(data_hoje['Open'].iloc[0]) < float(data_ontem['Close'].iloc[0]):
            pfechar = 'positive'
        elif float(data_hoje['Open'].iloc[0]) == float(data_ontem['Close'].iloc[0]):
            pfechar = 'neutral'
        else: 
            pfechar = 'null'
        
        # Selecionar ações de interesse
        if pfechar == 'positive' and rec.loc[i,'Rec_LEIA'] == 'positive':
            compra.append(rec.loc[i,'Code'])
            precos_abertura.append(float(data_hoje['Open'].iloc[0]))
            precos_fechar.append(data_ontem['Close'].iloc[0])
            
    sleep(1)    

#%% SIMULAÇÃO DE COMPRA

vender = compra.copy()
vendidas = pd.DataFrame(columns=['Code','Preço Compra','Preço Desejado','Preço Mínimo','Preço Venda','Hora Venda'])
vendidas['Code'] = compra
vendidas['Preço Compra'] = precos_abertura
vendidas['Preço Desejado'] = precos_fechar
vendidas['Preço Mínimo'] = [(1 - pa) * preco for preco in precos_abertura]
acoes_compra = ', '.join(compra)

print('Lista de Ações: \n')
print(vendidas[['Code','Preço Compra','Preço Desejado','Preço Mínimo']])
print('\n')

def generator():
  while vender != []: # Enquanto a lista para vender não estiver vazia
    yield

for _ in tqdm(generator()):
    
    print('\n')
    
    # Loop para as ações de interesse
    for i in range(0,len(compra)):
        
        if compra[i] in vender:
        
            ticker_symbol= compra[i]+'.SA'
        
            try:
                # Obter preço atual da ação
                data_agora = yf.download(ticker_symbol, start=hoje,end=amanha, interval='1m', progress=False).tail(1)
                preco_agora = float(data_agora['Open'].iloc[0])
                
                print(compra[i]+': '+str(preco_agora))
                
                # Verificar condições para venda
                if (preco_agora >= vendidas.loc[i,'Preço Desejado'] or preco_agora < vendidas.loc[i,'Preço Mínimo'] or 
                    datetime.now() >= datetime(datetime.now().year, datetime.now().month, datetime.now().day, 17, 15, 0)):
                    
                    vendidas.loc[i,'Preço Venda'] = preco_agora
                    vendidas.loc[i,'Hora Venda'] = datetime.now().strftime('%H:%M:%S')
                    vender.remove(compra[i])
                    
                    if preco_agora >= vendidas.loc[i,'Preço Desejado']:
                        print(compra[i] +' vendida acima do Preço Desejado!')
                    elif preco_agora < vendidas.loc[i,'Preço Mínimo']:
                        print(compra[i] +' vendida abaixo do Preço Mínimo!')
                    else:
                        print(compra[i] +' vendida no fechamento!')
            except:
                 print(compra[i]+' indisponível!')
                
     
    print('\n')
    sleep(60)
    
# Exportar resultado   
vendidas.to_csv('./outputs_realtime/vendidas_'+hoje+'.csv',encoding='ansi',sep=';')

sleep(1200)

#%% PREÇO REAL E LUCRO

vendidas = vendidas.dropna().reset_index(drop=True)

investimento = 100

# Loop para as ações de interesse
for i in range(0,len(vendidas)):
    ticker_symbol = vendidas.loc[i,'Code']+'.SA'
    data_hoje_total = yf.download(ticker_symbol, start=hoje,end=amanha,interval='1m', progress=False)
    hora_venda = datetime.strptime(hoje + ' ' + str(vendidas.loc[i,'Hora Venda']), ('%Y-%m-%d %H:%M:%S'))
    if hora_venda >= datetime(datetime.now().year, datetime.now().month, datetime.now().day, 17, 0, 0):
        preco_real = data_hoje_total['Open'].iloc[-1]
    else:
        hora_indice = hora_venda.strftime('%Y-%m-%d %H:%M')+':00-03:00'
        preco_real = data_hoje_total.loc[hora_indice,'Open']
    vendidas.loc[i,'Preço Real'] = preco_real
    vendidas.loc[i,'Variação'] = (preco_real-vendidas.loc[i,'Preço Compra'])/vendidas.loc[i,'Preço Compra']
    vendidas.loc[i,'Lucro'] = investimento/len(vendidas)*vendidas.loc[i,'Variação']

# Exportar resultado   
vendidas.to_csv('./outputs_realtime/vendidas_'+hoje+'.csv',encoding='ansi',sep=';')