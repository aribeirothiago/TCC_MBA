from selenium import webdriver
import pandas as pd
from time import sleep
import pathlib
import glob
import os
from pygooglenews import GoogleNews

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


#%%

df = pd.DataFrame(columns=['Code', 'Title'])

for i in range(0,len(tickers)):
    search = gn.search(f'"{tickers[i]}" -varzea', when = '1d')
    for item in search['entries']:
        if tickers[i] in item.title:
            df = df.append(pd.Series([tickers[i], item.title], index = ["Code","Title",]), ignore_index=True)
            

