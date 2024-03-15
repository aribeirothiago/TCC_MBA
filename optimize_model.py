import pandas as pd
import re
import spacy
import seaborn as sns
import matplotlib.pylab as plt
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB

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

# Definir o espaço de busca dos hiperparâmetros para o modelo RF
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 20, 50],
    'min_samples_split': [2, 5, 10, 20, 50],
    'min_samples_leaf': [1, 2, 4, 10, 20],
    'max_features': ['sqrt', 'log2']
}

class_weights = {'negative':1, 'neutral':1, 'positive':5}

# Criar o modelo com OneVsRestClassifier para o RF
model_rf = (RandomForestClassifier(class_weight=class_weights))

# Criar o objeto RandomizedSearchCV para o RF
random_search_rf = RandomizedSearchCV(model_rf, param_distributions=param_grid_rf, n_iter=10, cv=5)

# Realizar a busca de hiperparâmetros para o RF
random_search_rf.fit(X_tfidf_pt_spc, y_pt_spc)

# Imprimir os melhores hiperparâmetros encontrados para o RF
params_rf = random_search_rf.best_params_
print("Melhores hiperparâmetros para rf:")
print(params_rf)

# Definir o espaço de busca dos hiperparâmetros para o modelo NB
param_grid_nb = {
    'alpha': [0.1, 0.5, 1.0, 5.0, 10.0]
}

class_weights = {'negative':1, 'neutral':1, 'positive':5}

# Criar o modelo com OneVsRestClassifier para o RF
model_rf = (RandomForestClassifier(class_weight=class_weights))

# Criar o objeto RandomizedSearchCV para o RF
random_search_rf = RandomizedSearchCV(model_rf, param_distributions=param_grid_rf, n_iter=10, cv=5)

# Realizar a busca de hiperparâmetros para o RF
random_search_rf.fit(X_tfidf_pt_spc, y_pt_spc)

# Imprimir os melhores hiperparâmetros encontrados para o RF
params_rf = random_search_rf.best_params_
print("Melhores hiperparâmetros para rf:")
print(params_rf)

# Criar o modelo para NB
model_nb = (MultinomialNB())

# Criar o objeto RandomizedSearchCV para o RF
random_search_nb = RandomizedSearchCV(model_nb, param_distributions=param_grid_nb, n_iter=10, cv=5)

# Realizar a busca de hiperparâmetros para o RF
random_search_nb.fit(X_tfidf_pt_spc, y_pt_spc)

# Imprimir os melhores hiperparâmetros encontrados para o RF
params_nb = random_search_nb.best_params_
print("Melhores hiperparâmetros para nb:")
print(params_nb)

#%%

#Divisão treino e teste
def split_data(X, y, split_point=-400):
    X_train = X[:split_point]
    y_train = y[:split_point]
    X_test = X[split_point:]
    y_test = y[split_point:]
    return X_train, y_train, X_test, y_test

X_train_pt_spc, y_train_pt_spc, X_test_pt_spc, y_test_pt_spc = split_data(X_tfidf_pt_spc, y_pt_spc)

def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    model_rf = (RandomForestClassifier(n_estimators = params_rf['n_estimators'],
                                       min_samples_split = params_rf['min_samples_split'],
                                       min_samples_leaf = params_rf['min_samples_leaf'],
                                       max_features = params_rf['max_features'],
                                       max_depth = params_rf['max_depth'],
                                       random_state = 10,
                                       class_weight ='balanced'
                                       ))
    
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