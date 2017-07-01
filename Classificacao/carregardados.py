import csv
import pandas as pd

def carrega_dados():
    dados = []
    marcacoes = []
    arquivo = open('acesso.csv','rb')
    leitor = csv.reader(arquivo)
    leitor.next()
    for home, como_funciona,contato, comprou in leitor:
        dados.append([int(home),int(como_funciona),int(contato)])
        marcacoes.append([int(comprou)])
    return dados , marcacoes

def carrega_dados_panda():
    df = pd.read_csv('Curso.csv')
    df_X = df[['busca','logado','home']]
    df_Y = df['comprou']
    X_dummers = pd.get_dummies(df_X).astype(int)
    Y_durmers = df_Y
    return X_dummers.values, Y_durmers.values, len(df)