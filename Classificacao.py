from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.svm import LinearSVC
from carregardados import carrega_dados_panda
from collections import Counter
import numpy as np

def fit_and_predict(nome,modelo,dados_treino,marcacoes_treino,dados_validacao,marcacoes_validacao):
    k = 10
    scores = cross_val_score(modelo, dados_treino, marcacoes_treino, cv = k)
    media = np.mean(scores)
    msg = u"Porcentagem de acerto {0} é de {1}% ".format(nome,media*100)
    print msg
    return media

def teste_real(modelo, validacao_dados, validacao_marcacoes):

    resultado = modelo.predict(validacao_dados)

    acertos = resultado == validacao_marcacoes

    total_de_acertos = sum(acertos)
    total_de_elementos = len(validacao_marcacoes)

    taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

    msg = "Taxa de acerto do vencedor entre os algoritmos no mundo real: {0}".format(taxa_de_acerto)
    print(msg)

try:
    print "Inicio"
    resultados = {}
    X, Y, tamanho = carrega_dados_panda()
    tamanho_treino = tamanho * 0.8
    tamanho_validacao = tamanho - tamanho_treino
    dados_treino = X[:int(tamanho_treino)]
    marcacoes_treino = Y[:int(tamanho_treino)]

    dados_validacao =X[int(tamanho_validacao):]
    marcacoes_validacao = Y[int(tamanho_validacao):]

    modeloMultinomialNB = MultinomialNB()
    treinoMultinomial = fit_and_predict('Multinomial',modeloMultinomialNB,dados_treino,marcacoes_treino,dados_validacao,marcacoes_validacao)
    resultados[treinoMultinomial] = modeloMultinomialNB

    modeloAdaBoost = AdaBoostClassifier()
    treinoAdaboost  = fit_and_predict('AdaBoostClassifier',modeloAdaBoost,dados_treino,marcacoes_treino,dados_validacao,marcacoes_validacao)
    resultados[modeloAdaBoost] = treinoAdaboost

    modeloOneVsRest = OneVsRestClassifier(LinearSVC(random_state = 0))
    resultadoOneVsRest = fit_and_predict("OneVsRest", modeloOneVsRest, dados_treino, marcacoes_treino, dados_validacao, marcacoes_validacao)
    resultados[resultadoOneVsRest] = modeloOneVsRest
    
    modeloOneVsOne = OneVsOneClassifier(LinearSVC(random_state = 0))
    resultadoOneVsOne = fit_and_predict("OneVsOne", modeloOneVsOne, dados_treino, marcacoes_treino, dados_validacao, marcacoes_validacao)
    resultados[resultadoOneVsOne] = modeloOneVsOne
    
    maximo = max(resultados)
    vencedor = resultados[maximo]
    
    print "Vencerdor: "
    print vencedor
    vencedor.fit(dados_treino, marcacoes_treino)
    teste_real(vencedor, dados_validacao, marcacoes_validacao)
    
    acerto_base = max(Counter(marcacoes_validacao).itervalues())
    taxa_de_acerto_base = 100.0 * acerto_base / len(marcacoes_validacao)
    print("Taxa de acerto base: %f" % taxa_de_acerto_base)
    
    total_de_elementos = len(marcacoes_treino)
    print("Total de teste: %d" % total_de_elementos)
except Exception as e:
    print str(e)
