import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importando o datset
pokemon = pd.read_csv('Pokemon-dataset.csv')
print(pokemon.head(13))
lendarios = pokemon[pokemon['Legendary'] == True]
print(lendarios.head(20))


# Fazendo alguns plots:
fig = plt.figure(figsize=(14, 8))
sns.set(style="darkgrid")
sns.countplot(x="Type 1", hue="Legendary", data=pokemon, palette="gist_rainbow_r")
plt.title("Tipo Primário")

fig = plt.figure(figsize=(14, 8))
sns.set(style="darkgrid")
sns.countplot(x="Type 2", hue="Legendary", data=pokemon, palette='gist_rainbow_r')
plt.title("Tipo Secundário")

fig = plt.figure(figsize=(14,8))
sns.set(style="darkgrid")
sns.countplot(x=lendarios["Type 1"], data=lendarios)
plt.title("Contagem dos Lendários Segundo Tipo Primário")

fig = plt.figure(figsize=(14,8))
sns.set(style="darkgrid")
sns.countplot(x=lendarios["Type 2"], data=lendarios)
plt.title("Contagem dos Lendários Segundo Tipo Secundário")



# Quantos do tipo Psíquico e do tipo Dragão são lendários?
print(pokemon[pokemon["Type 1"]=="Psychic"]["Legendary"].value_counts())
print(pokemon[pokemon["Type 1"]=="Dragon"]["Legendary"].value_counts())
print("A probabilidade de um pokémon do tipo (1) psiquíco ser um pokémon lendário é:",14/57)
print("A probabilidade de um pokémon do tipo (1) dragão ser um pokémon lendário é:",12/32)


# Criando um novo dataframe com exceção das variáveis categóricas:
pokemon_valores = pokemon.drop(["Generation","Type 2","Type 1","Name", "#"], axis=1)
print(pokemon_valores.head())

# Mais alguns plots.
fig = plt.figure(figsize=(15,9))
sns.set(style="darkgrid")
sns.scatterplot(x="Sp. Atk",y="Sp. Def",data=pokemon_valores,
                hue="Legendary", size="Total", sizes=(25,195), palette="rocket_r")
plt.xlabel("Ataque Especial")
plt.ylabel("Defesa Especial")
plt.title("Ataque Especial vs. Defesa Especial")



fig = plt.figure(figsize=(15,9))
sns.set(style="darkgrid")
sns.scatterplot(x="Attack",y="Defense",data=pokemon_valores,
                hue="Legendary", size="Total", sizes=(25,195), palette="viridis_r")
plt.xlabel("Ataque")
plt.ylabel("Defesa")
plt.title("Ataque vs. Defesa")


#    Da pra ver que pokémons lendários no geral tem bons atributos Sp. Atk e Sp. Def combinados!
#    Podemos dizer o mesmo em relação à ataque e defesa!
#    Vamos então usar um método de classificação, o escolhido foi o KNN

#    Aqui começa a criação do modelo KNN
#    Importando tudo que vamos precisar:

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix


#       Usamos o train_test_split() para dividir o dataset em um dataset de treino e outro de teste.
#    Ele retornará uma tupla, que em sequência, é constituída de variável explicativa de treino (X_treino),
#    variável explicativa de teste, X_teste seguida das variáveis de resposta de treino e teste, y_treino e y_teste
#    respectivamente.
#       Primeiro devemos criar um dataframe que não contém a coluna lendário, e passar como primeiro parâmetro
#     para o método train_test_split(), e em seguida a própria coluna "Legendary", que será a variável explica
#     tiva, por fim o tamanho do dataset de treino, em geral escolhe-se o equivalente a um terço do
#     dataset original.

# Parâmetros
params_ = pokemon_valores.drop("Legendary", axis=1)
print(params_.head(6))
# Variáveis
X_treino, X_teste, y_treino, y_teste = train_test_split(params_,pokemon["Legendary"],test_size=0.33)

# Sondar o valor de K:
taxa_de_erro = []

treino_scores = []
teste_scores  = []

for i in range(1,25):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_treino,y_treino)
    predic_i = knn.predict(X_teste)
    taxa_de_erro.append(np.mean(predic_i != y_teste))
    treino_scores.append(knn.score(X_treino,y_treino))
    teste_scores.append(knn.score(X_teste,y_teste))

    
# Vamos observar o erro dado o valor de K
fig = plt.figure(figsize=(12,6))
plt.plot(range(1,25),taxa_de_erro,color='blue', linestyle='dashed', marker='.',
         markerfacecolor='red', markersize=7)
plt.title('Taxa de Erro vs. Valor de K')
plt.xlabel('K')
plt.ylabel('Taxa de Erro')

plt.figure(figsize=(12,6))
ax1=sns.lineplot(range(1,25),treino_scores,marker='o',label='Treino Score')
ax2=sns.lineplot(range(1,25),teste_scores,marker='o',label='Teste Score')

knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_treino,y_treino)
predic = knn.predict(X_teste)
# Vamos observar nossa matriz de confusão:
print(confusion_matrix(y_teste,predic))
# E o status do modelo:
print(classification_report(y_teste,predic))

# Um pequeno teste:
bulbasauro = np.array([318,45,49,49,65,65,45])
bulbasauro = bulbasauro.reshape(1,-1)
predic2 = knn.predict(bulbasauro)
print(predic2)
#
megalomon_e_juca = ([720,150,124,126,112,118,90], [433,94,120,44,60,55,60])
predic3 = knn.predict(megalomon_e_juca)
print(predic3)

