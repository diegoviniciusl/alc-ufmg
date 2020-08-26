#!/usr/bin/env python
# coding: utf-8

# # Exercício de Programação 2: PageRank
# 
# <font color="red">**Prazo de submissão: 23:55h do dia 24/08/2020** </font>
# 
# 2020.1 Álgebra Linear Computacional - DCC - UFMG
# 
# Erickson e Fabricio
# 
# Instruções:
# * Antes de submeter suas soluções, certifique-se de que tudo roda como esperado. Primeiro, **reinicie o kernel** no menu, selecione Kernel$\rightarrow$Restart e então execute **todas as células** (no menu, Cell$\rightarrow$Run All)
# * **Apenas o arquivo .py deve ser submetido**. Você deve salvar o seu notebook em Python script (no menu, File $\rightarrow$ Download .py) e enviar o script Python no Laboratório de Programação Virtual (lá você pode checar as instruções detalhadas para obter sua nota automática).
# * **Preste bastante atenção nos nomes das variáveis e métodos** (irá estar em negrito), se elas estiverem diferentes do que foi pedido no exercício, *sua resposta será considerada incorreta pelo corretor automático*.
# * Use apenas a biblioteca **numpy**.
# * Não deixe de preencher seu nome e número de matrícula na célula a seguir

# **Nome:** *Diego Vinicius de Oliveira Silva*<br>
#           *Lucca Silva Medeiros*<br>
#           *Vinícius Correia Fonseca de Castro*<br>
#           *Rafael Lenti Barbosa*
# 
# **Matrícula:** *2019054471*<br>
#                *2019054773*<br>
#                *2019055044*<br>
#                *2019054951*
# 
# * Todo material consultado na Internet deve ser referenciado (incluir URL).
# 
# Este trabalho está dividido em três partes:
#  * **Parte 0**: Esta parte não vale nota, mas é fundamental para entender o que se pede a seguir.
#  * **Parte 1**: Pagerank sem saltos aleatórios em grafo pequeno
#  * **Parte 2**: Pagerank (com saltos aleatórios) em grafo pequeno

# ## Parte 0: Revisão de conceitos
# 
# I. O **primeiro autovetor** (isto é, o autovetor associado ao maior autovalor em módulo) pode ser calculado rapidamente através do método da potência, desde que o *gap* entre o maior e o segundo maior autovalor (em módulo) seja grande. Uma implementação simples do método da potência é mostrada a seguir.

# In[1]:


import numpy as np

def powerMethod(A, niter=10):
    n = len(A)
    w = np.ones((n,1))/n
    for i in range(niter):
        w = A.dot(w)        
    return w


# II. Dado um grafo $G=(V,E)$, podemos obter uma **matriz de probabilidade de transição** $P$ dividindo-se cada linha de $A$ pela soma dos elementos da linha. Seja $D = A \times \mathbf{1}$ a matriz diagonal contendo a soma das linhas de $A$. Temos que
# 
# $$
# P = D^{-1} \times A.
# $$

# III. A matriz de probabilidade de transição $P$ de certos grafos direcionados satisfaz
# 
# $$
# v^\top P = v^\top \textrm{ou $P^\top v = v$},
# $$
# 
# onde $v$ é o primeiro autovetor de $P^\top$. A equação da direita é mais fácil de ser utilizada, pois ela tem a forma canônica $Ax=b$. Já a equação da direita é mais fácil de ser interpretada. Para todo $j=1,\ldots,n$,
# 
# $$
# \sum_{i=1} v_i P_{ij} = v_j \\
# \sum_{i=1} v_i \frac{A_{ij}}{D_{ii}} = v_j \\
# \sum_{i:(i,j) \in E} v_i \frac{1}{D_{ii}} = v_j
# $$

# IV. Assuma que $v$ seja normalizado de forma que $\sum_j v_j = 1$. O PageRank (sem saltos) de um vértice $j$ é dado por $v_j$, onde $v$ é o primeiro autovetor de $P^\top$. Esta é uma maneira de medir sua relevância. A intuição da Equação $\sum_{i:(i,j) \in E} v_i /D_{ii} = v_j$ é que a relevância de $j$ é a soma das relevâncias dos vértices $i$ que apontam para $j$ normalizados pelos seus respectivos graus de saída.

# ## Parte 1: Pagerank sem saltos aleatórios em grafo pequeno
# 
# Considere o grafo a seguir composto por $n=4$ vértices e $m=8$ arestas. 
# <img src="images/directedgraph.png"/>
# 
# Certifique-se de que encontrou as $m=8$ arestas.

# **1.1** Crie um numpy array chamado <b> A </b>, contendo a matriz de adjacência.

# In[2]:


import numpy as np

A = np.array([[0,1,1,0],[0,0,1,1],[0,0,0,1],[1,1,1,0]])


# **1.2** Escreva uma função chamada <b>matrizDeTransicao</b> que recebe como entrada uma matriz $n \times n$ e retorna a matriz de probabilidade de transição desta matriz. Aplique a função em <b>A</b> e armazene o resultado na variável <b>P</b>, e depois imprima <b>P</b>.

# In[3]:


def matrizDeTransicao(A):
    n = len(A)
    D = np.zeros((n, n))
    vecAux = np.ones((n, 1))
    soma = A@vecAux
    for i in range(n):
        D[i][i] = soma[i] 
    
    return np.linalg.inv(D)@A
    
P = matrizDeTransicao(A)
print(P)


# **1.3** Use a função <i>np.linalg.eig</i> para calcular o primeiro autovetor de $P^\top$. Normalize o autovetor pela sua soma em uma variável chamada <b>autovec</b> e imprima o resultado. (Observação: os elementos do autovetor serão retornados como números complexos, mas a parte imaginária será nula e pode ser ignorada.)

# In[4]:


PT = P.T

autovalor,autovector = np.linalg.eig(PT)
autovec = [autovector[0][0], autovector[1][0], autovector[2][0], autovector[3][0]]
soma = 0
i = 0

for valorX in autovec:
    soma = soma + valorX
    
for valorX in autovec:
    autovec[i] = valorX/soma
    i = i+1    
autovec = np.array(autovec)
print(autovec)


# **1.4** Verifique que o método da potência aplicado a $P^\top$ retorna uma aproximação para o primeiro autovetor. Atribua o resultado retornado pelo método na variável <b> result_pm </b> e imprima-o.

# In[5]:


result_pm = powerMethod(PT, niter=10)

print(result_pm)


# **1.5** Implemente uma função <b>powerMethodEps(A, epsilon)</b> que executa o método da potência até que a condição de convergência $\|w_{t} - w_{t-1}\| < \epsilon$ seja atingida. Para a matriz $P^\top$ com $\epsilon=10^{-5}$, armazene o resultado do método da potência na variável <b>result_pm_eps</b> *(1.5.1)*, e o número de iterações na variável <b>nb_iters</b> *(1.5.2)*.
# 
# Imprima o resultado das duas variáveis.

# In[6]:


def powerMethodEps(A, epsilon):
    n = len(A)
    w0 = np.ones((n,1))/n
    w1 = A.dot(w0)
    nb_iters = 1
    
    while(np.linalg.norm(w1-w0) >= epsilon):
        w0 = w1
        w1 = A.dot(w1)
        nb_iters = nb_iters + 1
        
    result_pm_eps = w1
    return result_pm_eps, nb_iters

result_pm_eps, nb_iters = powerMethodEps(PT, 0.00001)

print(result_pm_eps)
print(nb_iters)


# ## Parte II: Pagerank (com saltos aleatórios) em grafo pequeno
# 
# Agora iremos modificar a matriz A de forma a:
#  * adicionar um novo vértice 4, e
#  * adicionar uma aresta de 3 para 4.
#  
# Obviamente a matriz de probabilidade de transição não está definida para a nova matriz $A$. Vértices que não possuem arestas de saída (como o vértice 4) são conhecidos como *dangling nodes*. Para resolver este e outros problemas, incluiremos a possibilidade de realizar saltos aleatórios de um vértice para qualquer outro vértice.
# 
# Em particular, assume-se que com probabilidade $\alpha$, seguimos uma das arestas de saída em $A$ e, com probabilidade $1-\alpha$ realizamos um salto aleatório, isto é, transicionamos do vértice $v$ para um dos $n$ vértices do grafo (incluindo $v$) escolhido uniformemente. Quando não existem *dangling nodes*, a nova matriz de probabilidade de transição é dada por
# 
# $$
# P = \alpha D^{-1} A + (1-\alpha) \frac{\mathbf{1}\mathbf{1}^\top}{n}
# $$
# 
# Quando existem *dangling nodes*, a única possibilidade a partir desses nós é fazer saltos aleatórios. Mais precisamente, se $i$ é um vértice sem arestas de saída, desejamos que a $i$-ésima linha de $P$ seja o vetor $[1/n,\ldots,1/n]$. Uma forma de satisfazer essa definição é preencher com 1's as linhas de $A$ que correspondem aos *dangling nodes*. Uma desvantagem desta estratégia é que faz com que $A$ fique mais densa (mais elementos não-nulos).
# 
# Um valor típico usado para $\alpha$ é $0.85$.

# **2.1** Crie um novo numpy array chamado <b> A_new </b> contendo o vértice 4 e a aresta (3,4).

# In[7]:


A_new = np.array([[0,1,1,0,0],[0,0,1,1,0],[0,0,0,1,0],[1,1,1,0,1],[0,0,0,0,0]])


# **2.2** Crie uma função **fixDangling(M)** que retorna uma cópia modificada da matriz de adjacência **M** onde cada *dangling node* do grafo original possui arestas para todos os vértices do grafo. *Dica:* Você pode criar um vetor $d$ com os graus de saída e acessar as linhas de $M$ correpondentes aos *dangling nodes* por $M[d==0,:]$. Imprima uma nova matriz chamada **A_fixed** retornada após chamar *fixDangling* para **A_new**.  

# In[8]:


def fixDangling(M):
    n = len(M)
    V = M.copy()
    vecAux = np.ones((n, 1))
    d = M@vecAux
    for i in range(0,n):
        if d[i]==0:
            for j in range(0,n):
                V[i][j]=1
    return V

A_fixed = fixDangling(A_new)
print(A_fixed)


# **2.3** Crie uma função **matrizDeTransicao(M, alpha)** que receba como parâmetro também a probabilidade *alpha*  de não fazermos um salto aleatório. Você pode assumir que **M** foi retornada por *fixDanglig*, logo, não possui *dangling nodes*. Imprima as matrizes:
# 
#  * *(2.3.1)* **P_2** obtida ao chamar *matrizDeTransicao* para os parâmetros <b>A</b> e <b>alpha</b> = $0.85$;
#  * *(2.3.2)* **P_new** obtida ao chamar *matrizDeTransicao* para os parâmetros **A_fixed** e **alpha** = $0.85$.

# In[9]:


def matrizDeTransicao(M, alpha):
    n = len(M)
    
    D = np.zeros((n, n))
    vecAux = np.ones((n, 1))
    soma = M@vecAux
    for i in range(n):
        D[i][i] = soma[i] 
    E = np.ones((n, n))
    
    for i in range(n):
        if D[i][i] != 0:
            D[i][i] = 1/D[i][i]
        else:
             D[i][i] = 0
    
    pg = (((1-alpha)/n)*E)+(alpha*(D@M))
    return pg

P_2 = matrizDeTransicao(A, 0.85)
P_new = matrizDeTransicao(A_fixed, 0.85)
print(P_2)
print(P_new)


# **2.4** Armazene, respectivamente, o resultado do método da potência com:
# * *(2.4.1)* $P_2^\top$ e $\epsilon=10^{-5}$
# * *(2.4.2)* $P_\textrm{new}^\top$ e $\epsilon=10^{-5}$.
# 
# nas variáveis **pm_eps_P2** e **pm_eps_Pnew**; 

# In[10]:


P_2_T = P_2.T
P_new_T = P_new.T

pm_eps_P2, aux = powerMethodEps(P_2_T, 0.00001)
pm_eps_Pnew,aux = powerMethodEps(P_new_T, 0.00001)

print(pm_eps_P2)
print(pm_eps_Pnew)


# **2.5** Sejam $i_\max$ e $i_\min$ os índices dos vértices com maior e menor PageRank de **A_fixed**. Vamos verificar como a adição de um novo link pode ajudar a promover uma página web (vértice). Adicione uma aresta do vértice $i_\max$ para o vértice $i_\min$ (se já houver aresta, aumente de 1 para 2 o elemento da matriz de adjacência). Salve o valor do novo pagerank na variável **new_pagerank**. Qual é o novo pagerank de $i_\min$?

# In[11]:


A2 = np.array([[0,1,1,0,0],[0,0,1,1,0],[0,0,0,1,0],[2,1,1,0,1],[0,0,0,0,0]])
A2_fixed = fixDangling(A2)
P_25 = matrizDeTransicao(A2_fixed, 0.85)
P_25T = P_25.T
new_pagerank,aux = powerMethodEps(P_25T, 0.00001)
print(new_pagerank)


# Bibliografia:
# 
# - https://numpy.org/doc/stable/user/basics.creation.html#:~:text=j%5D%5D)-,Intrinsic%20NumPy%20Array%20Creation,The%20default%20dtype%20is%20float64.&text=%3E%3E%3E,-np.
# 
# - https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
# 
# - https://docs.scipy.org/doc/numpy-1.17.0/reference/generated/numpy.linalg.eig.html
# 
# - https://www.kite.com/python/answers/how-to-normalize-an-array-in-numpy-in-python
# 
# - https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.linalg.eig.html
# 
# - https://stackoverflow.com/questions/16296643/convert-tuple-to-list-and-back
# 
# - https://www.researchgate.net/post/Why_eigenvectors_seem_incorrect_in_python
# 

# In[ ]:




