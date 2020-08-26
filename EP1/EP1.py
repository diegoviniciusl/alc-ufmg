#!/usr/bin/env python
# coding: utf-8

# # Exercício de Programação 1
# 
# <font color="red">**Prazo de submissão: 23:55 do dia 19/03/2020** </font>
# 
# 2020.1 Álgebra Linear Computacional - DCC - UFMG
# 
# Erickson - Fabricio
# 
# Instruções:
# * Antes de submeter suas soluções, certifique-se de que tudo roda como esperado. Primeiro, **reinicie o kernel** no menu, selecione Kernel$\rightarrow$Restart e então execute **todas as células** (no menu, Cell$\rightarrow$Run All)
# * Apenas o arquivo .ipynb deve ser submetido. Ele não deve ser compactado.
# * Não deixe de preencher seu nome e número de matrícula na célula a seguir

# Nome do aluno: Diego Vinicius de Oliveira Silva
# 
# Matricula: 2019054471

# ## Parte 1: Python básico

# In[1]:


# Exercício 1 - Imprima na tela os números de 1 a 10 e armazene-os em uma lista.
list = []
for i in range(1, 11):
    print(i)
    list.append(i)
print(list)


# In[2]:


# Exercício 2 - Crie duas strings e concatene as duas em uma terceira string
string1 = "Diego "
string2 = "Vinicius"
string3 = string1 + string2
print(string3)


# In[3]:


# Exercício 3 - Crie uma tupla com os seguintes elementos: 1, 2, 2, 3, 4, 4, 4, 5 e depois utilize a função count do 
# objeto tupla para verificar quantas vezes o número 4 aparece na tupla
tuple = (1, 2, 2, 3, 4, 4, 4, 5)
print(tuple.count(4))


# In[4]:


# Exercício 4 - Crie um dicionário com 3 chaves e 3 valores e imprima na tela
dict = {"key_1":"value_1", "key_2":"value_2", "key_3":"value_3"}
print(dict)


# In[5]:


# Exercício 5 - Adicione mais um elemento ao dicionário criado no exercício anterior e imprima na tela
dict["key_4"] = "value_4"
print(dict)


# In[6]:


# Exercício 6 - Crie um novo dicionário invertendo o anterior (isto é, usando os valores como chave e vice-versa)
dict2 = {}
for i in dict:
    dict2[dict[i]] = i
print(dict2)


# In[7]:


# Exercício 7 - Crie uma função que receba uma string como argumento e retorne a mesma string em letras maiúsculas.
# Faça uma chamada à função, passando como parâmetro uma string
def upperCase(string):
    return string.upper()
print(upperCase("testE"))


# In[8]:


# Exercício 8 - Crie uma função que receba como parâmetro uma lista de 4 elementos, adicione 2 elementos a lista e 
# imprima a lista. Esta função não deve retornar nada.
def modifyList(list):
    list.append(5)
    list.append(6)
    print(list)
modifyList([1, 2, 3, 4])


# In[9]:


# Exercício 9 - Faça a correção dos erros no código abaixo e execute o programa. Dica: são 3 erros.
temperatura = float(input('Qual a temperatura? '))
if temperatura > 30:
    print('Vista roupas leves.')
else:
    print('Busque seus casacos.')


# In[10]:


# Exercício 10 - Crie uma função que conte quantas vezes a letra "r" aparece na frase abaixo. Use um placeholder (%) na 
# sua instrução de impressão
frase = "All models are wrong, but some are useful (George Box)"
def countR(string):
    return string.count('r')
print("The phrase contains %d r's" % countR(frase))


# ## Parte 2: numpy

# ## Questão 1
# 
# Crie as matrizes A, B e C abaixo e resolva as questões:
# 1. [ ] Calcule $(((A^T B)+B)C^{-1})$
# 2. [ ] Crie matrizes $\tilde A_{2x2}, \tilde B_{2x2}, \tilde C_{2x2}$, tal que sejam compostas pelo 2 primeiros elementos de cada linha das duas primeiras linhas. E repita a equação do item anterior.
# 
# **Dica** numpy.full, numpy.eye, numpy.ones e indexação de vetores.
# 
# $$
# A = \begin{bmatrix}
#  8 & 8 & 8\\
#  8 & 8 & 8\\
#  8 & 8 & 8
# \end{bmatrix}_{3\times 3}
# \qquad
# B = \begin{bmatrix}
#  1 & 1 & 1 & 1\\
#  1 & 1 & 1 & 1\\
#  1 & 1 & 1 & 1
# \end{bmatrix}_{3\times 4}
# \qquad
# C = \begin{bmatrix}
#  1 & 0 & 0 & 0 \\
#  0 & 1 & 0 & 0 \\
#  0 & 0 & 1 & 0 \\
#  0 & 0 & 0 & 1
# \end{bmatrix}_{4\times 4}
# $$

# In[11]:


# Código para Exercício 1
import numpy as np

def calcResult(A, B, C):
    return ((A.T@B) + B) @ (np.linalg.inv(C))
#1.1
A = np.full((3, 3), 8)
B = np.full((3, 4), 1)
C = np.eye(4)
print(calcResult(A, B, C))

#1.2
An = A[:2,:2]
Bn = B[:2,:2]
Cn = C[:2,:2]
print(calcResult(An, Bn, Cn))


# ## Questão 2
# **2A.** Escreva uma função python que recebe $m$ como entrada e executa os seguintes passos:
# 1. [ ] gera uma matriz aleatória $W_{m \times 4}$ (função **numpy.random.randn**),
# 2. [ ] divide cada uma das entradas por $\sqrt{m}$ (salva resultado em $\tilde W$),
# 3. [ ] calcula $Z = \tilde W^\top \times \tilde W$,
# 4. [ ] imprime $Z$,
# 5. [ ] calcula a norma Frobenius da diferença entre $Z$ e a matriz identidade $I_{4 \times 4}$.

# In[12]:


# Código para Exercício 2
import numpy as np
import math

def solveProblem(m):
    W = np.random.randn(m, 4)
    Wn = W / math.sqrt(m)
    Z = Wn.T @ Wn
    print(Z)
    froNorm = np.linalg.norm(Z - np.eye(4), 'fro')
    return froNorm
solveProblem(5)


# **2B.** Qual a norma da diferença obtida para $m=100$? E para $m=10000$?

# In[13]:


#Resposta:
print("\nPara m=100 a diferença é", solveProblem(100), ", e para m=10000 é", solveProblem(10000))


# **2C.** O que podemos dizer sobre a matriz $\tilde W$?

# Resposta:Wn é a matriz normalizada de W
