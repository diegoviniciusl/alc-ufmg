#!/usr/bin/env python
# coding: utf-8

# # Exercício de Programação 5: Regressão
# 
# <font color="red">**Prazo de submissão: 23:55 do dia 21/10/2020** </font>
# 
# 2020.1 Álgebra Linear Computacional - DCC - UFMG
# 
# Erickson - Fabricio
# 
# Instruções:
# * Antes de submeter suas soluções, certifique-se de que tudo roda como esperado. Primeiro, **reinicie o kernel** no menu, selecione Kernel$\rightarrow$Restart e então execute **todas as células** (no menu, Cell$\rightarrow$Run All)
# * Apenas o arquivo .ipynb deve ser submetido. Ele não deve ser compactado.
# * Não deixe de preencher seu nome e número de matrícula na célula a seguir
# 
# **Nome dos alunos e matrícula:**  
# Diego Vinicius de Oliveira Silva, 2019054471  <br>
# Vinícius Correia, 2019055044 <br>
# Rafael Lenti, 2019054951 <br>
# Lucca Silva, 2019054773 <br>

# ## Carregando os dados
# 
# Iremos carregar os dados usando a biblioteca ```pandas```. Não se preocupe se você não conhece a biblioteca, pois o nosso objetivo é apenas extrair a matriz de dados $X$. Segue uma descrição do dataset, retirada [daqui](http://statweb.stanford.edu/~owen/courses/202/Cereals.txt).
# 
# * Datafile Name: Cereals
# * Datafile Subjects: Food , Health
# * Story Names: Healthy Breakfast
# * Reference: Data available at many grocery stores
# * Authorization: free use
# * Description: Data on several variable of different brands of cereal.
# 
# A value of -1 for nutrients indicates a missing observation.
# Number of cases: 77
# Variable Names:
# 
#   1. Name: Name of cereal
#   2. mfr: Manufacturer of cereal where A = American Home Food Products; G =
#      General Mills; K = Kelloggs; N = Nabisco; P = Post; Q = Quaker Oats; R
#      = Ralston Purina
#   3. type: cold or hot
#   4. calories: calories per serving
#   5. protein: grams of protein
#   6. fat: grams of fat
#   7. sodium: milligrams of sodium
#   8. fiber: grams of dietary fiber
#   9. carbo: grams of complex carbohydrates
#   10. sugars: grams of sugars
#   11. potass: milligrams of potassium
#   12. vitamins: vitamins and minerals - 0, 25, or 100, indicating the typical percentage of FDA recommended
#   13. shelf: display shelf (1, 2, or 3, counting from the floor)
#   14. weight: weight in ounces of one serving
#   15. cups: number of cups in one serving
#   16. rating: a rating of the cereals

# In[1]:


#Execute esta célula para instalar o scikit learn e pandas caso já não tenha instalado
import sys
get_ipython().system('{sys.executable} -m pip install --user scikit-learn pandas')


# In[2]:


import pandas as pd
df = pd.read_table('cereal.txt',sep='\s+',index_col='name')
df


# A seguir iremos remover as linhas correspondentes aos cereais que possuem dados faltantes, representados pelo valor -1.
# Também iremos remover as colunas com dados categóricos 'mfr' e 'type', e os dados numéricos, 'shelf', 'weight' e 'cups'.

# In[3]:


import numpy as np
new_df = df.replace(-1,np.nan)
new_df = new_df.dropna()
new_df = new_df.drop(['mfr','type','shelf','weight','cups'],axis=1)
new_df


# Finalmente, iremos converter os dados nutricionais numéricos de ```new_df``` para uma matriz ```dados``` e as avaliações (ratings) para um vetor $y$. Os nomes dos cereais serão salvos em uma lista ```cereral_names``` e os nomes das colunas em uma lista ```col_names```.

# In[4]:


cereral_names = list(new_df.index)
print('Cereais:',cereral_names)
col_names = list(new_df.columns)
print('Colunas:',col_names)

dados = new_df.drop('rating', axis=1).values
print('As dimensões de dados são:',dados.shape)
y = new_df['rating'].values
print('As dimensões de y são:',y.shape)


# ## Estimando os parâmetros da regressão linear simples
# 
# Qual será a relação entre a avaliação $y$ e o número de calorias $x$ de um cereal? Para responder esta pergunta, considere uma regressão linear simples
# $$
# y = \beta_0 + \beta_1 x.
# $$
# Para encontrar os coeficientes $\beta_0$ e $\beta_1$ utilizando o método dos mínimos quadrados, basta resolver o sistema
# $$
# \begin{bmatrix}
# n & \sum_i x^{(i)} \\
# \sum_i x^{(i)} & \sum_i (x^{(i)})^2
# \end{bmatrix}
# \begin{bmatrix}
# \beta_0 \\ \beta_1
# \end{bmatrix}
# =
# \begin{bmatrix}
# \sum_i y^{(i)} \\ \sum_i x^{(i)} y^{(i)}
# \end{bmatrix}
# $$
# 
# Portanto, para encontrar $\beta_0$ e $\beta_1$, você precisa
# 1. Calcular a matriz
# $$
# A = \begin{bmatrix}
# n & \sum_i x^{(i)} \\
# \sum_i x^{(i)} & \sum_i (x^{(i)})^2
# \end{bmatrix}
# $$
# e o vetor
# $$
# c = \begin{bmatrix}
# \sum_i y^{(i)} \\ \sum_i x^{(i)} y^{(i)}
# \end{bmatrix}
# $$
# 2. Resolver $A \beta = c$, onde $\beta$ é o vetor de coeficientes.

# **Exercício 1 - Regressão simples:** Encontre os coeficientes $\beta_0$ e $\beta_1$ quando a variável independente é ```calories```. Dica: A variavel X abaixo já armazena os valores deste atributo.

# In[5]:


X = new_df['calories'].values
X = X.reshape(74, 1)
H = np.ones((74, 2))
H[:,-1:] = X
A = H.T @ H
C = H.T @ y
B = np.linalg.inv(A) @ C
print(B)


# **Exercício 2 - Regressão múltipla:** Considerando a nova tabela de dados X abaixo com os atributos 'calories', 'protein', 'fat', 'sugars' e 'vitamins' selecionados, estime os parâmetros da regressão múltipla para obter a variavel resposta ```rating``` 

# In[6]:


X = new_df.loc[:,['calories', 'protein', 'fat', 'sugars', 'vitamins']].values
H_mult = np.ones((74, 6))
H_mult[:,1:6] = X
A_mult = H_mult.T @ H_mult
C_mult = H_mult.T @ y
B_mult = np.linalg.inv(A_mult) @ C_mult
print(B_mult)


# **Exercício 3:** Nossos modelos de regressão linear são bons preditores da nota de avaliação do cereal? Qual o melhor modelo? Calcule os coeficientes de determinação e faça uma análise dos valores obtidos para responder a estas perguntas.

# In[7]:


y_media = np.average(y)
ones = np.ones(74)

y_chapeu = H @ B
r_denominator = np.linalg.norm(y - (y_media * ones), ord=2) ** 2

R = 1 - ((np.linalg.norm(y - y_chapeu, ord=2) ** 2) / r_denominator)

y_mult_chapeu = H_mult @ B_mult

R_mult = 1 - ((np.linalg.norm(y - y_mult_chapeu, ord=2) ** 2) / r_denominator)

print("Coeficiente do primeiro modelo:", R)
print("Coeficiente do segundo modelo:", R_mult)

print("\nO primeiro modelo que foi feito com base na regressão linear tem um coeficiente de determinação ruim (distante de 1), e por isso não pode ser considerado um bom preditor da avaliação do cereal.")
print("\nO segundo modelo, por outro lado, tem um coeficiente de determinação bom (próximo de 1), e por isso pode ser considerado um bom preditor para a avaliação do cereal.")
print("\nO melhor modelo é o segundo, feito pela regressão múltipla.")

