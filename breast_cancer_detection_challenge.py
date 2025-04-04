# -*- coding: utf-8 -*-
"""Breast Cancer Detection Challenge

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1wRbWHOiGLF2KpIRfkEu6rFcsiEA1xtkN
"""

# Importando as bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
import warnings
warnings.filterwarnings('ignore')

# Configurando o estilo das visualizações
plt.style.use('ggplot')
sns.set(style='whitegrid')

# 1. Carregando os dados
df = pd.read_csv('breast_cancer.csv')

# 2. Analisando os dados
print("Informações do dataset:")
print(df.info())
print("\nEstatísticas descritivas:")
print(df.describe())

# Verificando a distribuição das classes
print("\nDistribuição das classes:")
print(df['Classification'].value_counts())

# 3. Verificando dados ausentes
print("\nVerificando dados ausentes:")
print(df.isnull().sum())

# 4. Verificando dados duplicados
print("\nNúmero de entradas duplicadas:", df.duplicated().sum())

# Removendo as entradas duplicadas
df_clean = df.drop_duplicates()
print("Número de entradas após remoção de duplicatas:", len(df_clean))
print("Distribuição das classes após remoção de duplicatas:")
print(df_clean['Classification'].value_counts())

# 5. Análise exploratória dos dados
# Criando uma matriz de correlação
plt.figure(figsize=(12, 10))
correlation_matrix = df_clean.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlação')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
#plt.close()

# Comparando as distribuições das variáveis entre as classes
features = ['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA', 'Leptin', 'Adiponectin', 'Resistin', 'MCP.1']

plt.figure(figsize=(15, 20))
for i, feature in enumerate(features):
    plt.subplot(5, 2, i+1)
    sns.boxplot(x='Classification', y=feature, data=df_clean)
    plt.title(f'Distribuição de {feature} por Classe')
plt.tight_layout()
plt.savefig('feature_distributions.png')
#plt.close()

# 6. Preparando os dados para o treinamento
X = df_clean.drop('Classification', axis=1)
y = df_clean['Classification']

# 7. Balanceamento dos dados
# Aplicando RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_rus, y_rus = rus.fit_resample(X, y)

# Aplicando TomekLinks
tl = TomekLinks(sampling_strategy='all')
X_resampled, y_resampled = tl.fit_resample(X_rus, y_rus)

print("\nDistribuição das classes após balanceamento:")
print(pd.Series(y_resampled).value_counts())

# 8. Dividindo os dados em conjuntos de treino e teste (70% treino, 30% teste)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# 9. Treinando o modelo Random Forest
rf_model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=100)
rf_model.fit(X_train, y_train)

# 10. Fazendo previsões
y_pred = rf_model.predict(X_test)

# 11. Avaliando o modelo
print("\nMatrix de Confusão:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=['Saudável', 'Câncer'],
            yticklabels=['Saudável', 'Câncer'])
plt.xlabel('Previsão')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.savefig('confusion_matrix.png')
#plt.close()

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

print("\nAcurácia do modelo:", accuracy_score(y_test, y_pred))

# 12. Analisando a importância das características
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nImportância das características:")
print(feature_importance)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Importância das Características')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Filtrando apenas os casos diagnosticados com câncer
cancer_cases = df_clean[df_clean['Classification'] == 1]

# Calculando a média de glicose para esses casos
mean_glucose = cancer_cases['Glucose'].mean()

print(f"Média de glicose entre os diagnosticados com câncer: {mean_glucose:.2f}")

# Filtrando os casos diagnosticados com câncer
cancer_cases = df_clean[df_clean['Classification'] == 1]

# Criando o histograma
plt.figure(figsize=(10, 6))
sns.histplot(cancer_cases['Age'], bins=20, kde=True, color='red')

# Configurações do gráfico
plt.title('Distribuição das Idades entre Diagnosticados com Câncer')
plt.xlabel('Idade')
plt.ylabel('Frequência')
plt.grid(True)

# Salvando e exibindo o gráfico
plt.savefig('histograma_idades_cancer.png')
plt.show()

# Criando o gráfico de dispersão
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_clean, x='Glucose', y='BMI', hue='Classification', palette='coolwarm', alpha=0.7)

# Configurações do gráfico
plt.title('Relação entre Glucose e BMI agrupados por Classification')
plt.xlabel('Glucose')
plt.ylabel('BMI')
plt.legend(title='Classification', labels=['Saudável', 'Câncer'])
plt.grid(True)

# Salvando e exibindo o gráfico
plt.savefig('scatter_glucose_bmi.png')
plt.show()

# Criando o boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_clean, x='Classification', y='Glucose', palette='coolwarm')

# Configurações do gráfico
plt.title('Distribuição da Glicose por Classificação')
plt.xlabel('Classificação')
plt.ylabel('Glucose')
plt.xticks(ticks=[0, 1], labels=['Saudável', 'Câncer'])
plt.grid(True, linestyle='--', alpha=0.5)

# Salvando e exibindo o gráfico
plt.savefig('boxplot_glucose_classification.png')
plt.show()