import pandas as pd
from fuzzywuzzy import fuzz
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
import time

start_time = time.time()

# Inicialização dos dataframes
df_amazon = pd.read_csv('amazon_cleaned.csv', encoding='utf-8')
df_google = pd.read_csv('google_cleaned.csv', encoding='utf-8')

# Função para blocagem flexível
def block_by_criteria(column, criteria_func):
    return column.apply(lambda x: criteria_func(x) if pd.notnull(x) else '')

# Função para comparar blocos
def compare_blocks(df1, df2, threshold=50):
    potential_duplicates = []
    for block_key in df1['block_key'].unique():
        block_amazon = df1[df1['block_key'] == block_key]
        block_google = df2[df2['block_key'] == block_key]
        for _, row_amazon in block_amazon.iterrows():
            for _, row_google in block_google.iterrows():
                similarity = fuzz.ratio(row_amazon['name'], row_google['name'])
                if similarity > threshold:
                    potential_duplicates.append((row_amazon['id'], row_google['id'], similarity))
    return pd.DataFrame(potential_duplicates, columns=['Amazon_ID', 'Google_ID', 'Similarity'])

# Blocagem flexível
df_amazon['block_key'] = block_by_criteria(df_amazon['name'], lambda x: x[0].upper())
df_google['block_key'] = block_by_criteria(df_google['name'], lambda x: x[0].upper())

# Comparação de blocos
duplicates = compare_blocks(df_amazon, df_google)

# Definição da preparação para classificação de registros potencialmente duplicados
duplicates['is_duplicate'] = duplicates['Similarity'].apply(lambda x: 1 if x > 80 else 0)

X = duplicates[['Similarity']]
y = duplicates['is_duplicate']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinamento do classificador
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Avaliação do modelo
predictions = clf.predict(X_test)

# Salvar os potenciais duplicados para análise futura
duplicates.to_csv('duplicadasEncontradas.csv', index=False)

end_time = time.time()
execution_time = end_time - start_time

#Carregamento das duplicatas 
gabarito_df = pd.read_csv('Amzon_GoogleProducts_perfectMapping.csv')
duplicatas_df = pd.read_csv('duplicadasEncontradas.csv')

#Identificar os verdadeiros positivos (TP)
tp = duplicatas_df[duplicatas_df['Amazon_ID'].isin(gabarito_df['idAmazon'])]

#Verdadeiros positivos (TP) e Falsos positivos (FP)
tp_count = len(tp)
fp_count = len(duplicatas_df) - tp_count

#Identificar os falsos negativos
fn_count = len(gabarito_df[~gabarito_df['idAmazon'].isin(duplicatas_df['Amazon_ID'])])


#Calcular as métricas 
precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Tempo de execução do algoritmo: {execution_time:.2f} segundos.")


