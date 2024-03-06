import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

"""
Carregamento das duplicatas 
"""
gabarito_df = pd.read_csv('Amzon_GoogleProducts_perfectMapping.csv')
duplicatas_df = pd.read_csv('duplicadasEncontradas.csv')

"""
Identificar os verdadeiros positivos (TP)
"""
tp = duplicatas_df[duplicatas_df['Amazon_ID'].isin(gabarito_df['idAmazon'])]

"""
Verdadeiros positivos (TP) e Falsos positivos (FP)
"""
tp_count = len(tp)
fp_count = len(duplicatas_df) - tp_count

"""
Identificar os falsos negativos
"""
fn_count = len(gabarito_df[~gabarito_df['idAmazon'].isin(duplicatas_df['Amazon_ID'])])


"""
Calcular as métricas 
"""
precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
