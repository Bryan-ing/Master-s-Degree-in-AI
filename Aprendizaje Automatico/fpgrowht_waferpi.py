import os
os.environ["LOKY_MAX_CPU_COUNT"] = "2"

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

# 1. Define sample transaction data
dataset = [
    ['EdgeRing', 'Scratch', 'Center', 'Donut'],
    ['EdgeRing', 'Scratch', 'Center'],
    ['EdgeRing', 'Scratch'],
    ['Scratch', 'Center', 'Donut'],
    ['EdgeRing', 'Center', 'Donut']
]
#cada lista es un wafer distinto, y contiene los tipos de defecto detectados en ese wafer
#(en vez de "productos comprados", aquí son "tipos de defecto presentes en ese wafer")

# 2. Transform the transaction list into a one-hot encoded boolean DataFrame
te = TransactionEncoder() #transforma los 5 wafers (listas) en una tabla donde las
#columnas son tipos de defecto y cada fila un wafer con True/False
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

# 3. Mine frequent itemsets (Minimum Support Threshold = 40%)
frequent_itemsets = fpgrowth(df, min_support=0.4, use_colnames=True) #el min_support=0.4 le dice
#que porcentaje de todos los wafers contiene cierta combinación de defectos, le dice solo me interesa
#combinaciones de defectos que aparezcan en al menos 40% de los wafers.
#el fpgrowth es solo el nombre del algoritmo eficiente que usa para encontrar esas combinaciones
#sin tener que probar todas las combinaciones posibles una por una

# 4. Generate association rules (Minimum Confidence Threshold = 70%)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
#una vez que conoces que combinaciones de defectos son frecuentes filtra con reglas de tipo
#si aparece el defecto A entonces aparece el defecto B en el mismo wafer
#y filtra solo las reglas donde la confianza(confidence) sea de al menos 70%

# Print specific results
print("--- Frequent Itemsets ---")
print(frequent_itemsets)
print("\n--- Association Rules ---")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])