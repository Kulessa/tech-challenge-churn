---
name: churn-model-review
description: Revisar e comparar modelos de churn (Regressão Logística, Random Forest, MLP) com validação cruzada e gerar tabela comparativa. Use ao treinar, avaliar ou escolher o modelo campeão.
---

# Revisão de modelos de churn

1. Confirmar que o pré-processamento está dentro de um Pipeline (sem data leakage).
2. Usar StratifiedKFold (5 folds) com random_state=42.
3. Calcular accuracy, precision, recall, F1 e AUC-ROC por modelo.
4. Gerar tabela comparativa (modelo, métricas, tempo de treino) em docs/.
5. Justificar o modelo campeão pela métrica definida no ML Canvas.
