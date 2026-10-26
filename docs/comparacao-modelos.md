# Comparação de modelos de churn

## Integridade da avaliação

Na execução inicial, a avaliação do teste foi repetida após uma correção de código, e os resultados também ficaram visíveis. Embora não tenham orientado a escolha dos modelos, esse conjunto não pode mais ser considerado um teste cego independente. Como todos os registros do CSV participaram de análises anteriores, não existe neste dataset uma amostra de teste intocada. As métricas históricas de teste foram mantidas no CSV e nos notebooks, mas estão identificadas como exploratórias e não são usadas para escolher o campeão nem declaradas como resultado final.

A prática adotada nesta revisão é validação cruzada aninhada: cinco folds externos estimam o desempenho e três folds internos ajustam hiperparâmetros. Cada transformação é ajustada somente no treino do fold interno. Os resultados abaixo são estimativas de validação; escolher entre famílias de modelos pode ainda produzir otimismo de seleção. Uma avaliação prospectiva requer novos dados rotulados, coletados após o congelamento do pipeline.

## Protocolo e resultados da validação externa

O baseline fixo usa Regressão Logística com `C=1`; as demais entradas ajustam os hiperparâmetros nos folds internos. Valores apresentados como média ± desvio-padrão entre os folds externos.

| Modelo | Acurácia | Precisão | Recall | F1 | AUC-ROC | Tempo total (s) |
| --- | --- | --- | --- | --- | --- | --- |
| Regressão Logística | 0.8058 ± 0.0117 | 0.6589 ± 0.0235 | 0.5548 ± 0.0336 | 0.6023 ± 0.0290 | 0.8450 ± 0.0149 | 2.73 |
| Baseline — Regressão Logística (C=1) | 0.8052 ± 0.0125 | 0.6577 ± 0.0249 | 0.5532 ± 0.0355 | 0.6008 ± 0.0308 | 0.8450 ± 0.0149 | 0.26 |
| Random Forest | 0.8021 ± 0.0119 | 0.6628 ± 0.0278 | 0.5168 ± 0.0283 | 0.5807 ± 0.0279 | 0.8399 ± 0.0147 | 57.51 |
| MLP | 0.7997 ± 0.0086 | 0.6689 ± 0.0220 | 0.4853 ± 0.0215 | 0.5624 ± 0.0209 | 0.8407 ± 0.0145 | 19.07 |

## Campeão e artefato

Pelo maior F1 médio externo, o modelo selecionado é **Regressão Logística**. O pipeline salvo foi reajustado em todos os registros disponíveis, com parâmetros finais `{"classifier__C": 10.0}`. Não há métricas de teste finais (`test_metrics: null`), pois seria incorreto apresentar como independente um conjunto já consultado.
A diferença observada para o baseline fixo (C=1) na validação externa é +0.0014 em F1. Ela é pequena frente à variação entre folds; portanto, não afirmamos que o ganho seja estatisticamente confiável nem que o modelo supere o baseline de forma generalizável.

O pipeline completo, incluindo pré-processamento, está em `models/churn_model.joblib` (8.634 bytes). Os metadados em `models/model_metadata.json` registram o método aninhado, as versões e o estado da avaliação.

## Próxima avaliação

Antes de afirmar desempenho em produção, congele este pipeline e avalie-o uma única vez em novos dados rotulados que não tenham sido usados em exploração, ajuste ou seleção. Não escolha limiar nem hiperparâmetros com base nessa avaliação prospectiva.
