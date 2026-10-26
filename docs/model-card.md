# Model Card — Previsão de Churn

Documento descritivo do modelo de Regressão Logística e de seus limites. As
métricas agregadas e os diagnósticos por subgrupo estão registrados nos
metadados versionados do modelo. As métricas por subgrupo usam previsões
out-of-fold da validação aninhada; nenhum resultado é apresentado como
avaliação em teste independente.

## Detalhes do modelo

- **Nome:** Regressão Logística com pré-processamento em `Pipeline`.
- **Versão do artefato:** treinado em 26/10/2026; `models/churn_model.joblib`.
- **Implementação:** Scikit-Learn 1.9.1; Python 3.12.10.
- **Parâmetro selecionado:** `classifier__C = 10.0`.
- **Limiar de decisão:** 0,5. A predição é positiva quando a probabilidade de
  churn é maior ou igual ao limiar.
- **Classe positiva:** `Churn = Yes`.
- **Entradas:** 19 atributos de perfil, serviços, contrato e cobrança. O
  identificador `customerID` é removido.
- **Pré-processamento:** mediana e padronização para numéricas; imputação pela
  categoria mais frequente e one-hot para categóricas. Categorias desconhecidas
  são ignoradas na inferência.

## Uso pretendido

O modelo estima o risco de churn para apoiar a priorização de clientes pela
equipe de retenção. A probabilidade pode ajudar um profissional a organizar
análises e contatos.

**Fora de escopo:** decisões totalmente automatizadas, negação de serviços,
precificação individual, avaliação de crédito, inferência de causa, ou uso como
única base para decisões que afetem clientes. O modelo não foi validado para
uso operacional em uma empresa real.

## Fatores relevantes

O conjunto inclui `gender` e `SeniorCitizen`, atributos que podem estar
relacionados a grupos protegidos ou vulneráveis. A análise por grupos abaixo é
diagnóstica e não comprova equidade. O contrato também apresenta diferenças
expressivas de prevalência e de recall. Não foram definidos limites de
aceitação com o negócio nem conduzida auditoria jurídica ou de impacto.

## Métricas

O critério de seleção foi F1 para a classe churn; AUC-ROC complementa a
avaliação da ordenação dos riscos. Os valores são média ± desvio-padrão nos
cinco folds externos da validação cruzada aninhada, com limiar 0,5:

| Métrica | Resultado |
| --- | ---: |
| Acurácia | 0,8058 ± 0,0117 |
| Precisão | 0,6589 ± 0,0235 |
| Recall | 0,5548 ± 0,0336 |
| F1 | 0,6023 ± 0,0290 |
| AUC-ROC | 0,8450 ± 0,0149 |

A diferença de F1 em relação ao baseline de Regressão Logística com `C=1` é
0,0014; é pequena frente à variação entre folds e não demonstra uma melhora
confiável.

## Dados de avaliação

O protocolo de desenvolvimento foi validação cruzada aninhada estratificada:
cinco folds externos para estimar métricas e três folds internos para ajustar
hiperparâmetros. O pré-processamento foi ajustado dentro dos pipelines de cada
fold.

Não existe conjunto de teste independente disponível: o CSV completo já foi
consultado em análises e avaliações anteriores. Portanto, `test_metrics` e
`test_size` permanecem nulos. Os diagnósticos abaixo são **out-of-fold (OOF)**
nos folds externos, usando a configuração ajustada em cada fold. Eles não são
métricas de teste, não substituem avaliação prospectiva e podem refletir
otimismo de seleção entre famílias de modelos.

## Dados de treinamento

- **Fonte:** Telco Customer Churn, IBM Sample Data Sets, via
  [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
- **Registros usados no ajuste final:** 7.043.
- **Atributos preditores:** 19; o alvo `Churn` é convertido para 0/1.
- **Classe churn:** 26,54% na base analisada.
- **Nulos:** 11 valores de `TotalCharges`, convertidos para ausentes e imputados
  pela mediana no pipeline.
- **Limitação dos dados:** retrato estático e fictício, sem sequência temporal.

## Análises quantitativas por subgrupo

A tabela abaixo está registrada em `subgroup_metrics_oof` de
[`model_metadata.json`](../models/model_metadata.json). Foi calculada com
`src.evaluate.subgroup_metrics` usando as previsões OOF do campeão obtidas na
validação cruzada aninhada. Taxas são proporções; precisão, recall e F1 são
apresentados entre 0 e 1. Os grupos têm tamanhos diferentes e não representam
um teste independente.

| Atributo | Grupo | Clientes | Churn real | Churn previsto | Precisão | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gender | Female | 3.488 | 26,92% | 23,28% | 0,6576 | 0,5687 | 0,6099 |
| gender | Male | 3.555 | 26,16% | 21,41% | 0,6610 | 0,5409 | 0,5949 |
| SeniorCitizen | 0 | 5.901 | 23,61% | 18,28% | 0,6534 | 0,5061 | 0,5704 |
| SeniorCitizen | 1 | 1.142 | 41,68% | 43,26% | 0,6721 | 0,6975 | 0,6845 |
| Contract | Month-to-month | 3.875 | 42,71% | 40,52% | 0,6599 | 0,6260 | 0,6425 |
| Contract | One year | 1.473 | 11,27% | 0,20% | 0,3333 | 0,0060 | 0,0118 |
| Contract | Two year | 1.695 | 2,83% | 0,00% | 0,0000 | 0,0000 | 0,0000 |

As diferenças por `gender` são menores que as observadas por `SeniorCitizen` e
`Contract`, mas não bastam para declarar ausência de viés. O recall OOF para
`SeniorCitizen = 1` é maior que para `0`, junto com uma prevalência real maior.
Nos contratos de um e dois anos, o limiar fixo quase nunca produz positivos e o
recall é muito baixo ou zero. Isso mostra que o desempenho agregado esconde
comportamentos relevantes. Os resultados precisam ser reavaliados em novos
dados, com intervalos de incerteza e critérios acordados com o negócio; não se
deve alterar o limiar com base nesta análise exploratória.

## Considerações éticas

- A associação dos atributos com churn não identifica causas do cancelamento.
- `gender` e `SeniorCitizen` merecem avaliação de impacto e revisão humana
  antes de qualquer uso real.
- Uma oferta ou contato indevido pode causar incômodo e tratamento desigual.
- A equipe responsável deve poder contestar a recomendação e registrar
  decisões; não automatizar ações contra clientes.
- É necessária governança sobre acesso, retenção e proteção dos dados pessoais.

## Ressalvas e recomendações

1. Não interpretar validação cruzada ou os diagnósticos OOF como teste
   independente ou garantia de desempenho futuro.
2. Obter uma nova amostra rotulada, temporalmente posterior e não usada no
   desenvolvimento; congelar modelo e limiar antes de avaliá-la uma única vez.
3. Definir com negócio custos de falsos positivos/falsos negativos, horizonte
   do rótulo, metas por subgrupo e processo de revisão humana.
4. Monitorar distribuição de entradas, proporção de previsões, latência, erros
   e métricas quando os rótulos amadurecerem.
5. Planejar retreino somente com novos dados e repetir avaliação de desempenho
   e impacto.

## Referências internas

- [Comparação e validação](./comparacao-modelos.md)
- [ML Canvas](./ml-canvas.md)
- [`model_metadata.json`](../models/model_metadata.json)
- [`experimentos.csv`](./experimentos.csv)
