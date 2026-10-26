# ML Canvas: Previsão de Churn em Telecomunicações

> **Projeto:** Tech Challenge, Fase 1 (Machine Learning Engineering)
> **Versão:** 1.0 (26/10/2026)
>
> Este documento segue o *Machine Learning Canvas* (10 blocos), acrescido das seções de **stakeholders** e **métricas de negócio** exigidas pela Etapa 1 do desafio.
> Valores marcados como *(hipótese)* são premissas do projeto, a validar com o negócio. Os números do dataset devem ser reconfirmados na EDA (`notebooks/01_eda.ipynb`).

---

## 1. Proposta de valor

- **Problema:** uma operadora de telecomunicações está perdendo clientes em ritmo acelerado. A diretoria precisa saber, antes do cancelamento, quais clientes têm maior risco de sair.
- **Usuário final das previsões:** equipe de retenção / CRM.
- **Valor entregue:** em vez de aplicar ações de retenção em toda a base, a equipe concentra esforço e orçamento nos clientes de maior risco.
- **Objetivo de negócio (hipótese):** reduzir a taxa de churn em 15% (meta ilustrativa, inspirada no exemplo da Aula 1 do curso), a validar com a diretoria.
- **Fluxo de uso:**
  1. Os dados do cliente são enviados à API de inferência.
  2. O modelo devolve a propensão de churn (probabilidade entre 0 e 1).
  3. A equipe de retenção recebe a lista de clientes ordenada pelo risco.
  4. É feita uma ação de retenção (contato, oferta, mudança de plano).
  5. O resultado é acompanhado nos meses seguintes.

---

## 2. Stakeholders

| Stakeholder    | Papel            | Interesse / expectativa                        |
| -------------- | ---------------- | ---------------------------------------------- |
| Diretoria      | Patrocinadora    | Reduzir a perda de clientes e preservar        |
|                |                  | receita                                        |
| Equipe de      | Usuária direta   | Lista priorizada, clara e confiável de         |
| retenção       |                  |                                                |
| / CRM          | das previsões    | clientes em risco                              |
| Marketing e    | Aprovam          | Custo das campanhas e retorno sobre o          |
| Finanças       | orçamento e      | investimento                                   |
|                | ofertas          |                                                |
| Atendimento ao | Executa          | Saber o motivo provável do risco para abordaro |
| cliente        | contatos         | cliente                                        |
| Clientes       | Afetados pelas   | Receber ofertas relevantes, sem abordagem      |
|                | ações            | indevida                                       |
| Cientista de   | Constrói e       | Modelo reprodutível, avaliado e                |
| dados          |                  |                                                |
| / ML (autor)   | documenta o      | documentado                                    |
|                | modelo           |                                                |
| TI /           | Opera a API      | Serviço estável, testado e simples de executar |
| Engenharia     |                  |                                                |

---

## 3. Métricas de negócio

- **Métrica principal:** taxa de churn = clientes que cancelaram no período ÷ clientes ativos no início do período. Meta ilustrativa *(hipótese)*: redução relativa de 15%.
- **Métricas de apoio:**
  - número de clientes em risco efetivamente retidos;
  - receita mensal preservada (soma de `MonthlyCharges` dos clientes retidos);
  - custo total das ações de retenção;
  - retorno sobre o investimento (ROI) da campanha.
- **Valor líquido estimado da campanha *(hipótese)*:**

  `valor = VP × taxa_retenção × receita_mensal_média × meses_horizonte − (VP + FP) × custo_da_oferta`

  Onde **VP** são os clientes sinalizados que realmente cancelariam e **FP** são os sinalizados que não cancelariam (oferta desperdiçada).
- **Parâmetros a definir com o negócio:** taxa de sucesso da retenção, custo médio da oferta e horizonte de receita preservada.

---

## 4. Tarefa de ML (prediction task)

- **Tipo de problema:** classificação binária supervisionada.
- **Entrada:** atributos de um cliente (perfil, contrato, serviços contratados e cobrança).
- **Saída:** probabilidade de churn e a classe prevista (`Churn = Yes` ou `No`), com limiar de decisão ajustável.
- **Alvo (`Churn`):** indica se o cliente cancelou o serviço no período observado.
- **Classe positiva:** churn (`Yes`).

---

## 5. Decisões

- **Como as previsões geram valor:** clientes com probabilidade acima do limiar entram na lista de abordagem da equipe de retenção, ordenada pelo risco.
- **Limiar de decisão:** começa em 0,5 e é ajustado conforme o custo da oferta e o orçamento disponível. Limiar menor aumenta o *recall* (mais churners encontrados) e também os falsos positivos.
- **Humano no processo:** a equipe de retenção decide a oferta. O modelo indica o risco, não a causa do cancelamento, e não deve acionar ofertas de forma totalmente automática.

---

## 6. Fontes de dados

- **Fonte:** *Telco Customer Churn* (IBM Sample Data Sets), versão do Kaggle: <https://www.kaggle.com/datasets/blastchar/telco-customer-churn>.
- **Conteúdo:** dados de uma empresa fictícia de telefonia e internet residencial, com 7.043 clientes (Califórnia, 3º trimestre) e 21 colunas: `customerID`, 19 atributos e o alvo `Churn`.
- **Arquivo local:** `data/telco_churn.csv`, que não é versionado no Git (`.gitignore`).
- **Observação:** é um retrato único no tempo (sem histórico por data) e não representa uma base real.

---

## 7. Coleta de dados

- **Neste projeto:** os dados já vêm rotulados em um único arquivo; não há coleta contínua.
- **Em um cenário real:**
  - os atributos viriam dos sistemas de cadastro, faturamento e contratação de serviços;
  - o rótulo (cancelou ou não) só é conhecido depois de um intervalo de tempo, então é preciso definir o horizonte de previsão (por exemplo, cancelamento nos próximos 30 dias);
  - os rótulos novos alimentam o retreinamento do modelo.

---

## 8. Atributos (features)

Os 19 atributos de entrada, agrupados:

| Grupo          | Atributos                                                         |
| -------------- | ----------------------------------------------------------------- |
| Perfil do      | `gender` , `SeniorCitizen` , `Partner` , `Dependents`             |
| cliente (4)    |                                                                   |
| Conta          | `tenure` , `Contract` , `PaperlessBilling` , `PaymentMethod` ,    |
| e cobra nça    | `MonthlyCharges` , `TotalCharges`                                 |
| (6)            |                                                                   |
| Servi          | `PhoneService` , `MultipleLines` , `InternetService` ,            |
|                | `OnlineSecurity` ,                                                |
| ços            | `OnlineBackup` , `DeviceProtection` , `TechSupport` ,             |
|                | `StreamingTV` ,                                                   |
| contr atados   | `StreamingMovies`                                                 |
| (9)            |                                                                   |

**Tratamento previsto (dentro de um `Pipeline` do Scikit-Learn, para evitar vazamento de dados):**

- remover `customerID` (identificador, sem poder preditivo);
- converter `TotalCharges` para número; as linhas com valor em branco viram nulos e são imputadas;
- variáveis numéricas (`tenure`, `MonthlyCharges`, `TotalCharges`): imputação pela mediana e padronização;
- variáveis categóricas: codificação *one-hot*, ignorando categorias desconhecidas na predição;
- converter o alvo `Churn` para 0/1.

---

## 9. Construção de modelos

- **Modelos comparados:**
  1. Regressão Logística (baseline);
  2. Random Forest (árvores / *ensemble*);
  3. Rede neural simples (`MLPClassifier`).
- **Reprodutibilidade:** semente fixa (`random_state = 42`) em divisão de dados, validação cruzada e modelos.
- **Modelo final:** o campeão da comparação é salvo em `models/` (formato `.joblib`) junto com o pré-processamento.
- **Atualização *(hipótese)*:** retreinar periodicamente (por exemplo, a cada trimestre) ou quando o monitoramento indicar queda de desempenho. Em um cenário real, o tempo de treino não é restrição: os dados são tabulares e pequenos.

---

## 10. Geração de previsões

- **Modo:** online, por requisição, por meio de uma API REST com FastAPI.
  - `GET /health`: confirma que a API está no ar.
  - `POST /predict`: recebe os dados de um cliente e devolve a propensão de churn.
- **Também seria possível** executar em lote sobre toda a base de clientes (por exemplo, uma vez por semana), alimentando a lista de abordagem da equipe de retenção.
- **Latência:** não é crítica; a previsão pode levar alguns segundos sem afetar o uso.
- **Validação de entrada:** os dados recebidos são validados antes da predição (tipos e valores permitidos).

---

## 11. Avaliação offline (métrica técnica)

**Métricas técnicas:**

- **Principal (escolha do campeão): F1 da classe churn.** Equilibra precisão (evitar ofertas desperdiçadas) e *recall* (encontrar quem realmente vai sair).
- **Complementar: AUC-ROC.** Mede a capacidade de ordenar clientes por risco, independentemente do limiar.
- **Também reportadas:** precisão, *recall*, matriz de confusão e acurácia (apenas informativa).

**Por que não usar só a acurácia:** o alvo é desbalanceado. Segundo análises públicas do dataset, são cerca de 26,5% de churn (1.869 de 7.043 clientes) e 73,5% de não churn (5.174). Um modelo que sempre previsse "não churn" teria acurácia próxima de 73,5% sem encontrar nenhum cliente em risco (F1 da classe churn igual a zero). Esses números devem ser reconfirmados na EDA.

**Protocolo realizado e integridade da avaliação:**

1. O conjunto foi consultado em análises e avaliações anteriores, incluindo repetição da avaliação de teste. Esses resultados históricos são exploratórios; não há mais um teste cego independente neste CSV.
2. A comparação atual usa validação cruzada aninhada estratificada: 5 folds externos para estimar métricas e 3 folds internos para ajustar hiperparâmetros, com `random_state = 42`.
3. O pré-processamento é ajustado dentro do `Pipeline` de cada fold, somente nos dados de treino daquele fold.
4. A comparação inclui Regressão Logística, Random Forest e MLP. O critério de seleção é maior F1 médio externo; AUC-ROC, simplicidade e custo computacional complementam a análise.
5. As métricas agregadas são estimativas de desenvolvimento. A diferença de F1 do campeão para o baseline fixo é apenas +0,0014, pequena frente à variação entre folds; não se afirma uma melhoria confiável.
6. `test_metrics` e `test_size` permanecem nulos. Uma avaliação final exige dados novos e rotulados, não usados na exploração, seleção ou treinamento; o pipeline deve ser congelado antes dessa avaliação.

**Referências mínimas:** comparar com a Regressão Logística baseline e com o classificador que sempre prevê "não churn". A comparação informa diferenças observadas, mas não exige declarar superioridade quando a evidência não sustenta essa conclusão.

---

## 12. Avaliação em produção e monitoramento

*(Fora do escopo de implementação do desafio; descrito para completar o canvas.)*

- **Atraso dos rótulos:** o cancelamento só é confirmado depois de um tempo, então o desempenho real é medido com atraso.
- **O que acompanhar:**
  - distribuição dos atributos de entrada (por exemplo, `tenure`, `MonthlyCharges` e proporção de cada tipo de `Contract`), para detectar mudança nos dados (*data drift*);
  - taxa de churn prevista versus a observada;
  - F1, *recall* e AUC-ROC quando os rótulos chegarem;
  - latência e taxa de erro da API.
- **Impacto de negócio:** comparar um grupo que recebe a ação de retenção com um grupo de controle, para medir o efeito real da campanha.
- **Gatilho de retreino:** queda de desempenho abaixo de um limite definido ou mudança relevante na distribuição dos dados.

---

## 13. Riscos, limitações e premissas

- **Dataset fictício e estático:** é um retrato único no tempo, de uma empresa fictícia; o desempenho em uma operação real pode ser diferente.
- **Sem teste independente:** todos os registros disponíveis já participaram da exploração ou avaliação; os resultados históricos de teste não são finais. A validação cruzada aninhada não substitui uma avaliação prospectiva em novos dados.
- **Desbalanceamento de classes:** exige métricas além da acurácia; técnicas como ajuste de pesos ou de limiar podem ser avaliadas.
- **Atributos sensíveis:** `gender` e `SeniorCitizen` podem gerar tratamento desigual entre grupos. O efeito deve ser avaliado e registrado no *Model Card*.
- **`TotalCharges`:** 11 registros têm valor em branco; verificar na EDA se coincidem com `tenure = 0` e documentar o tratamento escolhido.
- **Correlação não é causalidade:** o modelo indica risco, não o motivo do cancelamento.
- **Uso responsável:** as ofertas devem ser decididas por pessoas, com o modelo como apoio, e o custo da ação deve ser menor que a receita preservada.
