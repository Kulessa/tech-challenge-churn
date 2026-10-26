# Previsão de Churn em Telecomunicações

Projeto individual de Machine Learning Engineering que compara modelos para
identificar clientes de telecomunicações com maior risco de cancelamento
(*churn*) e disponibiliza o pipeline selecionado por uma API REST.

## Contexto e resultado

O conjunto Telco Customer Churn contém 7.043 registros de uma empresa fictícia.
O alvo é `Churn` (`Yes` ou `No`), e a classe positiva é o cancelamento (`Yes`).
O pipeline faz imputação e padronização das variáveis numéricas, codificação
one-hot das categóricas e classificação por Regressão Logística.

A comparação selecionou a Regressão Logística ajustada (`C=10`) pelo maior F1
médio na validação cruzada aninhada. Os valores abaixo são médias ± desvios
padrão dos cinco folds externos; não são métricas de um teste independente.

| Modelo | F1 | AUC-ROC | Precisão | Recall |
| --- | ---: | ---: | ---: | ---: |
| Regressão Logística (`C=10`) | 0,6023 ± 0,0290 | 0,8450 ± 0,0149 | 0,6589 ± 0,0235 | 0,5548 ± 0,0336 |
| Baseline — Regressão Logística (`C=1`) | 0,6008 ± 0,0308 | 0,8450 ± 0,0149 | 0,6577 ± 0,0249 | 0,5532 ± 0,0355 |
| Random Forest | 0,5807 ± 0,0279 | 0,8399 ± 0,0147 | 0,6628 ± 0,0278 | 0,5168 ± 0,0283 |
| MLP | 0,5624 ± 0,0209 | 0,8407 ± 0,0145 | 0,6689 ± 0,0220 | 0,4853 ± 0,0215 |

O ganho de F1 sobre o baseline fixo é pequeno (0,0014) frente à variação entre
folds; não há evidência de melhoria confiável. O CSV já foi explorado durante o
desenvolvimento, portanto não existe teste independente disponível. As métricas
históricas de teste são exploratórias e não devem ser usadas como resultado
final. Uma avaliação prospectiva exige novos dados rotulados.

## Estrutura

```text
data/       CSV local (não versionado)
docs/       ML Canvas, comparação, Model Card, roteiro e exemplo da API
models/     pipeline scikit-learn e metadados versionados
notebooks/  EDA, baseline e comparação experimental
src/        configuração, dados, pré-processamento, treino, avaliação e API
tests/      testes automatizados
```

## Pré-requisitos e instalação

Use Python **3.12.10** (a série 3.12). As dependências diretas estão fixadas em
[`requirements.txt`](./requirements.txt), incluindo a versão de
scikit-learn necessária para carregar o artefato `.joblib`.

### Windows (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Linux ou macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Obter os dados

Baixe o conjunto [Telco Customer Churn no Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
Coloque o CSV em `data/telco_churn.csv`. O dataset não é incluído no Git; o
arquivo precisa estar presente para executar os notebooks ou treinar o modelo.

## Notebooks

Abra `notebooks/01_eda.ipynb`, `notebooks/02_baseline.ipynb` ou
`notebooks/03_model_comparison.ipynb` no VS Code, selecione o interpretador da
pasta `.venv` e execute as células. Os notebooks são experimentais; a lógica
usada pela API fica em `src/`.

## Treinar o pipeline

Com o CSV instalado no caminho indicado:

```powershell
python -m src.train
```

O comando ajusta o pipeline selecionado em todos os registros disponíveis e
atualiza `models/churn_model.joblib` e `models/model_metadata.json`. Não calcula
métricas de teste. Os resultados de validação cruzada já registrados nos
metadados são estimativas de desenvolvimento e não uma avaliação independente.
O pipeline de inferência inclui o pré-processamento para evitar divergência
entre treino e predição.

## Executar a API

Na raiz do projeto, ative o ambiente virtual e inicie:

```powershell
python -m uvicorn src.api.main:app --reload
```

O Swagger fica em <http://127.0.0.1:8000/docs>.

Verificação de saúde no PowerShell:

```powershell
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:8000/health
```

Predição usando o exemplo versionado:

```powershell
$body = Get-Content -Raw -Encoding UTF8 .\docs\exemplo_predict.json
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/predict `
  -ContentType "application/json" `
  -Body $body
```

As mesmas chamadas com `curl`:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  --data-binary "@docs/exemplo_predict.json"
```

`/predict` recebe os 19 atributos em JSON e retorna `churn_probability`,
`churn_prediction`, `threshold` e `model_version`. Entrada inválida recebe HTTP
422; se o artefato não estiver disponível, a API responde HTTP 503.

## Testes

```powershell
python -m pytest -q
```

Os testes usam dados sintéticos e não precisam do CSV para validar
pré-processamento, predição e endpoints.

## Reprodutibilidade, limitações e uso responsável

- Seed fixa: `RANDOM_STATE = 42`; validação aninhada estratificada com cinco
  folds externos e três internos.
- A avaliação usa um retrato estático de uma empresa fictícia, sem dimensão
  temporal, e não representa necessariamente clientes reais.
- A seleção de modelo pode introduzir otimismo; validação cruzada não substitui
  uma avaliação prospectiva em novos dados.
- Desbalanceamento, correlação sem causalidade e possíveis diferenças entre
  subgrupos limitam a interpretação.
- A probabilidade indica risco, não explica a causa do cancelamento. Use a
  saída como apoio à decisão humana; não automatize ofertas ou restrições ao
  cliente apenas com base no modelo.
- Defina monitoramento de dados, desempenho, latência e erros; planeje retreino
  apenas quando houver novos rótulos e um protocolo de avaliação adequado.

## Documentação

- [ML Canvas](./docs/ml-canvas.md)
- [Comparação de modelos](./docs/comparacao-modelos.md)
- [Model Card](./docs/model-card.md)
- [Roteiro do vídeo STAR](./docs/roteiro-video-star.md)
- [Exemplo de requisição](./docs/exemplo_predict.json)
