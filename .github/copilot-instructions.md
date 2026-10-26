# Instruções do projeto: Tech Challenge Churn

## Contexto
Pipeline preditivo de churn (telecom) com Scikit-Learn e FastAPI. Projeto individual, Fase 1 de ML Engineering.

## Regras gerais
- Responder sempre em português do Brasil.
- Código, nomes de variáveis e docstrings em inglês, seguindo PEP 8 (snake_case, classes em CamelCase).
- Usar type hints e docstrings em todas as funções públicas.
- Fixar seed (RANDOM_STATE = 42) em todo treino, split e validação cruzada.
- Não explicar além do que foi pedido.

## Estrutura
- `notebooks/`: apenas experimentação (EDA, baseline, comparação).
- `src/`: código produtivo em módulos .py (pré-processamento, treino, predição, API).
- `tests/`: testes com pytest.
- `models/`: modelos salvos (.joblib).
- `data/`: dados (não versionar arquivos grandes).
- `docs/`: ML Canvas, Model Card, roteiro do vídeo.

## Stack obrigatória
- Scikit-Learn: Pipeline, ColumnTransformer, LogisticRegression, RandomForestClassifier, MLPClassifier.
- FastAPI com endpoints `/health` e `/predict`; validação de entrada com Pydantic.
- Pytest com no mínimo: um teste de pré-processamento e um teste de status da API.

## Modelagem
- Métrica técnica principal: F1 e AUC-ROC. Classe positiva = churn.
- Comparar sempre baseline (Regressão Logística) x árvore/ensemble x MLP com validação cruzada estratificada.
- Nunca usar dados de teste para ajustar pré-processamento (evitar data leakage).

## Commits
- Conventional Commits (feat:, fix:, docs:, test:, refactor:, chore:).
