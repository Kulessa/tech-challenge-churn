# Roteiro do vídeo — Método STAR

**Duração prevista:** aproximadamente 5 minutos.  
**Ritmo:** fala natural, sem ler as instruções de tela.  
**Preparação:** deixar o repositório aberto, a API iniciada e o exemplo JSON
pronto antes de gravar.

## S — Situação (0:00–0:45)

**Mostrar:** título do projeto e seção de contexto no [ML Canvas](./ml-canvas.md).

**Fala:**

Olá! Neste projeto, trabalhei no desafio de prever churn em uma operadora.
Churn é o cancelamento do serviço. Identificar esse risco antes do pedido pode
ajudar a equipe de retenção a priorizar contatos, em vez de agir somente depois
que o cliente decide sair.

Usei o Telco Customer Churn, conjunto do Kaggle sobre uma empresa fictícia, com
7.043 clientes, 19 atributos e a indicação de cancelamento. É um retrato
estático, sem histórico temporal e sem garantia de representar uma operação
real. O projeto demonstra um fluxo técnico; não promete reduzir churn em
produção.

## T — Tarefa (0:45–1:30)

**Mostrar:** tabela de critérios no README e a tabela de resultados da comparação.

**Fala:**

Minha tarefa foi construir um processo reproduzível, comparar modelos de
classificação e disponibilizar o campeão por uma API. Usei o F1 da classe
positiva, churn, como critério principal, porque preciso considerar tanto os
clientes encontrados quanto os falsos alertas. Também acompanhei a AUC-ROC,
precisão e recall.

As entregas incluem análise exploratória, baseline, comparação de modelos,
pipeline de treinamento, testes automatizados, documentação e os endpoints
health e predict. A semente aleatória é 42. Todo o pré-processamento fica
dentro do pipeline para que seja ajustado junto com o modelo, sem vazar
informação entre os folds.

## A — Ação (1:30–3:45)

**Mostrar:** EDA, notebook de comparação, depois a estrutura `src/` e os testes.

**Fala:**

Primeiro fiz a análise exploratória. Verifiquei tipos, ausências,
duplicidades, distribuição do alvo e relações entre os atributos. A coluna
TotalCharges veio como texto e foi convertida para número; os valores em
branco viraram ausentes e são imputados no pipeline. Removi customerID, porque
é apenas um identificador, e converti churn para zero e um.

Depois comparei Regressão Logística, Random Forest e MLP. Para separar a
seleção dos hiperparâmetros da estimativa de desempenho, usei validação
cruzada aninhada: cinco folds externos e três internos. O pré-processamento é
ajustado dentro dos pipelines de cada fold. Isso fornece uma estimativa de
desenvolvimento mais cuidadosa do que medir o modelo nos mesmos dados usados
para ajustá-lo.

Há uma ressalva importante: o conjunto inteiro já tinha sido consultado em
etapas anteriores, então não existe aqui um teste independente. Por isso,
mantenho os resultados históricos de teste identificados como exploratórios e
os metadados deixam as métricas de teste nulas. A comparação indicou a
Regressão Logística com C igual a 10 como campeã por F1 médio. A diferença
para o baseline de C igual a 1 é pequena, então não afirmo que seja uma
melhoria confiável ou que vá se repetir em novos clientes.

Em seguida organizei o código de produção em módulos: configuração,
carregamento e limpeza dos dados, pré-processamento, treinamento, predição e
API. A API valida os campos recebidos com Pydantic, carrega o pipeline no
início e oferece os endpoints health e predict. Se o modelo não estiver
disponível, a predição responde com erro explícito, em vez de retornar um
resultado fictício.

Também criei testes independentes do CSV para conversão e transformação dos
dados, categorias novas, coerência da probabilidade e da classe, validação de
entrada e ciclo de vida dos endpoints. A documentação explica a instalação, o
treinamento, a chamada da API, o uso pretendido e os limites do modelo.

## R — Resultado (3:45–5:00)

**Mostrar:** README e Model Card; abrir `/docs`; chamar `/health` e enviar o
JSON de exemplo para `/predict`.

**Fala:**

Nos cinco folds externos, a Regressão Logística selecionada obteve F1 médio de
0,6023, com desvio de 0,0290, e AUC-ROC média de 0,8450, com desvio de 0,0149.
O baseline fixo teve F1 médio de 0,6008. Esses valores são estimativas de
validação cruzada, não resultados de um teste cego nem garantia de
desempenho futuro.

Na análise diagnóstica out-of-fold por contrato, observei recall muito baixo
para os contratos anuais e bienais, apesar do resultado agregado. Isso reforça
a necessidade de medir diferenças entre grupos e de revisar qualquer ação com
uma pessoa. Os dados são fictícios, a associação não prova causalidade e uma
avaliação prospectiva depende de novos rótulos.

Agora a API informa que está saudável e que o modelo foi carregado. Na
requisição de exemplo, ela devolve uma probabilidade, a decisão pelo limiar,
o próprio limiar e a versão do artefato. Esse resultado deve priorizar uma
análise humana, não disparar uma oferta automaticamente. Como próximos passos,
eu buscaria dados novos, congelaria o pipeline antes da avaliação e
combinaria com o negócio metas, custos e monitoramento.

## Lista de verificação da gravação

- [ ] Gravar em 1080p, com fonte legível e áudio claro.
- [ ] Fechar notificações e deixar abertas as pastas `notebooks/`, `src/`,
  `tests/`, `models/` e `docs/`.
- [ ] Ativar o ambiente virtual e iniciar a API antes da gravação.
- [ ] Confirmar que `/docs`, `/health` e `docs/exemplo_predict.json` estão
  acessíveis.
- [ ] Ensaiar transições e manter a duração total em até cinco minutos.
