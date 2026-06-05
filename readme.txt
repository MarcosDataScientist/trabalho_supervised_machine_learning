Atividade Pratica - Classificacao de Sentimentos e Analise Comparativa

Disciplina: Aprendizado de Maquina Supervisionado (UEL - CCE - Departamento de Computacao)

Objetivo
- Desenvolver uma solucao em Python (via Jupyter Notebook) para classificar avaliacoes de produtos em sentimentos Positivo ou Negativo.
- Comparar duas abordagens para o mesmo problema:
  1) Baseline tradicional com TF-IDF + classificador do scikit-learn
  2) Abordagem com Transformer pre-treinado (ex.: BERTimbau ou DistilBERT)

Abordagens obrigatorias
1. Abordagem Tradicional (Baseline)
- Vetorizacao: TF-IDF
- Classificador: Regressao Logistica, SVM ou Naive Bayes (scikit-learn)

2. Abordagem com Deep Learning (Transformer)
- Modelo: BERTimbau (recomendado para portugues) ou DistilBERT (mais rapido)
- Estrategia: Fine-tuning do modelo para classificacao de texto
  (ou extracao de features, se for a estrategia escolhida e justificada)

Dataset e divisao de dados
- Utilizar dataset publico em portugues (ex.: B2W-Reviews01, Buscape, IMDb traduzido)
- Dividir estritamente em:
  - Treino
  - Validacao
  - Teste

Pre-processamento (com justificativa)
- Explicar e justificar as etapas aplicadas na abordagem tradicional
  (ex.: limpeza de texto, remocao de stopwords, pontuacao, lowercasing).
- Explicar por que parte dessas tecnicas nao deve ser aplicada no BERT
  (tokenizacao e representacao contextual ja fazem parte da abordagem Transformer).

Metricas obrigatorias (no conjunto de teste)
- Matriz de Confusao
- Precision
- Recall
- F1-Score

Analise de erros (obrigatoria)
- Apresentar pelo menos 3 exemplos de divergencia entre os modelos:
  - casos em que o Baseline errou e o Transformer acertou, ou
  - casos em que o Transformer errou e o Baseline acertou.
- Explicar o motivo linguistico do erro
  (ex.: ironia, negacao complexa, ambiguidade, contexto).

Formato sugerido do Notebook (Jupyter)
1. Introducao e objetivo da atividade
2. Dataset (origem, carga, exploracao inicial)
3. Preparacao e divisao treino/validacao/teste
4. Pipeline 1: TF-IDF + modelo classico
5. Pipeline 2: Transformer (BERTimbau/DistilBERT)
6. Avaliacao comparativa (metricas + matriz de confusao)
7. Analise de erros (3 exemplos minimos)
8. Conclusao (comparacao final e aprendizados)

Entregaveis
- Notebook Jupyter completo e executavel
- Documento de relatorio ou apresentacao contendo todos os pontos solicitados
- Preparacao para apresentacao dos resultados

Observacoes praticas
- Fixar seed para reprodutibilidade.
- Registrar versoes de bibliotecas utilizadas.
- Se houver limitacao de hardware, usar DistilBERT e/ou subconjunto balanceado com justificativa.

Alunos:

Lucas Antônio Cunha Rodrigues da Silva
Marcos Vinícius Beregula Ferreira
Michel Iago 
