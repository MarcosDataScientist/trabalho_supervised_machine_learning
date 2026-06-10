# Atividade de Pesquisa e Prática — Classificação de Sentimentos

Este documento apresenta a formatação organizada e detalhada das diretrizes estabelecidas no arquivo [Aula14-Atividadepratica.pdf](file:///Users/marcosvinicius/Desktop/codigos/faculdade/supervised_ml/trabalho_supervised_machine_learning/Aula14-Atividadepratica.pdf).

---

## 🎯 Desafio: Classificação de Sentimentos e Análise Comparativa

### Contexto
Uma plataforma de e-commerce contratou a equipe para automatizar a triagem de avaliações de produtos deixadas pelos usuários. O objetivo é classificar essas avaliações em sentimentos **Positivos** ou **Negativos**.

---

## 🛠️ O que deve ser desenvolvido

Os alunos deverão criar um script ou notebook em Python que implemente e compare **duas abordagens distintas** para o mesmo problema:

### 1. Abordagem Tradicional (Baseline)
*   **Vetorização:** TF-IDF (*Term Frequency-Inverse Document Frequency*).
*   **Classificador:** Algoritmo clássico de Aprendizado de Máquina (sugestões: *Regressão Logística*, *SVM* ou *Naive Bayes*) utilizando a biblioteca `scikit-learn`.

### 2. Abordagem Baseada em Aprendizado Profundo (Deep Learning)
*   **Modelo:** Utilizar um modelo de linguagem pré-treinado baseado em Transformers da Hugging Face:
    *   *BERTimbau:* Treinado especificamente para o português brasileiro.
    *   *DistilBERT:* Modelo destilado, ideal para execução mais rápida.
*   **Tarefa:** Realizar o **Fine-Tuning** (ajuste fino) ou a **extração de features** deste modelo para a tarefa de classificação de texto.

### 2.1 Entendendo as Estratégias: Fine-Tuning vs. Extração de Features

A escolha entre estas duas técnicas afeta diretamente o custo computacional e o desempenho final do classificador:

*   **Fine-Tuning (Ajuste Fino):**
    *   *Funcionamento:* Conecta-se uma camada classificadora no topo do Transformer e treina-se o **modelo inteiro**. Durante o aprendizado, os erros são retropropagados por todas as camadas pré-treinadas, atualizando os seus pesos.
    *   *Vantagens:* Ajusta as representações internas de linguagem especificamente para as gírias e vocabulário do domínio (ex: avaliações de produtos), gerando acurácia superior.
    *   *Desvantagens:* Alto custo computacional e de memória, exigindo aceleração gráfica (GPU ou MPS) para um tempo razoável.
*   **Extração de Features (Feature Extraction):**
    *   *Funcionamento:* Mantém-se o Transformer com todos os pesos **congelados** (estáticos). O texto é processado para gerar vetores de alta dimensão (geralmente extraindo a saída do token especial `[CLS]`). Estes vetores servem como dados de entrada estáticos para treinar um classificador clássico clássico (como Regressão Logística ou SVM).
    *   *Vantagens:* Extremamente rápido e leve, pois executa o Transformer apenas em sentido direto (*forward pass*) uma única vez.
    *   *Desvantagens:* Menor flexibilidade, já que o modelo de linguagem não se adapta à semântica e termos específicos do seu dataset.

---

## 📋 Requisitos Técnicos e Entregáveis

> [!IMPORTANT]
> A implementação deve atender rigorosamente a todos os critérios listados abaixo.

| Requisito | Descrição |
| :--- | :--- |
| **Dataset** | Utilizar um dataset público em português (ex: *B2W-Reviews01*, *Buscapé*, ou subconjunto do *IMDb* traduzido). O conjunto deve ser dividido estritamente em **Treino**, **Validação** e **Teste**. |
| **Pré-processamento** | Justificar as etapas de limpeza de texto adotadas (remoção de stop-words, pontuação ou lowercasing) para a abordagem tradicional e **explicar** por que algumas dessas etapas não devem ser feitas ao usar o modelo BERT/Transformer. |
| **Métricas de Avaliação** | Gerar uma **Matriz de Confusão** e reportar as métricas de **Precision**, **Recall** e **F1-Score** para ambas as abordagens no conjunto de teste. |
| **Análise de Erros** | Identificar pelo menos **3 exemplos** de divergência onde a abordagem tradicional falhou mas o Transformer acertou (ou vice-versa). Deve-se explicar textualmente o motivo linguístico do erro (ex: presença de ironia, negações complexas ou palavras ambíguas). |

---

## 📦 Entrega e Apresentação

> [!WARNING]
> Ambos os itens abaixo são necessários para a conclusão da atividade:

1.  **Documento:** Submeter um documento (relatório ou apresentação em formato PDF/Markdown) contendo todos os pontos solicitados e as análises correspondentes.
2.  **Apresentação:** Será necessária a apresentação oral dos resultados obtidos perante o professor e a turma.
