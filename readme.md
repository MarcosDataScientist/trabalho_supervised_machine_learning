# Classificação de Sentimentos e Análise Comparativa
### Disciplina: Aprendizado de Máquina Supervisionado
**Curso:** Ciência de Dados e Inteligência Artificial — 3º Ano (5º Semestre)  
**Instituição:** Universidade Estadual de Londrina (UEL)  
**Professor:** Bruno Faiçal  

---

## 👥 Equipe de Desenvolvimento
* **Lucas Antônio Cunha Rodrigues da Silva**
* **Marcos Vinícius Beregula Ferreira**
* **Michel Iago**

---

## 📌 1. Objetivo do Projeto
O objetivo deste trabalho prático é desenvolver e comparar duas abordagens para a classificação automática de avaliações de e-commerce brasileiras, dividindo as opiniões entre **Sentimento Positivo (1 - Recomenda)** e **Sentimento Negativo (0 - Não Recomenda)**, conforme os requisitos estabelecidos na atividade prática.

A análise comparativa envolve:
1. **Abordagem Tradicional (Baseline):** Vetorização de palavras com TF-IDF associada ao classificador clássico Regressão Logística.
2. **Abordagem de Deep Learning (Transformers):** Processamento de linguagem natural baseado no modelo de linguagem pré-treinado **BERTimbau** (fine-tuning).

---

## 📊 2. Dataset, Pré-processamento e Balanceamento
O dataset utilizado é o **B2W-Reviews01**, que contém avaliações de produtos reais em português brasileiro.

### 2.1 Limpeza e Preparação Comum
* Remoção de valores ausentes nas colunas críticas: `review_title`, `review_text`, `recommend_to_a_friend` e `overall_rating`.
* Mapeamento do target `recommend_to_a_friend` para binário: `'yes'` $\rightarrow$ **1 (Positivo)** e `'no'` $\rightarrow$ **0 (Negativo)**.

### 2.2 Balanceamento e Incerteza (Entropia de Shannon)
* **Desbalanceamento Original:** A classe de sentimento positivo representava ~73.6% (94.904 amostras), enquanto a classe negativa correspondia a apenas ~26.4% (33.945 amostras).
* **Entropia de Shannon:** A entropia calculada nos dados originais foi de **0.8319 bits** (de um máximo de 1.0 para distribuições perfeitamente balanceadas).
* **Sub-amostragem Aleatória (Random Undersampling):** Para evitar que os classificadores fossem tendenciosos em relação à classe majoritária, aplicou-se undersampling na classe positiva para equipará-la à classe negativa. O conjunto de dados final balanceado resultou em **67.890 amostras** (33.945 de cada classe), com entropia perfeita de **1.0 bit**.

---

## ⚙️ 3. Abordagem Tradicional (TF-IDF + Regressão Logística)

### 3.1 Pré-processamento Específico
Para a abordagem de bag-of-words, o texto passou por:
* **Lowercasing:** Padronização de todas as letras para minúsculas.
* **Strip Accents (Unicode):** Remoção de acentuações gráficas para diminuir a dispersão do vocabulário.
* **Remoção de Stopwords:** Descarte de conectivos sem carga semântica de sentimento (como "de", "com", "o"). As stopwords em português também tiveram seus acentos removidos previamente para evitar falhas de tokenização.

### 3.2 Engenharia de Features e Validação Cruzada (10-Fold CV)
A avaliação foi dividida em três cenários incrementais de características no dataset completo balanceado de 67.890 amostras, avaliados através de validação cruzada estratificada em 10 dobras (10-Fold CV):

> [!TIP]
> **Fusão de Matrizes (hstack):** No cenário C3, a nota numérica (`overall_rating`) foi padronizada com `StandardScaler` e acoplada horizontalmente às colunas esparsas do TF-IDF usando `scipy.sparse.hstack`. Isso evitou a explosão do consumo de memória RAM que ocorreria se a matriz esparsa de textos fosse convertida para densa.

* **C1 - Apenas Texto do Review:** Vetorização unigrama/bigrama do texto bruto do review.
* **C2 - Título + Texto do Review:** Concatenação do título da avaliação ao texto do review antes da vetorização.
* **C3 - Título + Texto + Nota Numérica (`overall_rating`):** Combinação do texto completo com a nota numérica dada pelo cliente (escala de 1 a 5).

### 3.3 Resultados Clássicos (Métricas Out-of-Fold)
As métricas consolidadas a partir das previsões Out-of-Fold (OOF) nos 10 folds foram:

| Cenário de Features | Acurácia OOF | Precision (Classe 0) | Recall (Classe 0) | F1-Score (Classe 0) | Precision (Classe 1) | Recall (Classe 1) | F1-Score (Classe 1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C1 (Apenas Texto)** | 0.8928 | 0.88 | 0.91 | 0.90 | 0.91 | 0.88 | 0.89 |
| **C2 (Título + Texto)** | 0.9139 | 0.91 | 0.92 | 0.92 | 0.92 | 0.90 | 0.91 |
| **C3 (Título + Texto + Nota)** | 0.9455 | 0.95 | 0.94 | 0.94 | 0.94 | 0.95 | 0.94 |

### 3.4 Rigor Estatístico: Testes t Pareados e Bonferroni
Para comprovar que o aumento incremental no desempenho entre os cenários clássicos não ocorreu por acaso, foi conduzido um teste de hipótese estatístico:
* **Validação dos Pressupostos:** O teste de Shapiro-Wilk confirmou a normalidade das diferenças de acurácia entre os folds (p-valor > 0.05 em todas as comparações).
* **Teste t Pareado:** A comparação de médias nos 10 folds apontou que todas as diferenças são estatisticamente significativas (p-valor < 0.05).
* **Correção de Bonferroni:** Para manter o controle do Erro Tipo I global sob múltiplas comparações simultâneas (3 testes), o limite de rejeição foi reajustado para:
  $$\alpha_{\text{ajustado}} = \frac{0.05}{3} \approx 0.0167$$
  Mesmo com o ajuste conservador, todas as comparações mantiveram p-valores muito inferiores ao limiar (ex.: $p \approx 3.75 \times 10^{-10}$ para C2 vs C3), atestando a superioridade estatística do modelo alimentado com a nota numérica.

---

## 🧠 4. Abordagem de Deep Learning (BERTimbau)

### 4.1 Justificativa de Pré-processamento Diferenciado
Ao contrário do baseline tradicional, ao alimentar o Transformer:
1. **Não removemos Stopwords:** O mecanismo de auto-atenção do BERT precisa dos conectivos e pronomes para modelar as relações sintáticas e semânticas completas entre as palavras no contexto de cada frase.
2. **Não removemos acentos ou letras maiúsculas:** O modelo utilizado foi o `neuralmind/bert-base-portuguese-cased`, treinado especificamente para diferenciar caixas alta/baixa. Em análise de sentimento, termos acentuados ou em caixa alta (ex: *"MUITO RUIM"*) carregam ênfases emocionais fundamentais que o modelo aprende a usar.

### 4.2 Configuração Experimental do Ajuste Fino (Fine-Tuning)
Devido a limitações de processamento local, o treinamento foi configurado da seguinte forma:
* **Subconjunto de Treino:** 5.000 amostras estratificadas da base de treinamento balanceada.
* **Validação e Teste:** Executados nas bases de validação (10.863 amostras) e teste (13.578 amostras) completas, para manter a comparação justa com os resultados do TF-IDF.
* **Hiperparâmetros:**
  * Comprimento máximo da sequência (`MAX_LEN`): 128 tokens (cobrindo o título e o corpo do review).
  * Tamanho do lote de treinamento (`BATCH_SIZE`): 8 amostras.
  * Épocas de treinamento (`EPOCHS`): 2 épocas (evitando overfitting).
  * Taxa de aprendizado (`learning_rate`): $2 \times 10^{-5}$ (recomendada para evitar esquecimento catastrófico do conhecimento prévio).

### 4.3 Treinamento de Duas Variantes de Entrada
Como o BERT aceita apenas sequências de texto (ao contrário do `hstack` numérico do clássico), duas arquiteturas de entrada foram submetidas ao ajuste fino:

* **Variante A (BERTimbau Texto-Only):** Recebe o título e o corpo do review concatenados.
* **Variante B (BERTimbau + Rating Textual):** A nota numérica (1 a 5 estrelas) é mapeada para uma representação em português (`1: muito ruim`, `2: ruim`, `3: neutro`, `4: bom`, `5: muito bom`) e inserida como prefixo no início do texto.
  * *Exemplo de entrada:* `[NOTA: muito bom] Amei minha calça. Veste super bem...`

---

## 🏆 5. Resultados e Análise Comparativa Final

Todas as métricas abaixo foram aferidas no **conjunto de teste holdout independente** (13.578 amostras perfeitamente balanceadas):

| Modelo / Abordagem | Acurácia | Precision (0) | Recall (0) | F1-Score (0) | Precision (1) | Recall (1) | F1-Score (1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + Reg. Logística (Só Review)** | 0.8928 | 0.88 | 0.91 | 0.90 | 0.91 | 0.88 | 0.89 |
| **TF-IDF + Reg. Logística (Título + Review)** | 0.9139 | 0.91 | 0.92 | 0.92 | 0.92 | 0.90 | 0.91 |
| **TF-IDF + Reg. Logística (C3: Completo + Nota)** | 0.9455 | 0.95 | 0.94 | 0.94 | 0.94 | 0.95 | 0.94 |
| **BERTimbau (Apenas Texto)** | 0.9327 | 0.93 | 0.94 | 0.94 | 0.94 | 0.93 | 0.94 |
| **BERTimbau (+ Rating Textual)** | **0.9513** | **0.96** | **0.95** | **0.95** | **0.95** | **0.96** | **0.95** |

> [!IMPORTANT]
> **Teste de McNemar:** O teste de McNemar foi aplicado para verificar a relevância estatística no conjunto de teste. O resultado indicou que a variante do **BERTimbau (+ Rating Textual)** superou a Regressão Logística C3 clássica e o BERT texto-only de forma altamente significativa ($p \approx 7.5 \times 10^{-8}$ e $p \approx 7.6 \times 10^{-12}$, respectivamente), registrando o melhor desempenho geral do projeto.

---

## 🔍 6. Análise Qualitativa de Erros
Identificamos 4 categorias principais de falhas que mostram os desafios semânticos e as nuances linguísticas das avaliações:

### ❌ Exemplo 1: Regressão Logística Erra (Negativo $\rightarrow$ Positivo) \| BERTimbau Acerta (Negativo)
* **Texto:** Título: *"Linha Rosa (Linha da Morte)"* | Corpo: *"O produto é excelente porém o aparelho que recebi [...] fiquei muito decepcionado [...] a Samsung sabe desse problema mas trata como algo casual"*
* **Motivo Linguístico:** A presença de termos com peso positivo elevado isolados (como *"excelente"*) confunde o TF-IDF. O BERTimbau, por compreender a sintaxe de dependência a partir da conjunção adversativa *"porém"*, detecta com precisão a quebra de expectativa e a decepção final do usuário.

### ❌ Exemplo 2: Regressão Logística Erra (Negativo $\rightarrow$ Positivo) \| BERTimbau Acerta (Negativo)
* **Texto:** Título: *"Útil, mas frágil"* | Corpo: *"O produto é ótimo para limpeza [...] Só que o cabo desse é muito frágil e curto [...] pra mim é bem ruim usar"*
* **Motivo Linguístico:** Ocorrência de múltiplos lexemas de polaridades opostas em uma mesma avaliação (*ótimo*, *útil* vs *frágil*, *ruim*). A regressão clássica falha pela contagem simples de frequências, enquanto o Transformer pondera o contexto sintático e o peso negativo explícito no título para acertar a predição.

### ❌ Exemplo 3: BERTimbau Erra (Positivo $\rightarrow$ Negativo) \| Regressão Logística Acerta (Positivo)
* **Texto:** Título: (Avaliação de entrega lenta) | Corpo: *"O produto é muito bom, mas tive problemas com a entrega [...] Só depois de 20 dias [...] Portanto eu recomendo o produto, mas não recomendo a loja"*
* **Motivo Linguístico:** O texto foca majoritariamente em frustrações logísticas (*"não recomendo a loja"*, *"problemas com a entrega"*). O BERTimbau pondera esse forte sentimento negativo geral e erra a classificação. A Regressão Logística, somando a contagem bruta de termos direcionados ao item (*"muito bom"*, *"recomendo o produto"*), atinge o peso necessário para prever a classe positiva.

### ❌ Exemplo 4: Ambos Falham (Dataset Ruidoso / Inconsistência)
* **Texto:** Título: *"Amei o produto"* | Corpo: *"Jogo de panelas excelente material, entrega a tempo, design muito bom."*
* **Rótulo Real no Dataset:** 0 (Negativo)
* **Motivo Linguístico:** Há uma inconsistência de rotulação (ruído de anotação) nos dados brutos do e-commerce. O usuário elogiou fortemente o produto em todas as frentes, mas acabou marcando que "não recomendava a um amigo". Nesse cenário, o erro dos modelos reflete o comportamento anômalo do usuário humano.

### 💡 Exemplo 5: O impacto desambiguador do Rating Textual
* **Texto:** Título: *"composição"* | Corpo: *"Para comprar o produto eu gostaria de saber se é 100% algodão ou tem elastano. Essa informação existia, porém foi retirada de todas as calças. obrigado..."*
* **Rótulo Real:** 0 (Negativo) | **BERT Texto-Only classificou como:** 1 (Positivo) | **BERT + Rating classificou como:** 0 (Negativo)
* **Explicação:** A avaliação é apenas uma dúvida sem termos de ódio explícitos, o que faz o modelo de texto errar. Contudo, ao ler a nota dada pelo cliente (Nota 1 $\rightarrow$ *"muito ruim"*), o BERT desambigua a dúvida neutra e atribui corretamente o sentimento negativo à crítica.

---

## 💾 7. Exportação e Implantação Prática
Para viabilizar a implantação local e o funcionamento offline dos modelos em ambiente de produção (através da extensão do Google Chrome contida no diretório `../pratico`), os artefatos de treinamento foram serializados:
1. **Modelos Clássicos (`.pkl`):** O vetorizador TF-IDF, o scaler numérico e os pesos da Regressão Logística (clf2 e clf3) foram salvos via `joblib` no diretório `../pratico/modelos/modelo_regressao_logistica/`.
2. **Modelos de Deep Learning (Transformers):** O tokenizador e os pesos finais ajustados do BERTimbau (texto e rating) foram exportados através do HuggingFace Trainer para as pastas `../pratico/modelos/modelo_bertimbau` e `../pratico/modelos/modelo_bertimbau_rating`, permitindo o carregamento direto do modelo pelo servidor local Flask (`pratico/app.py`).
