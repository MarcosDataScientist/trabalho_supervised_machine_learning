# Relatório — Atividade Prática: Classificação de Sentimentos e Análise Comparativa

**Disciplina:** Aprendizado de Máquina Supervisionado (UEL — CCE)  
**Professor:** Bruno Faiçal  
**Data:** 05/06/2026  

**Alunos:**
- Lucas Antônio Cunha Rodrigues da Silva
- Marcos Vinícius Beregula Ferreira
- Michel Iago

**Artefatos do projeto:** `analise_sentimento.ipynb`, `B2W-Reviews01 - B2W-Reviews01.csv`, `requeriments.txt`

---

## 1. Objetivo e contexto do trabalho

### Situação 1 — Método convencional (TF-IDF + Regressão Logística)

Automatizar a triagem de avaliações de produtos de um e-commerce, classificando-as em sentimentos **negativos** (não recomenda) ou **positivos** (recomenda), com base no rótulo `recommend_to_a_friend` do dataset B2W-Reviews01.

Removemos colunas que não têm relação direta com o texto da avaliação (identificadores, demografia do revisor, categorias de site etc.) para **isolar a capacidade preditiva do conteúdo textual**. Embora gênero ou faixa etária possam influenciar o tom das avaliações, o foco deste trabalho não é modelar perfil do consumidor, e sim o **sentimento expresso no texto**.

Foram treinados três baselines progressivos:

1. Apenas `review_text`
2. `review_title` + `review_text`
3. `review_title` + `review_text` + `overall_rating` (via `hstack` com `StandardScaler`)

Na etapa tradicional, a variante **título + avaliação + rating** foi a mais eficiente no conjunto de teste.

### Situação 2 — Deep Learning (BERTimbau)

Optamos pelo **BERTimbau** (`neuralmind/bert-base-portuguese-cased`) por ser um modelo BERT completo treinado especificamente para o português brasileiro, o que garante maior acurácia semântica e sintática no idioma nacional, realizando o fine-tuning de forma eficiente localmente com aceleração MPS (Metal Performance Shaders) no macOS.

---

## 2. Dataset e divisão dos dados


Fonte  B2W-Reviews01 (avaliações em português) 
Registros após `dropna` nas colunas de interesse  128.849 
Rótulo  `yes` → 1 (positivo), `no` → 0 (negativo) 
Colunas finais para modelagem  `review_title`, `review_text`, `overall_rating` 

**Split (estratificado, `random_state=42`):**

1. 80% temporário / 20% **teste** (25.770 amostras)
2. Do temporário: 80% **treino** / 20% **validação** (82.463 treino, 20.616 validação)

O mesmo particionamento foi reutilizado na etapa BERTimbau, garantindo comparabilidade entre abordagens.

**Observação (BERTimbau):** por limitação de tempo e hardware, o fine-tuning usou **5.000 amostras** amostradas estratificadamente do conjunto de treino (`N_TREINO_BERT = 5000`), enquanto o baseline TF-IDF foi treinado no treino completo (~82k). Essa diferença deve ser considerada na interpretação dos resultados.

---

## 3. Pré-processamento e justificativas

### 3.1 Etapas comuns (ambas as abordagens)


Remoção de nulos  `dropna` em título, texto, rótulo e rating - Evita rótulos ou entradas vazias 
Remoção de colunas extras  IDs, datas, demografia, categorias  Foco no texto; reduz vazamento de contexto não textual 
Agregação título + texto  `review_plus_title = título + " " + texto`  Título resume o sentimento; melhora o baseline e alimenta o BERT 
 `random_state=42`  Fixado em splits e amostragem BERT  Reprodutibilidade 

### 3.2 Abordagem tradicional (TF-IDF + Regressão Logística)


 **Lowercasing**  `TfidfVectorizer(lowercase=True)`  Reduz duplicidade “Bom” / “bom”; bag-of-words não preserva casing 
 **Remoção de acentos**  `strip_accents='unicode'`  Aproxima variantes ortográficas (“ótimo” / “otimo”) no vocabulário esparsificado 
 **Stopwords em português**  `stopwords.words("portuguese")` no vectorizer  Palavras funcionais (“de”, “a”, “que”) têm pouco poder discriminativo e inflam a dimensionalidade 
 **TF-IDF**  Dois vectorizers (`review` e `full`) com `fit` só no treino  Representação esparsa adequada a vocabulário grande; evita vazamento com `transform` em val/teste 
 **Rating numérico**  `StandardScaler` + `hstack` (não passa pelo TF-IDF)  Escala numérica comparável às features esparsas; rating correlaciona com recomendação 

**O que não foi aplicado de forma agressiva:** remoção manual de pontuação, stemming ou lematização. O `TfidfVectorizer` já tokeniza por palavras; pontuação isolada tende a peso baixo, e stemming em português exigiria biblioteca extra com ganho incerto neste dataset.

**Aviso observado na execução:** o sklearn alertou que algumas stopwords portuguesas (ex.: “não”, “só”) podem ser tokenizadas de forma inconsistente. Isso reforça que remover “não” como stopword pode **prejudicar** a detecção de negação — um ponto relevante na análise de erros.

### 3.3 Abordagem BERTimbau — o que **não** deve ser replicado do pipeline tradicional

Etapa tradicional  Por que **evitar** no BERT 

 Remoção de stopwords  O tokenizador WordPiece precisa da sequência original; stopwords ajudam na estrutura sintática 
 Lowercasing manual agressivo  O modelo usado é **cased** (`neuralmind/bert-base-portuguese-cased`); maiúsculas podem carregar ênfase 
 Remoção de pontuação  Pontuação pode sinalizar ironia, exclamação, hesitação |
 Stemming / lematização  Destrói forma superficial que o pré-treinamento aprendeu a explorar 
 TF-IDF antes do modelo  O Transformer aprende representações contextuais; a entrada deve ser **texto bruto** tokenizado 

**Pipeline BERT adotado:** texto `título + review_text` como string → `AutoTokenizer` (truncamento `MAX_LEN=128`) → fine-tuning com `AutoModelForSequenceClassification` (2 classes).

---

## 4. Resultados no conjunto de teste (25.770 amostras)

### 4.1 Baseline TF-IDF + Regressão Logística

| Modelo | Acurácia | F1 classe 0 (neg.) | F1 classe 1 (pos.) |

| Só `review_text` | 0,913 | 0,83 | 0,94 |
| Título + `review_text` | **0,930** | 0,87 | 0,95 |
| Título + texto + **rating** | **0,956** | 0,92 | 0,97 |

Matriz de confusão — **título + review** (comparável ao texto do BERT):

```
[[ 5787  1002]
 [  797 18184]]
```

### 4.2 BERTimbau — TESTE (resultado obtido na execução)

```
==== BERTIMBAU - TESTE ====
              precision    recall  f1-score   support

           0       0.85      0.86      0.86      6789
           1       0.95      0.95      0.95     18981

    accuracy                           0.92     25770
   macro avg       0.90      0.90      0.90     25770
weighted avg       0.92      0.92      0.92     25770

[[ 5835   954]
 [ 1017 17964]]

Acurácia: 0.9235157159487777
```

**Sim — a acurácia em teste do BERTimbau foi encontrada e está correta:** **92,35%** no mesmo conjunto de teste (25.770 avaliações), com desempenho forte na classe positiva (F1 ≈ 0,95) e recall da classe negativa (~0,86) superior ao baseline “só review” (recall ≈ 0,80), porém ligeiramente abaixo do baseline “título + review” em acurácia global (93,02%).

### 4.3 Leitura comparativa honesta

BERTimbau vs TF-IDF **título + review**  Acurácias muito próximas (92,35% vs 93,02%); empate técnico com leve vantagem do baseline nesta execução 
BERTimbau vs TF-IDF **+ rating**  O baseline com rating (95,6%) supera o BERT, pois usa sinal numérico explícito que o Transformer não recebeu 
Custo computacional  BERT: fine-tuning com 5k amostras, 2 épocas; baseline: treino completo, inferência muito mais rápida 
Generalização linguística  O BERT tende a capturar melhor contexto (negação, ironia); o TF-IDF depende de n-grams e pode confundir polaridade 


## 5. Análise de erros (mínimo 3 exemplos)

Os exemplos abaixo foram selecionados no dataset B2W por **padrões linguísticos** típicos de discordância entre bag-of-words e modelos contextuais. Eles ilustram *por que* as matrizes de confusão ainda registram centenas de erros em cada modelo (baseline: 1.002 FP + 797 FN no modelo título+review; BERT: 954 FP + 1.017 FN).

> **Legenda:** Positivo = recomenda (`yes`); Negativo = não recomenda (`no`).  

---

### Exemplo 1 — Baseline tende a errar / BERT tende a acertar  
**Padrão:** elogio + adversativa + crítica (polaridade mista)

- **Título:** Linha Rosa (Linha da Morte)  
- **Trecho:** *"O produto é excelente porém o aparelho que recebi [...] fiquei muito decepcionado [...] a Samsung sabe desse problema mas trata como algo casual"*  
- **Rótulo verdadeiro:** Negativo (0)

**Motivo linguístico:** A palavra *"excelente"* tem peso TF-IDF forte para a classe positiva, enquanto a reviravolta (*"porém"*, *"decepcionado"*) define o sentimento global. Regressão logística em bag-of-words trata features de forma mais independente; o Transformer modela dependência entre tokens e costuma priorizar o fechamento negativo da avaliação.

---

### Exemplo 2 — Baseline tende a errar / BERT tende a acertar  
**Padrão:** negação implícita e contraste (*"ótimo para X, só que..."*)

- **Título:** Útil, mas frágil  
- **Trecho:** *"O produto é ótimo para limpeza [...] Só que o cabo desse é muito frágil e curto [...] pra mim é bem ruim usar"*  
- **Rótulo verdadeiro:** Negativo (0)

**Motivo linguístico:** Há lexemas positivos (*ótimo*, *útil*) e negativos (*frágil*, *ruim*) no mesmo documento. Sem composição profunda de negação/escopo, o baseline pode oscilar conforme frequência de termos. O título já sinaliza contraste (*"mas frágil"*), o que modelos contextuais exploram melhor ao ler título e corpo juntos.

---

### Exemplo 3 — Transformer tende a errar / Baseline tende a acertar  
**Padrão:** recomendação condicional e reclamação logística (positivo com trecho negativo forte)

- **Título:** (avaliação positiva com problema de entrega)  
- **Trecho:** *"O produto é muito bom, mas tive problemas com a entrega [...] Só depois de 20 dias [...] Portanto eu recomendo o produto, mas não recomendo a loja"*  
- **Rótulo verdadeiro:** Positivo (1) — recomenda o produto

**Motivo linguístico:** O texto contém *"problemas"*, atraso e até a expressão *"não recomendo a loja"*. Um Transformer pode superponderar spans negativos locais e prever classe 0, enquanto o baseline, somado à repetição de *"recomendo o produto"* e palavras positivas sobre o item, pode acertar o rótulo global. Ilustra **ambiguity de escopo**: o que está sendo avaliado (produto vs experiência de compra).

---

### Exemplo 4 (complementar) — Ambiguidade e possível ruído de rótulo  
**Padrão:** título extremo vs corpo discordante

- **Título:** Amei o produto  
- **Corpo:** *"Jogo de panelas excelente material, entrega a tempo, design muito bom."*  
- **Rótulo no dataset:** Negativo (0)

**Motivo linguístico:** Há **inconsistência aparente** entre título, corpo e rótulo. Tanto TF-IDF quanto BERT podem falhar por ruído de anotação ou contexto externo (produto errado, defeito não mencionado no texto). Reforça que métricas altas não eliminam a necessidade de **auditoria qualitativa** das amostras.

---

## 6. Overfitting e validação (baseline)

No gráfico do notebook, a diferença entre acurácia de treino e validação no melhor modelo textual ficou em torno de **0,46 p.p.**, indicando **pouco overfitting** para TF-IDF + regressão logística nesta configuração. O BERTimbau deve ser monitorado pela curva de loss/accuracy em validação durante as épocas (2 épocas configuradas).


## 7. Conclusões

1. **Objetivo cumprido:** Foi implementada e avaliada uma solução completa em Jupyter, com baseline TF-IDF (três variantes) e fine-tuning BERTimbau no mesmo split de teste.

2. **Métricas obrigatórias:** Matriz de confusão, precision, recall, F1 e acurácia foram obtidas no **teste** para ambas as abordagens. A acurácia do BERTimbau em teste é **92,35%**, conforme a saída reportada.

3. **Pré-processamento:** A limpeza agressiva (stopwords, lowercasing, remoção de acentos) é adequada ao vocabulário estático do TF-IDF, mas **não deve ser replicada** no pipeline do BERT, que depende de tokenização e contexto preservados.

4. **Comparação:** Com apenas texto (título + review), BERTimbau e o melhor baseline textual ficam **equivalentes** (~92–93%), com vantagem do baseline quando se inclui **rating** (95,6%). O Transformer permanece relevante para casos com polaridade mista, ironia e negação escopada — cenários descritos na análise de erros.

5. **Limitações:** Subconjunto de 5k para treino BERT; modelo específico para PT-BR (BERTimbau); exemplos de erro devem ser validados no notebook com predições salvas (`pred_baseline`, `pred_bert`) para citar IDs reais na apresentação.

6. **Próximo passo sugerido:** Célula no notebook que exporta automaticamente 3 linhas do teste com `y_true`, `pred_tfidf`, `pred_bert` e o texto — para alinhar 100% o relatório com a execução final.


Muito obrigado,

Da equipe.
