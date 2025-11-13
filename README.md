### Principais funcionalidades
- **Upload de CSV** com validação de colunas obrigatórias.
- **Aba Análise**: filtros por porte, raça, vacinação e saúde, com:
  - métricas rápidas (contagem, taxa média de adoção, tempo médio no abrigo);
  - gráficos interativos (Plotly) de distribuição, boxplots, dispersão e ranking de raças;
  - matriz de correlação entre variáveis numéricas.
- **Aba Machine Learning**:
  - pré-processamento com `ColumnTransformer` (imputação, One-Hot Encoding e padronização);
  - seleção de modelo: DecisionTree, RandomForest, KNN, LogisticRegression;
  - ajustes de hiperparâmetros via UI (sliders);
  - divisão treino/teste estratificada;
  - métricas (acurácia e F1) e matriz de confusão;
  - formulário para predizer a adoção de um novo cão com base no modelo treinado.

---

### Requisitos
- Python 3.10+
- Pacotes/libs (arquivo `requirements.txt`):
  - streamlit, pandas, numpy, plotly, scikit-learn

Instalação:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

### Inicialização

```bash
streamlit run app.py
```

Link `http://localhost:8501`.

---

### Formato do CSV esperado
O arquivo deve conter ao menos as colunas abaixo (nomes exatos). Registros não caninos serão filtrados automaticamente para `PetType == "dog"`.

| Coluna               | Tipo/Valores                     | Descrição                                                                 |
|----------------------|----------------------------------|---------------------------------------------------------------------------|
| PetID                | texto/ID                         | Identificador do pet                                                      |
| PetType              | texto                            | Tipo do animal (ex.: `"dog"`)                                            |
| Breed                | texto                            | Raça                                                                      |
| AgeMonths            | numérico                         | Idade em meses                                                            |
| Color                | texto                            | Cor                                                                       |
| Size                 | texto                            | Porte                                                                     |
| WeightKg             | numérico                         | Peso em kg                                                                |
| Vaccinated           | inteiro {0,1}                    | 1 se vacinado, 0 caso contrário                                          |
| HealthCondition      | inteiro {0,1}                    | 0 saudável, 1 com condição médica                                         |
| TimeInShelterDays    | numérico                         | Tempo no abrigo (dias)                                                    |
| AdoptionFee          | numérico                         | Taxa de adoção (moeda livre)                                              |
| PreviousOwner        | inteiro {0,1}                    | 1 se já teve dono, 0 caso contrário                                       |
| AdoptionLikelihood   | inteiro {0,1}                    | Rótulo alvo: 1 provável adoção, 0 improvável                              |

Notas de tratamento:
- Valores não numéricos nas colunas numéricas são convertidos para `NaN` e imputados.
- `Vaccinated`, `HealthCondition`, `PreviousOwner`, `AdoptionLikelihood` são convertidos para inteiros.
- A análise utiliza filtros e mostra apenas uma amostra/tabulações; o treino usa a base completa de cães com rótulo (`AdoptionLikelihood` não nulo).

---

### Como usar
1. Inicie o app (`streamlit run app.py`) e envie o seu CSV na página inicial.
2. Na aba **Análise**, ajuste filtros e explore:
   - métricas de topo;  
   - gráficos de adoção por porte, idade por classe, tempo no abrigo, peso x idade, ranking por raça (mín. 20 cães) e correlação.
3. Na aba **Machine Learning**:
   - escolha o modelo (DecisionTree, RandomForest, KNN, LogisticRegression);
   - ajuste hiperparâmetros (sliders);
   - defina a proporção de teste e clique em “Treinar modelo”;
   - consulte acurácia, F1 e a matriz de confusão;
   - use o formulário “Predição para novo cão” para simular um novo registro e obter a previsão (e, se disponível, a probabilidade).

---

### Modelos e hiperparâmetros (UI)
- DecisionTree: `max_depth`
- RandomForest: `n_estimators`, `max_depth`
- KNN: `n_neighbors`
- LogisticRegression: `C`, `max_iter` (fixo em 1000 no código)

Pré-processamento:
- numéricos: imputação por mediana + `StandardScaler`
- categóricos: imputação por modo + `OneHotEncoder(handle_unknown="ignore")`

---

### Métricas exibidas
- Acurácia (`accuracy_score`)
- F1-score binário (`f1_score`, com `zero_division=0`)
- Matriz de confusão (Plotly Heatmap)

---

### Estrutura do projeto
```
more-dogs/
├─ app.py
├─ requirements.txt
└─ data/               # arquivos de dados locais
```
