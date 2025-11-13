# More Dogss
---
## O que faz

- Analisa dados de cães em abrigos através de gráficos e estatísticas
- Treina modelos de aprendizado de máquina para prever probabilidade de adoção
- Permite fazer predições individuais ou em lote
- Salva modelos treinados para uso posterior
- Oferece API REST para integração com outros sistemas

## Requisitos

- Python 3.10 ou superior
- Pacotes/libs listadas no arquivo `requirements.txt`

## Instalação

1. Crie um ambiente virtual (recomendado):

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# Windows: .venv\Scripts\activate
```

2. Instale as dependências:

```bash
pip install -r requirements.txt
```

## Como usar

### Interface Web (Streamlit)

Inicie a aplicação:

```bash
streamlit run app.py
```

`http://localhost:8501`

**Passos:**

1. Faça upload de um arquivo CSV com os dados dos pets
2. Na aba **Análise**, explore os dados com filtros e gráficos
3. Na aba **Machine Learning**:
   - Escolha um modelo (DecisionTree, RandomForest, KNN, LogisticRegression)
   - Ajuste os parâmetros usando os controles deslizantes
   - Clique em "Treinar modelo" para treinar
   - Use o formulário para fazer predições individuais
   - Faça upload de um arquivo test.csv para predições em lote

### API Flask

Inicie a API:

```bash
python run_api.py
```

**Endpoints principais:**

- `GET /api/models` - Lista modelos disponíveis
- `POST /api/predict` - Faz predição individual (envie JSON)
- `POST /api/predict/batch` - Faz predição em lote (envie CSV ou JSON)

**Exemplo de predição individual:**

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Breed": "Labrador",
    "Color": "Brown",
    "Size": "Large",
    "AgeMonths": 24,
    "WeightKg": 15.5,
    "TimeInShelterDays": 30,
    "AdoptionFee": 200,
    "Vaccinated": 1,
    "HealthCondition": 0,
    "PreviousOwner": 0
  }'
```

## Formato do arquivo CSV

O arquivo CSV deve conter as seguintes colunas:

| Coluna             | Tipo   | Descrição                                                  |
| ------------------ | ------ | ---------------------------------------------------------- |
| PetID              | texto  | Identificador único do pet                                 |
| PetType            | texto  | Tipo do animal (deve ser "dog")                            |
| Breed              | texto  | Raça do cão                                                |
| AgeMonths          | número | Idade em meses                                             |
| Color              | texto  | Cor                                                        |
| Size               | texto  | Porte (Small, Medium, Large)                               |
| WeightKg           | número | Peso em quilogramas                                        |
| Vaccinated         | 0 ou 1 | 1 se vacinado, 0 se não                                    |
| HealthCondition    | 0 ou 1 | 0 se saudável, 1 se tem condição                           |
| TimeInShelterDays  | número | Dias no abrigo                                             |
| AdoptionFee        | número | Taxa de adoção                                             |
| PreviousOwner      | 0 ou 1 | 1 se já teve dono, 0 se não                                |
| AdoptionLikelihood | 0 ou 1 | 1 se provável adoção, 0 se improvável (apenas para treino) |

**Notas:**

- Apenas registros com PetType igual a "dog" serão analisados
- Valores numéricos inválidos são corrigidos automaticamente
- Colunas numéricas podem ter valores vazios (serão preenchidos automaticamente)

## Scripts auxiliares

**Dividir dados em treino e teste:**

```bash
python scripts/split_data.py
```

Este script divide o arquivo `data/pet_adoption_data.csv` em 70% para treino e 30% para teste.

**Testar a API:**

```bash
python scripts/test_api.py
```

## Modelos

1. **DecisionTree** - Árvore de decisão

   - Parâmetro: max_depth (profundidade máxima)

2. **RandomForest** - Floresta aleatória

   - Parâmetros: n_estimators (número de árvores), max_depth

3. **KNN** - K-vizinhos mais próximos

   - Parâmetro: n_neighbors (número de vizinhos)

4. **LogisticRegression** - Regressão logística
   - Parâmetro: C (força da regularização)

## Estrutura do projeto

```
more-dogs/
├── app.py                  # Interface web (Streamlit)
├── run_api.py             # Inicia a API Flask
├── requirements.txt        # Dependências
├── README.md               # Este arquivo
│
├── src/                    # Código compartilhado
│   └── model_utils.py     # Funções para gerenciar modelos
│
├── api/                    # API Flask
│   ├── app.py             # Aplicação Flask
│   └── run.py             # Script alternativo
│
├── scripts/                # Scripts auxiliares
│   ├── split_data.py      # Divide dados
│   └── test_api.py        # Testa API
│
├── data/                   # Arquivos de dados
│   ├── pet_adoption_data.csv
│   ├── train.csv
│   └── test.csv
│
└── models/                 # Modelos treinados (criado automaticamente)
    └── *.pkl
```

## Métricas de avaliação

Os modelos são avaliados usando:

- **Acurácia**: Porcentagem de predições corretas
- **F1-Score**: Medida que combina precisão e recall
- **Matriz de Confusão**: Mostra quantos acertos e erros de cada tipo
