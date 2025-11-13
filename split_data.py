import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Caminhos dos arquivos
input_file = "data/pet_adoption_data.csv"
train_file = "data/train.csv"
test_file = "data/test.csv"

# Verifica se o arquivo existe
if not os.path.exists(input_file):
    print(f"Erro: Arquivo {input_file} não encontrado!")
    exit(1)

# Carrega os dados
print(f"Carregando dados de {input_file}...")
dados = pd.read_csv(input_file)

print(f"Total de registros: {len(dados)}")

# Divide os dados (70% treino, 30% teste)
# Usa stratify se a coluna AdoptionLikelihood existir para manter proporção
stratify_col = None
if "AdoptionLikelihood" in dados.columns:
    stratify_col = dados["AdoptionLikelihood"]
    print("Usando estratificação baseada em AdoptionLikelihood")

train_data, test_data = train_test_split(
    dados,
    test_size=0.3,
    random_state=42,
    stratify=stratify_col if stratify_col is not None else None
)

# Salva os arquivos
print(f"\nSalvando {len(train_data)} registros em {train_file}...")
train_data.to_csv(train_file, index=False)

print(f"Salvando {len(test_data)} registros em {test_file}...")
test_data.to_csv(test_file, index=False)

print("\nDivisão concluída!")
print(f"Treino: {len(train_data)} registros ({len(train_data)/len(dados)*100:.1f}%)")
print(f"Teste: {len(test_data)} registros ({len(test_data)/len(dados)*100:.1f}%)")

