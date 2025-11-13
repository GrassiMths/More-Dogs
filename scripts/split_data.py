import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Caminhos dos arquivos
INPUT_FILE = "data/pet_adoption_data.csv"
TRAIN_FILE = "data/train.csv"
TEST_FILE = "data/test.csv"

# Proporção de dados para teste
TEST_SIZE = 0.3
RANDOM_STATE = 42  # Para reprodutibilidade


def main():
    # Verifica se o arquivo existe
    if not os.path.exists(INPUT_FILE):
        print(f"Erro: Arquivo {INPUT_FILE} não encontrado!")
        print(f"Certifique-se de que o arquivo existe no diretório data/")
        exit(1)
    
    # Carrega os dados
    print(f"Carregando dados de {INPUT_FILE}...")
    dados = pd.read_csv(INPUT_FILE)
    print(f"Total de registros: {len(dados)}")
    
    # Prepara estratificação se a coluna existir
    stratify_col = None
    if "AdoptionLikelihood" in dados.columns:
        stratify_col = dados["AdoptionLikelihood"]
        print("Usando estratificação baseada em AdoptionLikelihood")
    
    # Divide os dados
    print(f"\n Dividindo dados ({int((1-TEST_SIZE)*100)}% treino, {int(TEST_SIZE*100)}% teste)...")
    train_data, test_data = train_test_split(
        dados,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_col if stratify_col is not None else None
    )
    
    # Salva os arquivos
    print(f"\n Salvando {len(train_data)} registros em {TRAIN_FILE}...")
    train_data.to_csv(TRAIN_FILE, index=False)
    
    print(f" Salvando {len(test_data)} registros em {TEST_FILE}...")
    test_data.to_csv(TEST_FILE, index=False)
    
    # Resumo
    print("\n Divisão concluída!")
    print(f"   Treino: {len(train_data)} registros ({len(train_data)/len(dados)*100:.1f}%)")
    print(f"   Teste: {len(test_data)} registros ({len(test_data)/len(dados)*100:.1f}%)")


if __name__ == "__main__":
    main()

