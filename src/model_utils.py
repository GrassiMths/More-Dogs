import os
import joblib
from datetime import datetime

# Cria pasta models se não existir
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def salvar_modelo(pipe, modelo_nome, acc, f1):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_arquivo = f"{MODELS_DIR}/{modelo_nome}_{timestamp}.pkl"
    
    info = {
        "modelo": pipe,
        "nome": modelo_nome,
        "acurácia": acc,
        "f1_score": f1,
        "timestamp": timestamp
    }
    
    joblib.dump(info, nome_arquivo)
    return nome_arquivo


def listar_modelos():
    modelos = []
    
    if os.path.exists(MODELS_DIR):
        for arquivo in os.listdir(MODELS_DIR):
            if arquivo.endswith(".pkl"):
                caminho = os.path.join(MODELS_DIR, arquivo)
                try:
                    info = joblib.load(caminho)
                    modelos.append({
                        "arquivo": arquivo,
                        "caminho": caminho,
                        "nome": info.get("nome", "Desconhecido"),
                        "acurácia": info.get("acurácia", 0),
                        "f1_score": info.get("f1_score", 0),
                        "timestamp": info.get("timestamp", "")
                    })
                except Exception:
                    pass
    
    return sorted(modelos, key=lambda x: x["timestamp"], reverse=True)


def carregar_modelo(arquivo_ou_indice):
    modelos = listar_modelos()
    
    if isinstance(arquivo_ou_indice, int):
        # Busca por índice
        if 0 <= arquivo_ou_indice < len(modelos):
            return joblib.load(modelos[arquivo_ou_indice]["caminho"])
        else:
            return None
    else:
        # Busca por nome do arquivo
        for m in modelos:
            if m["arquivo"] == arquivo_ou_indice or m["caminho"] == arquivo_ou_indice:
                return joblib.load(m["caminho"])
        return None


def limpar_modelos():
    removidos = 0
    
    if os.path.exists(MODELS_DIR):
        for arquivo in os.listdir(MODELS_DIR):
            if arquivo.endswith(".pkl"):
                try:
                    os.remove(os.path.join(MODELS_DIR, arquivo))
                    removidos += 1
                except Exception:
                    pass
    
    return removidos

