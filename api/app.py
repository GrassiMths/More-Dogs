from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
import sys
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_utils import listar_modelos, carregar_modelo

app = Flask(__name__)
CORS(app)

@app.route('/api/models', methods=['GET'])
def get_models():
    try:
        modelos = listar_modelos()
        return jsonify({
            "success": True,
            "total": len(modelos),
            "models": [
                {
                    "arquivo": m["arquivo"],
                    "nome": m["nome"],
                    "acurácia": m["acurácia"],
                    "f1_score": m["f1_score"],
                    "timestamp": m["timestamp"]
                }
                for m in modelos
            ]
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/models/<arquivo>', methods=['GET'])
def get_model_info(arquivo):
    try:
        modelo_info = carregar_modelo(arquivo)
        if modelo_info is None:
            return jsonify({
                "success": False,
                "error": "Modelo não encontrado"
            }), 404
        
        return jsonify({
            "success": True,
            "modelo": {
                "nome": modelo_info.get("nome", "Desconhecido"),
                "acurácia": modelo_info.get("acurácia", 0),
                "f1_score": modelo_info.get("f1_score", 0),
                "timestamp": modelo_info.get("timestamp", "")
            }
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "Body JSON necessário"
            }), 400
        
        modelo_identificador = data.get("modelo")
        if modelo_identificador is None:
            # Usa o modelo mais recente se não especificado
            modelos = listar_modelos()
            if not modelos:
                return jsonify({
                    "success": False,
                    "error": "Nenhum modelo disponível"
                }), 404
            modelo_info = joblib.load(modelos[0]["caminho"])
        else:
            modelo_info = carregar_modelo(modelo_identificador)
            if modelo_info is None:
                return jsonify({
                    "success": False,
                    "error": "Modelo não encontrado"
                }), 404
        
        modelo = modelo_info["modelo"]
        
        features = {
            "Breed": data.get("Breed"),
            "Color": data.get("Color"),
            "Size": data.get("Size"),
            "AgeMonths": data.get("AgeMonths"),
            "WeightKg": data.get("WeightKg"),
            "TimeInShelterDays": data.get("TimeInShelterDays"),
            "AdoptionFee": data.get("AdoptionFee"),
            "Vaccinated": data.get("Vaccinated"),
            "HealthCondition": data.get("HealthCondition"),
            "PreviousOwner": data.get("PreviousOwner")
        }
        
        # Valida campos obrigatórios
        campos_obrigatorios = [
            "Breed", "Color", "Size", "AgeMonths", "WeightKg",
            "TimeInShelterDays", "AdoptionFee", "Vaccinated",
            "HealthCondition", "PreviousOwner"
        ]
        faltantes = [c for c in campos_obrigatorios if features[c] is None]
        if faltantes:
            return jsonify({
                "success": False,
                "error": f"Campos obrigatórios faltando: {faltantes}"
            }), 400
        
        df = pd.DataFrame([features])
        
        # Faz predição
        predicao = int(modelo.predict(df)[0])
        probabilidade = None
        if hasattr(modelo, "predict_proba"):
            probabilidade = float(modelo.predict_proba(df)[0, 1])
        
        return jsonify({
            "success": True,
            "predição": predicao,
            "predição_texto": "Provável" if predicao == 1 else "Improvável",
            "probabilidade_adoção": probabilidade,
            "modelo_usado": modelo_info["nome"]
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    try:
        # Verifica se é CSV
        if 'file' in request.files:
            arquivo = request.files['file']
            if arquivo.filename.endswith('.csv'):
                df = pd.read_csv(arquivo)
            else:
                return jsonify({
                    "success": False,
                    "error": "Arquivo deve ser CSV"
                }), 400
        # Verifica se é JSON
        elif request.is_json:
            data = request.get_json()
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                return jsonify({
                    "success": False,
                    "error": "JSON deve ser um array de objetos"
                }), 400
        else:
            return jsonify({
                "success": False,
                "error": "Envie um arquivo CSV ou JSON array"
            }), 400
        
        # Obtém modelo
        modelo_identificador = request.form.get('modelo') or (request.get_json() or {}).get('modelo')
        if modelo_identificador is None:
            modelos = listar_modelos()
            if not modelos:
                return jsonify({
                    "success": False,
                    "error": "Nenhum modelo disponível"
                }), 404
            modelo_info = joblib.load(modelos[0]["caminho"])
        else:
            modelo_info = carregar_modelo(modelo_identificador)
            if modelo_info is None:
                return jsonify({
                    "success": False,
                    "error": "Modelo não encontrado"
                }), 404
        
        modelo = modelo_info["modelo"]
        
        # Valida colunas necessárias
        colunas_necessarias = [
            "Breed", "Color", "Size", "AgeMonths", "WeightKg",
            "TimeInShelterDays", "AdoptionFee", "Vaccinated",
            "HealthCondition", "PreviousOwner"
        ]
        faltantes = [c for c in colunas_necessarias if c not in df.columns]
        if faltantes:
            return jsonify({
                "success": False,
                "error": f"Colunas faltando: {faltantes}"
            }), 400
        
        # Prepara dados (remove colunas extras)
        X = df[colunas_necessarias].copy()
        
        # Faz predições
        predicoes = modelo.predict(X)
        probabilidades = None
        if hasattr(modelo, "predict_proba"):
            probabilidades = modelo.predict_proba(X)[:, 1].tolist()
        
        # Monta resultado
        resultados = []
        for idx, pred in enumerate(predicoes):
            resultado = {
                "predição": int(pred),
                "predição_texto": "Provável" if pred == 1 else "Improvável"
            }
            if probabilidades:
                resultado["probabilidade_adoção"] = probabilidades[idx]
            if "PetID" in df.columns:
                resultado["PetID"] = df.iloc[idx]["PetID"]
            resultados.append(resultado)
        
        return jsonify({
            "success": True,
            "total": len(resultados),
            "modelo_usado": modelo_info["nome"],
            "resultados": resultados
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

