import requests
import json

BASE_URL = "http://localhost:5000"

def test_home():
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status: {response.status_code}")
        print(f"Resposta: {json.dumps(response.json(), indent=2)}")
        print()
        return True
    except Exception as e:
        print(f"Erro: {e}\n")
        return False


def test_list_models():
    try:
        response = requests.get(f"{BASE_URL}/api/models")
        print(f"Status: {response.status_code}")
        print(f"Resposta: {json.dumps(response.json(), indent=2)}")
        print()
        return response.json()
    except Exception as e:
        print(f"Erro: {e}\n")
        return None


def test_predict():
    data = {
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
    }
    try:
        response = requests.post(
            f"{BASE_URL}/api/predict",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status: {response.status_code}")
        print(f"Resposta: {json.dumps(response.json(), indent=2)}")
        print()
        return True
    except Exception as e:
        print(f"Erro: {e}\n")
        return False


def test_predict_batch_json():
    data = [
        {
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
        },
        {
            "Breed": "Golden Retriever",
            "Color": "Golden",
            "Size": "Medium",
            "AgeMonths": 12,
            "WeightKg": 10.0,
            "TimeInShelterDays": 60,
            "AdoptionFee": 150,
            "Vaccinated": 0,
            "HealthCondition": 1,
            "PreviousOwner": 1
        }
    ]
    try:
        response = requests.post(
            f"{BASE_URL}/api/predict/batch",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status: {response.status_code}")
        print(f"Resposta: {json.dumps(response.json(), indent=2)}")
        print()
        return True
    except Exception as e:
        print(f"Erro: {e}\n")
        return False


def main():
    try:
        if not test_home():
            print("Falha ao conectar à API. Verifique se está rodando.")
            return
        
        models = test_list_models()
        
        if models and models.get("total", 0) > 0:
            test_predict()
            test_predict_batch_json()
            print("Todos os testes concluídos!")
        else:
            print("Nenhum modelo disponível.")
            print("Treine um modelo primeiro via Streamlit")
    
    except requests.exceptions.ConnectionError:
        print("Erro: Não foi possível conectar à API.")
    except Exception as e:
        print(f"Erro inesperado: {e}")


if __name__ == "__main__":
    main()

