import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# título
st.title("Análise e Predição de Adoção de Cães")

# upload do csv
arquivo = st.file_uploader("Envie o arquivo CSV dos pets", type="csv")

if not arquivo:
    st.info("Envie um arquivo CSV para iniciar.")
    st.stop()

# leitura
dados = pd.read_csv(arquivo)

# validação de colunas
colunas = [
    "PetID","PetType","Breed","AgeMonths","Color","Size","WeightKg",
    "Vaccinated","HealthCondition","TimeInShelterDays","AdoptionFee",
    "PreviousOwner","AdoptionLikelihood"
]
faltantes = [c for c in colunas if c not in dados.columns]
if faltantes:
    st.error(f"Colunas faltando: {faltantes}")
    st.stop()

# filtro só cães
dados = dados[dados["PetType"].astype(str).str.lower() == "dog"].copy()
if dados.empty:
    st.warning("Nenhum cão encontrado no CSV.")
    st.stop()

# coerção de tipos básicos
for c in ["AgeMonths","WeightKg","TimeInShelterDays","AdoptionFee"]:
    dados[c] = pd.to_numeric(dados[c], errors="coerce")
for c in ["Vaccinated","HealthCondition","PreviousOwner","AdoptionLikelihood"]:
    dados[c] = pd.to_numeric(dados[c], errors="coerce").fillna(0).astype(int)

# abas
aba_an, aba_ml = st.tabs(["Análise", "Machine Learning"])

# ------------------------- análise -------------------------
with aba_an:
    # filtros
    st.sidebar.header("Filtros")
    portes = sorted(dados["Size"].dropna().unique().tolist())
    raças = sorted(dados["Breed"].dropna().unique().tolist())
    porte_sel = st.sidebar.multiselect("Porte", portes, default=portes)
    raça_sel = st.sidebar.multiselect("Raça", raças, default=raças)
    vacinado_sel = st.sidebar.selectbox("Vacinado", ["todos", "vacinado", "não vacinado"])
    saude_sel = st.sidebar.selectbox("Saúde", ["todos", "saudável", "condição médica"])

    # aplica filtros
    base = dados[dados["Size"].isin(porte_sel) & dados["Breed"].isin(raça_sel)].copy()
    if vacinado_sel != "todos":
        base = base[base["Vaccinated"] == (1 if vacinado_sel == "vacinado" else 0)]
    if saude_sel != "todos":
        base = base[base["HealthCondition"] == (0 if saude_sel == "saudável" else 1)]

    # métricas
    col1, col2, col3 = st.columns(3)
    col1.metric("total de cães", len(base))
    col2.metric("prop. provável adoção", f'{base["AdoptionLikelihood"].mean():.2%}')
    col3.metric("tempo médio no abrigo (dias)", f'{base["TimeInShelterDays"].mean():.1f}')

    # prévia
    st.subheader("Prévia dos Dados (filtrados)")
    st.dataframe(base.head(20))

    # gráficos
    st.subheader("Gráficos de Análise")
    base["ClasseAdoção"] = base["AdoptionLikelihood"].map({0:"improvável", 1:"provável"})

    st.caption("adoção por porte")
    g1 = base.groupby("Size")["AdoptionLikelihood"].mean().reset_index().rename(columns={"AdoptionLikelihood":"TaxaAdoção"})
    st.plotly_chart(px.bar(g1, x="Size", y="TaxaAdoção", text="TaxaAdoção", range_y=[0,1]), use_container_width=True)

    st.caption("idade (meses) por classe de adoção")
    st.plotly_chart(px.box(base, x="ClasseAdoção", y="AgeMonths"), use_container_width=True)

    st.caption("tempo no abrigo por classe de adoção")
    st.plotly_chart(px.histogram(base, x="TimeInShelterDays", color="ClasseAdoção", barmode="overlay", nbins=30, opacity=0.7), use_container_width=True)

    st.caption("peso x idade (cor = adoção)")
    st.plotly_chart(px.scatter(base, x="AgeMonths", y="WeightKg", color="ClasseAdoção", hover_data=["Breed","Size","PreviousOwner","Vaccinated"]), use_container_width=True)

    st.caption("top 10 raças por taxa de adoção (mín. 20 cães)")
    g5 = base.groupby("Breed").agg(qtd=("Breed","size"), taxa=("AdoptionLikelihood","mean")).reset_index()
    g5 = g5[g5["qtd"] >= 20].sort_values("taxa", ascending=False).head(10)
    if not g5.empty:
        st.plotly_chart(px.bar(g5, x="taxa", y="Breed", orientation="h", text="taxa", range_x=[0,1]), use_container_width=True)
    else:
        st.info("não há raças com amostra mínima para ranking.")

    st.caption("impacto de vacinação e saúde na adoção")
    g6 = base.groupby(["Vaccinated","HealthCondition"])["AdoptionLikelihood"].mean().reset_index()
    g6["Vac"] = g6["Vaccinated"].map({0:"não vacinado",1:"vacinado"})
    g6["Saúde"] = g6["HealthCondition"].map({0:"saudável",1:"condição médica"})
    st.plotly_chart(px.bar(g6, x="Vac", y="AdoptionLikelihood", color="Saúde", barmode="group", range_y=[0,1]), use_container_width=True)

    st.caption("correlação entre variáveis numéricas")
    num_cols = ["AgeMonths","WeightKg","TimeInShelterDays","AdoptionFee","PreviousOwner","Vaccinated","HealthCondition","AdoptionLikelihood"]
    corr = base[num_cols].corr().round(2)
    st.plotly_chart(px.imshow(corr, text_auto=True, aspect="auto", zmin=-1, zmax=1, color_continuous_scale="RdBu"), use_container_width=True)

# ------------------------- machine learning -------------------------
with aba_ml:
    st.subheader("Treino e Predição")

    # separa X e y a partir de todos os cães (sem filtros de análise)
    df_ml = dados.dropna(subset=["AdoptionLikelihood"]).copy()
    y = df_ml["AdoptionLikelihood"].astype(int)
    X = df_ml.drop(columns=["AdoptionLikelihood","PetID","PetType"])

    # define colunas
    cat_cols = ["Breed","Color","Size"]
    num_cols = ["AgeMonths","WeightKg","TimeInShelterDays","AdoptionFee","Vaccinated","HealthCondition","PreviousOwner"]

    # pré-processamento
    # imputa, codifica, escala
    num_pipe = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ])
    cat_pipe = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore"))
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )

    # escolha de modelo
    modelo_nome = st.selectbox("Modelo", ["DecisionTree", "RandomForest", "KNN", "LogisticRegression"])

    # hiperparâmetros simples
    if modelo_nome == "DecisionTree":
        max_depth = st.slider("max_depth", 1, 30, 8)
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    elif modelo_nome == "RandomForest":
        n_estimators = st.slider("n_estimators", 50, 500, 200, step=50)
        max_depth = st.slider("max_depth", 1, 30, 12)
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
    elif modelo_nome == "KNN":
        n_neighbors = st.slider("n_neighbors", 1, 25, 7)
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    else:
        C = st.slider("C (regularização)", 0.01, 5.0, 1.0)
        clf = LogisticRegression(C=C, max_iter=1000, n_jobs=-1)

    # split
    test_size = st.slider("tamanho do teste", 0.1, 0.5, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    # pipeline final
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])

    # treino
    if st.button("Treinar modelo"):
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # métricas
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cmat = confusion_matrix(y_test, y_pred)

        m1, m2 = st.columns(2)
        m1.metric("acurácia", f"{acc:.3f}")
        m2.metric("f1-score", f"{f1:.3f}")

        # matriz de confusão
        st.caption("matriz de confusão")
        cm_df = pd.DataFrame(cmat, index=["real:0","real:1"], columns=["pred:0","pred:1"])
        st.plotly_chart(px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues"), use_container_width=True)

        # guarda no estado
        st.session_state["modelo_treinado"] = pipe

    st.markdown("---")

    # formulário de predição
    st.subheader("Predição para novo cão")

    with st.form("form_pred"):
        # inputs simples baseados nas colunas
        raça_i = st.selectbox("Raça", sorted(dados["Breed"].dropna().unique().tolist()))
        cor_i = st.selectbox("Cor", sorted(dados["Color"].dropna().unique().tolist()))
        porte_i = st.selectbox("Porte", sorted(dados["Size"].dropna().unique().tolist()))
        idade_i = st.number_input("Idade (meses)", min_value=0.0, step=1.0, value=float(dados["AgeMonths"].median()))
        peso_i = st.number_input("Peso (kg)", min_value=0.0, step=0.1, value=float(dados["WeightKg"].median()))
        tempo_i = st.number_input("Tempo no abrigo (dias)", min_value=0.0, step=1.0, value=float(dados["TimeInShelterDays"].median()))
        taxa_i = st.number_input("Taxa de adoção (USD)", min_value=0.0, step=1.0, value=float(dados["AdoptionFee"].median()))
        vac_i = st.selectbox("Vacinado", ["não", "sim"])
        saude_i = st.selectbox("Saúde", ["saudável", "condição médica"])
        dono_i = st.selectbox("Teve dono antes", ["não", "sim"])
        enviar = st.form_submit_button("Prever")

    # executa predição
    if enviar:
        if "modelo_treinado" not in st.session_state:
            st.warning("treine um modelo antes de prever.")
        else:
            novo = pd.DataFrame([{
                "Breed": raça_i,
                "Color": cor_i,
                "Size": porte_i,
                "AgeMonths": idade_i,
                "WeightKg": peso_i,
                "TimeInShelterDays": tempo_i,
                "AdoptionFee": taxa_i,
                "Vaccinated": 1 if vac_i == "sim" else 0,
                "HealthCondition": 0 if saude_i == "saudável" else 1,
                "PreviousOwner": 1 if dono_i == "sim" else 0
            }])
            modelo = st.session_state["modelo_treinado"]
            if hasattr(modelo, "predict_proba"):
                prob = float(modelo.predict_proba(novo)[0,1])
                st.metric("probabilidade de adoção", f"{prob:.1%}")
            pred = int(modelo.predict(novo)[0])
            st.success("previsto: provável adoção" if pred == 1 else "previsto: improvável adoção")

