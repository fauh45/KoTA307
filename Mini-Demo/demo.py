import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.manifold import TSNE

RANDOM_STATE = 69

"## Mini Demo"
"Berikut ini adalah demo kecil untuk memperlihatkan proses dilakukannya rekomendasi."

"### Pemilihan Model"
"Pilih model mana yang digunakan untuk melakukan rekomendasi (embeddings sudah di komputasi di waktu sebelumnya)"

model_selection = st.selectbox("Model", ("ELMo", "BERT"))


@st.cache_data
def get_model_embeddings(model_selection: str):
    if model_selection == "ELMo":
        file_to_load = "./best_elmo.csv"
    else:
        file_to_load = "./best_bert.csv"

    df = pd.read_csv(file_to_load)
    df.set_index("Lineitem sku", inplace=True)

    df["embeddings"] = df["embeddings"].apply(
        lambda x: np.fromstring(x, sep=",", dtype=float),
    )

    df["product_desc_trunc"] = df["Product description"].str.slice(0, 300)
    df["product_desc_trunc"] = (
        df["product_desc_trunc"]
        .str.wrap(30)
        .apply(lambda x: x.replace("\n", "<br>") + "...")
    )

    return df


@st.cache_data
def calculate_projections(X):
    emb_proj = TSNE(
        n_components=2,
        perplexity=30,
        n_iter=1500,
        learning_rate="auto",
        init="pca",
        random_state=RANDOM_STATE,
    )

    return emb_proj.fit_transform(X)


with st.spinner("Loading Embeddings..."):
    df = get_model_embeddings(model_selection)

with st.spinner("Calculating projection..."):
    emb_proj = calculate_projections(np.array(df["embeddings"].tolist()))

    df["proj_x"] = emb_proj[:, 0]
    df["proj_y"] = emb_proj[:, 1]

fig = px.scatter(
    df,
    x="proj_x",
    y="proj_y",
    color=df.index,
    hover_data=["product_desc_trunc"],
)
st.plotly_chart(fig)
st.caption(
    "Visualisasi dari embeddings semua produk dengan menggunakan t-SNE (perplexity=30, iteration=1500, learning rate=auto)."
)

"### Pemilihan Konsumen"
"Rekomendasi dimulai dari satu konsumen yang akan di rekomendasikan."
