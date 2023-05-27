import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
import random

from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances

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


def description_cleaner(description: str):
    line = re.sub(r"-+", " ", description)
    line = re.sub(r"[^a-zA-Z0-9, ]+", " ", line)
    line = re.sub(r"[ ]+", " ", line)
    line += "."

    return line.strip()


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
df_dataset = pd.read_csv("./cleaned_main_dataset.csv")
df_dataset = df_dataset.loc[:, ~df_dataset.columns.str.contains("^Unnamed")]
df_dataset["Product description"] = df_dataset["Product description"].apply(
    description_cleaner
)

unique_buyer = df_dataset["Email"].value_counts(dropna=True)
unique_buyer = unique_buyer[unique_buyer >= 3]

unique_buyer_list = unique_buyer.index.to_list()
random.shuffle(unique_buyer_list)

chosen_user = st.selectbox("User Email (Anonymized)", unique_buyer_list)

"### Seed Product"
"Basis dari rekomendasi suatu konsumen adalah produk terakhir yang dibelinya. Untuk validasi, yang diambil untuk demontrasi adalah produk pertama"

customer_bought = df_dataset[df_dataset["Email"] == chosen_user]
seed_product = customer_bought.iloc[0].copy()

customer_bought = customer_bought.tail(-1)

seed_product

ground_truth = customer_bought.head(3)

"### Ground Truth"
"3 produk selanjutnya yang dibeli oleh konsumen."

ground_truth

"### Hasil Rekomendasi"
"Hasil rekomendasi dari perbandingan embeddings seed product dengan produk lainnya, diambil 3 produk dengan cosine distance paling kecil (paling similar)"

seed_embeddings = df.at[seed_product["Lineitem sku"], "embeddings"]
seed_embeddings = np.reshape(seed_embeddings, (-1, seed_embeddings.shape[0]))

embeddings_distance = (
    df.loc[df.index != seed_product["Lineitem sku"]]["embeddings"]
    .apply(lambda x: x.reshape(1, -1))
    .apply(lambda emb2: cosine_distances(seed_embeddings, emb2))
    .sort_values(ascending=False)
    .head(3)
)

selected_product = df[df.index.isin(embeddings_distance.index)]

selected_product

selected_product["label"] = "Selected"

ground_truth_product = df[df.index.isin(ground_truth["Lineitem sku"])]
ground_truth_product["label"] = "Ground Truth"

seed_product_product = df[df.index == seed_product["Lineitem sku"]]
seed_product_product["label"] = "Seed Product"

all_shown_product_combined = pd.concat([selected_product, ground_truth_product, seed_product_product])

fig = px.scatter(
    all_shown_product_combined,
    x="proj_x",
    y="proj_y",
    color="label",
    hover_data=["product_desc_trunc"],
)
st.plotly_chart(fig)
st.caption(
    "Visualisasi dari embeddings produk yang direkomendasi dengan menggunakan t-SNE (projeksi masih sama dengan visualisasi sebelumnya, visualisasi ini hanyalah subset pada produk yang direkomendasikan)."
)