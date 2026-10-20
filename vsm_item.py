
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse import csr_matrix, save_npz

df = pd.read_csv("events_P1_training.csv")

interaksi = df[["visitorid", "itemid"]].drop_duplicates()

item_ids = pd.Index(interaksi["itemid"].unique())
visitor_ids = pd.Index(interaksi["visitorid"].unique())

baris = item_ids.get_indexer(interaksi["itemid"])
kolom = visitor_ids.get_indexer(interaksi["visitorid"])

vsm_item = csr_matrix(
    (
        np.ones(len(interaksi), dtype=np.int8),
        (baris, kolom)
    ),
    shape=(len(item_ids), len(visitor_ids))
)

total_elemen = vsm_item.shape[0] * vsm_item.shape[1]

print(" KARAKTERISTIK VSM ITEM ")
print("Jumlah Item           :", len(item_ids))
print("Jumlah Visitor        :", len(visitor_ids))
print("Jumlah pasangan unik  :", len(interaksi))
print("Shape matriks         :", vsm_item.shape)
print("Jumlah dimensi        :", len(vsm_item.shape))
print("Jumlah elemen matriks :", total_elemen)
print("Jumlah nilai nonzero  :", vsm_item.nnz)
print("Kerapatan matriks     :", round(
    vsm_item.nnz / total_elemen, 8
))

sampel_item = item_ids[:15]
sampel_set = set(sampel_item)

sampel_interaksi = interaksi[
    interaksi["itemid"].isin(sampel_set)
]

G = nx.Graph()

for item in sampel_item:
    G.add_node(f"I-{item}", bipartite="item")

visitor_sampel = sampel_interaksi["visitorid"].unique()

for visitor in visitor_sampel:
    G.add_node(f"V-{visitor}", bipartite="visitor")

for row in sampel_interaksi.itertuples(index=False):
    G.add_edge(f"I-{row.itemid}", f"V-{row.visitorid}")

print("\n GRAF NETWORKX (SAMPEL) ")
print("Jumlah node graf :", G.number_of_nodes())
print("Jumlah edge graf :", G.number_of_edges())

pos = nx.bipartite_layout(
    G,
    [n for n, data in G.nodes(data=True)
     if data["bipartite"] == "item"]
)

plt.figure(figsize=(14, 8))
nx.draw(
    G,
    pos,
    with_labels=False,
    node_size=35,
    width=0.5
)
plt.title("Graf Bipartit Item-Visitor (Sampel P1 Training)")
plt.savefig("plot_vsm_item.png", dpi=200)
plt.show()

save_npz("vsm_item_P1.npz", vsm_item)

pd.Series(item_ids, name="itemid").to_csv(
    "indeks_item_VSM_P1.csv", index=False
)

pd.Series(visitor_ids, name="visitorid").to_csv(
    "indeks_visitor_VSM_P1.csv", index=False
)

print("\nFile VSM Item berhasil disimpan.")
