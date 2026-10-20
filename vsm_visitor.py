
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse import csr_matrix
from pathlib import Path

df = pd.read_csv("events_P1_training.csv")

interaksi = df[["visitorid", "itemid"]].drop_duplicates()

visitor_ids = pd.Index(interaksi["visitorid"].unique())
item_ids = pd.Index(interaksi["itemid"].unique())

baris = visitor_ids.get_indexer(interaksi["visitorid"])
kolom = item_ids.get_indexer(interaksi["itemid"])

vsm_user = csr_matrix(
    (
        np.ones(len(interaksi), dtype=np.int8),
        (baris, kolom)
    ),
    shape=(len(visitor_ids), len(item_ids))
)

print(" KARAKTERISTIK VSM USER ")
print("Jumlah Visitor        :", len(visitor_ids))
print("Jumlah Item            :", len(item_ids))
print("Jumlah pasangan unik   :", len(interaksi))
print("Shape matriks          :", vsm_user.shape)
print("Jumlah dimensi         :", len(vsm_user.shape))
print("Jumlah elemen matriks  :", vsm_user.shape[0] * vsm_user.shape[1])
print("Jumlah nilai nonzero   :", vsm_user.nnz)
print("Kerapatan matriks      :", round(
    vsm_user.nnz / (vsm_user.shape[0] * vsm_user.shape[1]), 8
))

sampel_visitor = visitor_ids[:15]
sampel_set = set(sampel_visitor)

sampel_interaksi = interaksi[
    interaksi["visitorid"].isin(sampel_set)
]

G = nx.Graph()

for visitor in sampel_visitor:
    G.add_node(f"V-{visitor}", bipartite="visitor")

item_sampel = sampel_interaksi["itemid"].unique()

for item in item_sampel:
    G.add_node(f"I-{item}", bipartite="item")

for row in sampel_interaksi.itertuples(index=False):
    G.add_edge(f"V-{row.visitorid}", f"I-{row.itemid}")

print("\n=== GRAF NETWORKX (SAMPEL) ===")
print("Jumlah node graf :", G.number_of_nodes())
print("Jumlah edge graf :", G.number_of_edges())

pos = nx.bipartite_layout(
    G,
    [n for n, data in G.nodes(data=True)
     if data["bipartite"] == "visitor"]
)

plt.figure(figsize=(14, 8))
nx.draw(
    G,
    pos,
    with_labels=False,
    node_size=35,
    width=0.5
)
plt.title("Graf Bipartit Visitor-Item (Sampel P1 Training)")
plt.tight_layout()
plt.savefig("plot_vsm_visitor.png", dpi=200)
plt.show()

from scipy.sparse import save_npz

save_npz("vsm_visitor_P1.npz", vsm_user)
pd.Series(visitor_ids, name="visitorid").to_csv(
    "indeks_visitor_P1.csv", index=False
)
pd.Series(item_ids, name="itemid").to_csv(
    "indeks_item_P1.csv", index=False
)

print("\nFile VSM dan indeks berhasil disimpan.")
