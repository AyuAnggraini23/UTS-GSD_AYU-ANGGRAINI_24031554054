
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
from neo4j import GraphDatabase

vsm = load_npz("vsm_visitor_P1.npz").tocsr()
visitor_df = pd.read_csv("indeks_visitor_P1.csv")

print("Kolom indeks Visitor:", visitor_df.columns.tolist())

visitor_ids = visitor_df.iloc[:, 0].to_numpy()

jumlah_interaksi = np.asarray(vsm.sum(axis=1)).ravel()
ref_idx = int(np.argmax(jumlah_interaksi))
ref_id = int(visitor_ids[ref_idx])

print("\nVisitor acuan:", ref_id)
print("Jumlah interaksi:", int(jumlah_interaksi[ref_idx]))

similarities = cosine_similarity(
    vsm[ref_idx], vsm, dense_output=True
).ravel()

similarities[ref_idx] = -1

top_idx = np.argsort(similarities)[::-1][:5]

hasil = []
for idx in top_idx:
    hasil.append({
        "visitor_acuan": ref_id,
        "visitor_mirip": int(visitor_ids[idx]),
        "similarity": float(similarities[idx])
    })

hasil_df = pd.DataFrame(hasil)

print("\nLima Visitor dengan similarity tertinggi:")
print(hasil_df.to_string(index=False))

hasil_df.to_csv("hasil_similarity_visitor.csv", index=False)

URI = "bolt://127.0.0.1:7687"
USERNAME = "neo4j"
PASSWORD = input("UTS_GSDD: ")

driver = GraphDatabase.driver(
    URI, auth=(USERNAME, PASSWORD)
)

query = """
MATCH (u:Visitor {visitorId: $ref_id})
MATCH (v:Visitor {visitorId: $other_id})
MERGE (u)-[r:SIMILAR_TO]->(v)
SET r.similarity = $similarity
"""

with driver.session() as session:
    for row in hasil:
        session.run(query, **{
            "ref_id": row["visitor_acuan"],
            "other_id": row["visitor_mirip"],
            "similarity": row["similarity"]
        })

driver.close()

print("\nNilai similarity berhasil ditulis ke Neo4j.")
print("Hasil juga tersimpan di hasil_similarity_visitor.csv")
