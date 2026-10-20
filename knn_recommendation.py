
import random
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

vsm = load_npz("vsm_visitor_P1.npz").tocsr()

visitor_df = pd.read_csv("indeks_visitor_P1.csv")
item_df = pd.read_csv("indeks_item_P1.csv")

visitor_ids = visitor_df["visitorid"].to_numpy()
item_ids = item_df.iloc[:, 0].to_numpy()

test_df = pd.read_csv("events_P2_testing.csv")

test_df["visitorid"] = test_df["visitorid"].astype(int)
test_df["itemid"] = test_df["itemid"].astype(int)

test_users = set(test_df["visitorid"].unique())
training_users = set(visitor_ids.astype(int))
eligible_users = sorted(test_users & training_users)

if not eligible_users:
    raise ValueError(
        "Tidak ada user yang ditemukan pada P1 dan P2."
    )

random.seed(42)
target_id = random.choice(eligible_users)
target_idx = int(np.where(visitor_ids == target_id)[0][0])

print("User terpilih:", target_id)

relevant_items = set(
    test_df.loc[
        test_df["visitorid"] == target_id, "itemid"
    ].unique()
)

print("Item relevan pengguna pada P2:", sorted(relevant_items))
print(
    "Item relevan tersedia di katalog P1:",
    bool(relevant_items.intersection(set(item_ids)))
)

similarities = cosine_similarity(
    vsm[target_idx], vsm, dense_output=True
).ravel()

similarities[target_idx] = -1

neighbor_indices = np.where(similarities > 0)[0]

if len(neighbor_indices) == 0:
    raise ValueError(
        "User terpilih tidak memiliki tetangga "
        "dengan similarity positif."
    )

neighbor_indices = neighbor_indices[
    np.argsort(similarities[neighbor_indices])[::-1][:5]
]

neighbor_scores = similarities[neighbor_indices]

print("\nLima tetangga KNN:")
for idx in neighbor_indices:
    print(
        int(visitor_ids[idx]),
        round(float(similarities[idx]), 6)
    )

neighbor_matrix = vsm[neighbor_indices]
item_scores = np.asarray(
    neighbor_matrix.T @ neighbor_scores
).ravel()

seen_items = set(
    item_ids[vsm[target_idx].indices].astype(int)
)

candidate_indices = [
    i for i in range(len(item_ids))
    if int(item_ids[i]) not in seen_items
]

ranked_indices = sorted(
    candidate_indices,
    key=lambda i: item_scores[i],
    reverse=True
)

ranked_indices = sorted(
    candidate_indices,
    key=lambda i: item_scores[i],
    reverse=True
)
top5 = [
    int(item_ids[i]) for i in ranked_indices[:5]
]

print("\nLima rekomendasi item:")
for rank, item_id in enumerate(top5, start=1):
    print(rank, "Item ID:", item_id)

print("\nTiga item rekomendasi sesuai soal:")
for rank, item_id in enumerate(top5[:3], start=1):
    print(rank, "Item ID:", item_id)

hits = [
    1 if item in relevant_items else 0
    for item in top5
]

hit_at_5 = int(any(hits))

dcg_at_5 = sum(
    rel / np.log2(rank + 1)
    for rank, rel in enumerate(hits, start=1)
)

ideal_relevances = sorted(hits, reverse=True)
idcg_at_5 = sum(
    rel / np.log2(rank + 1)
    for rank, rel in enumerate(ideal_relevances, start=1)
)

ndcg_at_5 = (
    dcg_at_5 / idcg_at_5 if idcg_at_5 > 0 else 0.0
)

print("\nEvaluasi:")
print("Item relevan pada P2:", len(relevant_items))
print("Hit@5:", hit_at_5)
print("NDCG@5:", round(float(ndcg_at_5), 6))

pd.DataFrame({
    "rank": range(1, len(top5) + 1),
    "itemid": top5,
    "relevan_di_P2": [
        int(item in relevant_items) for item in top5
    ]
}).to_csv("hasil_rekomendasi_KNN.csv", index=False)

print("\nHasil tersimpan di hasil_rekomendasi_KNN.csv")


import numpy as np

top5_items = [80131, 344603, 273326, 287090, 116444]

relevant_items_p2 = [300755]

hit = int(any(item in relevant_items_p2 for item in top5_items))

dcg = sum(
    1 / np.log2(rank + 2)
    for rank, item in enumerate(top5_items)
    if item in relevant_items_p2
)

ideal_relevant = min(len(relevant_items_p2), 5)
idcg = sum(
    1 / np.log2(rank + 2)
    for rank in range(ideal_relevant)
)

ndcg = dcg / idcg if idcg > 0 else 0.0

print("\nEvaluasi Rekomendasi:")
print(f"Item relevan pada P2: {relevant_items_p2}")
print(f"Hit@5: {hit}")
print(f"NDCG@5: {ndcg:.4f}")