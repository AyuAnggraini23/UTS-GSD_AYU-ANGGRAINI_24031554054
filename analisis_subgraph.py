
from neo4j import GraphDatabase
import pandas as pd

URI = "bolt://127.0.0.1:7687"
AUTH = ("neo4j", "UTS_GSDD")

QUERY = """
MATCH (v:Visitor)-[r]->(i:Item)
WHERE type(r) IN ['VIEWED', 'ADDED_TO_CART', 'BOUGHT']
  AND r.timestamp >= $start
  AND r.timestamp <= $end
RETURN
    count(DISTINCT v) AS jumlah_visitor,
    count(DISTINCT i) AS jumlah_item,
    count(r) AS jumlah_interaksi
"""

def epoch_ms(tanggal):
    return int(
        pd.Timestamp(tanggal, tz="UTC").timestamp() * 1000
    )

periode = {
    "P1 Training": (
        "2015-05-03 00:00:00",
        "2015-05-17 23:59:59"
    ),
    "P2 Testing": (
        "2015-09-01 00:00:00",
        "2015-09-18 23:59:59"
    )
}

driver = GraphDatabase.driver(URI, auth=AUTH)

try:
    driver.verify_connectivity()

    with driver.session(database="neo4j") as session:
        for nama, (awal, akhir) in periode.items():
            hasil = session.run(
                QUERY,
                start=epoch_ms(awal),
                end=epoch_ms(akhir)
            ).single()

            print(f"\n=== {nama} ===")
            print("Jumlah Visitor   :", hasil["jumlah_visitor"])
            print("Jumlah Item      :", hasil["jumlah_item"])
            print("Jumlah Interaksi :", hasil["jumlah_interaksi"])

finally:
    driver.close()
