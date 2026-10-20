
import os
import pandas as pd
from tqdm import tqdm
from neo4j import GraphDatabase

URI = "bolt://127.0.0.1:7687"
AUTH = ("neo4j", "UTS_GSDD")

CSV_FILE = "events.csv"
BATCH_SIZE = 10000


def import_batch(tx, batch):
    query = """
    UNWIND $batch AS row
    MERGE (v:Visitor {visitorId: row.visitorid})
    MERGE (i:Item {itemId: row.itemid})
    WITH v, i, row
    CALL {
        WITH v, i, row
        WITH v, i, row WHERE row.event = 'view'
        CREATE (v)-[:VIEWED {timestamp: row.timestamp}]->(i)
        UNION ALL
        WITH v, i, row
        WITH v, i, row WHERE row.event = 'addtocart'
        CREATE (v)-[:ADDED_TO_CART {timestamp: row.timestamp}]->(i)
        UNION ALL
        WITH v, i, row
        WITH v, i, row WHERE row.event = 'transaction'
        CREATE (v)-[:BOUGHT {
            timestamp: row.timestamp,
            transactionId: row.transactionid
        }]->(i)
    }
    """
    tx.run(query, batch=batch)


def main():
    if not os.path.exists(CSV_FILE):
        print(f"File tidak ditemukan: {CSV_FILE}")
        return

    driver = GraphDatabase.driver(URI, auth=AUTH)

    try:
        driver.verify_connectivity()
        print("Berhasil terhubung ke Neo4j.")

        with driver.session(database="neo4j") as session:
            session.run("""
                CREATE CONSTRAINT visitor_id_uniq IF NOT EXISTS
                FOR (v:Visitor) REQUIRE v.visitorId IS UNIQUE
            """).consume()

            session.run("""
                CREATE CONSTRAINT item_id_uniq IF NOT EXISTS
                FOR (i:Item) REQUIRE i.itemId IS UNIQUE
            """).consume()

            total_rows = sum(
                1 for _ in open(CSV_FILE, encoding="utf-8")
            ) - 1

            total_chunks = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE

            chunks = pd.read_csv(
                CSV_FILE, chunksize=BATCH_SIZE
            )

            print(f"Total baris: {total_rows:,}")
            print(f"Ukuran batch: {BATCH_SIZE:,}")
            print("Memulai impor...")

            for chunk in tqdm(
                chunks,
                total=total_chunks,
                desc="Mengimpor data"
            ):
                chunk["transactionid"] = (
                    chunk["transactionid"]
                    .fillna("")
                    .astype(str)
                )

                records = chunk.to_dict(orient="records")
                session.write_transaction(import_batch, records)

        print("Impor selesai!")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
