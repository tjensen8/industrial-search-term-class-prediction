"""
Uploads Grainger data to sqlite for more efficient data processing.
"""
import sqlite3 as sql
import argparse
import pandas as pd
from tqdm import tqdm
import logging

logging.basicConfig(
    filename="/home/programming/dsa-g/create_db.log", level=logging.INFO, filemode="w"
)


def make_db(loc: str, table_name: str) -> sql.connect:
    # also makes a copy of the database if it doesn't exist already
    conn = sql.connect(loc)
    c = conn.cursor()

    # enforce caps, string
    table_name = str(table_name).upper()

    # drop the table info
    c.execute(f"""DROP TABLE IF EXISTS {table_name}""")
    logging.info(f"Dropped table {table_name} from database located at {loc}")

    # create the table if it doesn't exist
    c.execute(
        f"""CREATE TABLE IF NOT EXISTS {table_name} (search_term TEXT, class TEXT);"""
    )
    logging.info(f"Recreated {table_name} in database located at {loc}")

    c.close()
    return conn


def upload_data_to_db(
    conn: sql.connect, table_name: str, loc: str, delimeter="\t", chunksize=5000
) -> bool:
    table_name = str(table_name).upper()

    # chunk large file for upload to database
    logging.info(f"Reading in file located at {loc}")
    chunks = pd.read_csv(loc, delimiter="\t", chunksize=chunksize, header=None)
    n_chunks = round(sum(1 for row in open(loc, "r")) / chunksize)
    logging.info(f"Number of Chunks Found: {n_chunks}")
    for chunk_num, chunk in enumerate(chunks):
        logging.info(f"Processing pandas chunk: {chunk_num} / {n_chunks}")
        # iteratively add the chunks to the database
        chunk.rename(columns={0: "search_term", 1: "class"}, inplace=True)
        chunk.to_sql(table_name, con=conn, if_exists="append", index=False)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Grainger Database Creator",
        description="Creates a database for Grainger search query text and associated classes.",
    )
    parser.add_argument(
        "--db",
        default="/home/programming/dsa-g/data/grainger.db",
        help="Database name as a path.",
    )

    parser.add_argument(
        "--table",
        default="search_terms",
        help="The name of the table containing the Grainger data.",
    )

    parser.add_argument(
        "--r",
        "--raw_data",
        default="/home/programming/dsa-g/data/data_train.tsv",
        help="The location of the raw data to be loaded into the database.",
    )

    args = parser.parse_args()

    conn = make_db(args.db, args.table)
    logging.info("Database created.")
    resp = upload_data_to_db(conn, args.table, args.r)

    if resp:
        print("Data uploaded successfully.")
        logging.info(f"Data uploaded successfully to database at {args.db}")
    else:
        print("[!Error] Data Not Uploaded Successfully")
        logging.error("Data Not Uploaded Successfully")
