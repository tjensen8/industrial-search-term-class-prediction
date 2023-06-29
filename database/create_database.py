"""
Uploads Grainger data to sqlite for more efficient data processing.
"""
import sqlite3 as sql
import argparse
import csv
from tqdm import tqdm
from typing import List, AnyStr


def make_db(loc: str, table_name: str) -> sql.connect:
    # also makes a copy of the database if it doesn't exist already
    conn = sql.connect(loc)
    c = conn.cursor()

    # enforce caps, string
    table_name = str(table_name).upper()

    # drop the table info
    c.execute(f"""DROP TABLE IF EXISTS {table_name}""")

    # create the table if it doesn't exist
    c.execute(
        f"""CREATE TABLE IF NOT EXISTS {table_name} (search_term TEXT, class TEXT);"""
    )
    c.close()
    return conn


def extract_data_from_file(
    loc: str, delimeter: str = "\t", columns: List[AnyStr] = ["search_term", "class"]
) -> csv.reader:
    # returns a csv reader to be run line by line
    # with open(loc, "r") as f:
    f = open(loc, "r")
    # reader = csv.DictReader(f, delimiter=delimeter)
    reader = csv.reader(f, delimiter=delimeter)
    return reader, f


def upload_data_to_db(
    reader: csv.reader, conn: sql.connect, table_name: str, file: open
) -> bool:
    table_name = str(table_name).upper()

    # iterates lines by line and uploads data to database
    c = conn.cursor()
    iters = len(list(reader))
    for row in tqdm(reader, total=iters):
        c.execute(
            f"""
        
        INSERT INTO "{table_name}" values ("{row[0]}", "{row[1]}")
        
        """
        )
    c.close()
    file.close()

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
    reader, file = extract_data_from_file(args.r)
    resp = upload_data_to_db(reader, conn, args.table, file)

    if resp:
        print("Data uploaded successfully.")
    else:
        print("[!Error] Data Not Uploaded Successfully")
