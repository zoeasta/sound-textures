#!/usr/bin/env python3
"""Insert generated texture metadata into PostgreSQL from manifest.json."""

import os
import json
import psycopg2

# Connection parameters — set DB_PASSWORD env var or edit here
DB_HOST = "localhost"
DB_PORT = 5432
DB_USER = "postgres"
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
DB_NAME = "sound_textures"

MANIFEST_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "output", "manifest.json"
)


def insert_textures(manifest_path=MANIFEST_PATH):
    """Read manifest.json and insert each texture's params into the DB."""
    with open(manifest_path) as f:
        textures = json.load(f)

    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT,
        user=DB_USER, password=DB_PASSWORD,
        dbname=DB_NAME
    )
    cur = conn.cursor()

    insert_sql = """
        INSERT INTO textures
            (filename, noise_type, base_frequency, duration,
             amplitude, filter_cutoff, mod_rate, mod_depth)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """

    count = 0
    skipped = 0
    for t in textures:
        # Skip if filename already exists in DB
        cur.execute("SELECT 1 FROM textures WHERE filename = %s",
                    (t["filename"],))
        if cur.fetchone() is not None:
            skipped += 1
            continue

        cur.execute(insert_sql, (
            t["filename"],
            t["noise_type"],
            t["base_frequency"],
            t["duration"],
            t["amplitude"],
            t["filter_cutoff"],
            t["mod_rate"],
            t["mod_depth"],
        ))
        count += 1

    conn.commit()
    print(f"Inserted {count} textures into database.", end="")
    if skipped:
        print(f" ({skipped} duplicates skipped.)", end="")
    print()

    cur.close()
    conn.close()


if __name__ == "__main__":
    insert_textures()
