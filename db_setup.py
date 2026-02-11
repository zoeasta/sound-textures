#!/usr/bin/env python3
"""Create the sound_textures database and textures table in PostgreSQL."""

import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Connection parameters — set DB_PASSWORD env var or edit here
DB_HOST = "localhost"
DB_PORT = 5432
DB_USER = "postgres"
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
DB_NAME = "sound_textures"


def create_database():
    """Create the sound_textures database if it doesn't exist."""
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT,
        user=DB_USER, password=DB_PASSWORD,
        dbname="postgres"
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
    if cur.fetchone() is None:
        cur.execute(f"CREATE DATABASE {DB_NAME}")
        print(f"Database '{DB_NAME}' created.")
    else:
        print(f"Database '{DB_NAME}' already exists.")

    cur.close()
    conn.close()


def create_table():
    """Create the textures table."""
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT,
        user=DB_USER, password=DB_PASSWORD,
        dbname=DB_NAME
    )
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS textures (
            id SERIAL PRIMARY KEY,
            filename TEXT NOT NULL,
            noise_type TEXT NOT NULL,
            base_frequency DOUBLE PRECISION,
            duration DOUBLE PRECISION,
            amplitude DOUBLE PRECISION,
            filter_cutoff DOUBLE PRECISION,
            mod_rate DOUBLE PRECISION,
            mod_depth DOUBLE PRECISION,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    print("Table 'textures' is ready.")

    cur.close()
    conn.close()


if __name__ == "__main__":
    create_database()
    create_table()
