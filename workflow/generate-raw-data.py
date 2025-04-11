import pandas as pd
from pathlib import Path
import time
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
from typing import List
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl
import polars.datatypes as pl_types

# Get institution ID from snakemake config
INSTITUTION_IDS = snakemake.params["institution_ids"]

db_name = snakemake.config["db_name"]
db_password = snakemake.config["db_password"]
db_user = snakemake.config["db_user"]
db_port = snakemake.config["db_port"]

# Database connection parameters
db_params = {
    'dbname': db_name,
    'user': db_user,
    'password': db_password,
    'host': 'localhost',
    'port': db_port
}

engine = create_engine(
    f"postgresql://{db_params['user']}:{quote_plus(db_params['password'])}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}"
)

# Format institution IDs for SQL query
institution_ids_str = "', '".join(INSTITUTION_IDS)

SETUP_QUERIES = [
    f"""
    -- First create temporary table of institution-affiliated authors
    CREATE TEMP TABLE affiliated_authors AS
    SELECT DISTINCT openalex_author_id
    FROM (
        SELECT author_id as openalex_author_id
        FROM openalex.works_authorships
        WHERE institution_id IN ('{institution_ids_str}')
    ) AS authors;

    CREATE INDEX ON affiliated_authors(openalex_author_id);
    """,
    """
    -- Create temporary table for author mapping
    CREATE TEMP TABLE author_mapping AS
    SELECT
        a.id as openalex_author_id,
        ROW_NUMBER() OVER (ORDER BY a.id) - 1 as author_id
    FROM openalex.authors a
    WHERE a.id IN (SELECT openalex_author_id FROM affiliated_authors);

    CREATE INDEX ON author_mapping(openalex_author_id);
    CREATE INDEX ON author_mapping(author_id);
    """,
    """
    -- Create temporary table for paper mapping
    -- This includes ALL papers by affiliated authors
    CREATE TEMP TABLE paper_mapping AS
    SELECT id as openalex_paper_id,
           ROW_NUMBER() OVER (ORDER BY id) - 1 as paper_id
    FROM (
        SELECT DISTINCT w.id
        FROM openalex.works w
        INNER JOIN openalex.works_authorships wa ON w.id = wa.work_id
        WHERE wa.author_id IN (SELECT openalex_author_id FROM affiliated_authors)
    ) papers;

    CREATE INDEX ON paper_mapping(openalex_paper_id);
    CREATE INDEX ON paper_mapping(paper_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS temp_works_authorships_work_id_idx
    ON openalex.works_authorships(work_id);

    CREATE INDEX IF NOT EXISTS temp_works_authorships_author_id_idx
    ON openalex.works_authorships(author_id);
    """
]

QUERIES = {
    'author_table': """
    SELECT
        am.author_id,
        a.id as openalex_author_id,
        a.display_name as name,
        a.orcid
    FROM openalex.authors a
    INNER JOIN author_mapping am ON a.id = am.openalex_author_id
    ORDER BY am.author_id;
    """,

    'paper_table': """
    SELECT
        pm.paper_id,
        w.id as openalex_paper_id,
        w.title,
        w.publication_year as year
    FROM openalex.works w
    INNER JOIN paper_mapping pm ON w.id = pm.openalex_paper_id
    ORDER BY pm.paper_id;
    """,

    'author_paper_table': """
    WITH base_data AS (
        SELECT
            wa.work_id,
            wa.author_id,
            wa.raw_affiliation_string,
            pm.paper_id,
            am.author_id as mapped_author_id
        FROM openalex.works_authorships wa
        INNER JOIN paper_mapping pm ON wa.work_id = pm.openalex_paper_id
        INNER JOIN author_mapping am ON wa.author_id = am.openalex_author_id
    )
    SELECT
        ROW_NUMBER() OVER (ORDER BY work_id, author_id) - 1 as author_paper_id,
        paper_id,
        mapped_author_id as author_id,
        string_agg(DISTINCT raw_affiliation_string, ';') as affiliations
    FROM base_data
    GROUP BY work_id, author_id, paper_id, mapped_author_id
    ORDER BY paper_id, mapped_author_id;
    """,

    'paper_concept_table': """
    SELECT
        pm.paper_id,
        wc.concept_id,
        wc.score
    FROM openalex.works_concepts wc
    INNER JOIN paper_mapping pm ON wc.work_id = pm.openalex_paper_id
    ORDER BY pm.paper_id, wc.score DESC;
    """,
    
    'concept_table': """
    SELECT * FROM openalex.concepts
    """
}

TABLE_SCHEMAS = {
    'author_table': {
        'author_id': pl.Int64,
        'openalex_author_id': pl.Utf8,
        'name': pl.Utf8,
        'orcid': pl.Utf8
    },
    'paper_table': {
        'paper_id': pl.Int64,
        'openalex_paper_id': pl.Utf8,
        'title': pl.Utf8,
        'year': pl.Int64
    },
    'author_paper_table': {
        'author_paper_id': pl.Int64,
        'paper_id': pl.Int64,
        'author_id': pl.Int64,
        'affiliations': pl.Utf8
    },
    'paper_concept_table': {
        'paper_id': pl.Int64,
        'concept_id': pl.Utf8,
        'score': pl.Float64
    }
}

def validate_data(engine):
    """Validate data before export"""
    try:
        check_query = f"""
        SELECT COUNT(*) as count
        FROM openalex.works_authorships
        WHERE institution_id IN ('{institution_ids_str}');
        """
        df = pl.read_database(query=check_query, connection=engine)
        count = df[0, 0]  # Get first row, first column
        if count == 0:
            print("No authors found for the specified institutions!")
            return False
        print(f"Found {count:,} author-paper relationships for the institutions")
        return True
    except Exception as e:
        print(f"Validation failed: {e}")
        return False

def setup_temp_tables(engine):
    """Create temporary tables for mapping IDs"""
    try:
        with engine.connect() as conn:
            for query in SETUP_QUERIES:
                conn.execute(text(query))
                conn.commit()
        print("✓ Temporary mapping tables created")
        return True
    except Exception as e:
        print(f"Error creating temporary tables: {e}")
        raise

def export_to_csv(table_name, query, engine, output_file):
    """Execute query and export results with timing and stats"""
    print(f"\nExporting {table_name}...")
    start_time = time.time()
    try:
        # Get schema for this table
        schema = TABLE_SCHEMAS.get(table_name)

        # Execute query and load into Polars DataFrame with explicit schema
        df = pl.read_database(
            query=query,
            connection=engine,
            schema_overrides=schema
        )

        # Export to CSV
        df.write_csv(output_file)

        # Calculate statistics
        duration = time.time() - start_time
        size_mb = Path(output_file).stat().st_size / (1024 * 1024)

        # Print results
        print(f"✓ Successfully exported {df.height:,} rows")
        print(f"  File size: {size_mb:.2f}MB")
        print(f"  Duration: {duration:.2f} seconds")

        return df.height

    except Exception as e:
        print(f"Error exporting {table_name}: {e}")
        raise

def get_northeast_institutions(engine):
    """Retrieve institutions in the Northeast region using precise state boundary checks"""
    try:
        with engine.connect() as conn:
            # First, ensure PostGIS extension is available
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
            conn.commit()

            # Execute the query using Polars
            df = pl.read_database(query=query, connection=engine)
            print(f"Found {df.height:,} institutions in the Northeast region")

            # Show distribution by state
            print("\nInstitutions by state:")
            print(df.group_by('state').agg(pl.count()).sort('count', descending=True))

            return df
    except Exception as e:
        print(f"Error retrieving institutions: {e}")
        raise

# Map output files to table names
output_files = {
    'paper_table': snakemake.output.raw_paper_table_file,
    'author_table': snakemake.output.raw_author_table_file,
    'author_paper_table': snakemake.output.raw_author_paper_table_file,
    'paper_concept_table': snakemake.output.raw_paper_concept_table_file,
    'concept_table': snakemake.output.raw_concept_table_file
}

try:
    # Validate data first
    if not validate_data(engine):
        raise Exception("Validation failed")

    # Setup temporary tables
    if not setup_temp_tables(engine):
        raise Exception("Failed to create temporary tables")

    # Export each table
    results = {}
    for table_name, query in QUERIES.items():
        if table_name in output_files:
            try:
                df = export_to_csv(table_name, query, engine, output_files[table_name])
                results[table_name] = df
            except Exception as e:
                print(f"Failed to export {table_name}: {e}")
                raise

except Exception as e:
    print(f"\nError during export process: {e}")
    raise
finally:
    engine.dispose()
    print("\nDatabase connection closed")
