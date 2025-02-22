import pandas as pd
from pathlib import Path
import time
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

# Get institution ID from snakemake config
INSTITUTION_ID = snakemake.config["institution_id"]

# Database connection parameters
db_params = {
    'dbname': 'openalex',
    'user': 'skojaku',
    'password': 'xxx',
    'host': 'localhost',
    'port': '5432'
}

engine = create_engine(
    f"postgresql://{db_params['user']}:{quote_plus(db_params['password'])}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}"
)

SETUP_QUERIES = [
    f"""
    -- First create temporary table of institution-affiliated authors
    CREATE TEMP TABLE affiliated_authors AS
    SELECT DISTINCT openalex_author_id
    FROM (
        SELECT author_id as openalex_author_id
        FROM openalex.works_authorships
        WHERE institution_id = '{INSTITUTION_ID}'
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
    SELECT
        ROW_NUMBER() OVER (ORDER BY wa.work_id, wa.author_id) - 1 as author_paper_id,
        pm.paper_id,
        am.author_id,
        string_agg(DISTINCT wa.raw_affiliation_string, ';') as affiliations
    FROM openalex.works_authorships wa
    INNER JOIN paper_mapping pm ON wa.work_id = pm.openalex_paper_id
    INNER JOIN author_mapping am ON wa.author_id = am.openalex_author_id
    GROUP BY wa.work_id, wa.author_id, pm.paper_id, am.author_id
    ORDER BY pm.paper_id, am.author_id;
    """,

    'paper_concept_table': """
    SELECT
        pm.paper_id,
        wc.concept_id,
        wc.score
    FROM openalex.works_concepts wc
    INNER JOIN paper_mapping pm ON wc.work_id = pm.openalex_paper_id
    ORDER BY pm.paper_id, wc.score DESC;
    """
}

def validate_data(engine):
    """Validate data before export"""
    try:
        check_query = f"""
        SELECT COUNT(*) as count
        FROM openalex.works_authorships
        WHERE institution_id = '{INSTITUTION_ID}';
        """
        with engine.connect() as conn:
            result = pd.read_sql_query(check_query, conn)
            count = result['count'].iloc[0]
            if count == 0:
                print("No authors found for the specified institution!")
                return False
            print(f"Found {count:,} author-paper relationships for the institution")
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
    """Execute query and export results to CSV with timing and stats"""
    print(f"\nExporting {table_name}...")
    start_time = time.time()
    try:
        # Execute query and load into DataFrame
        df = pd.read_sql_query(query, engine)

        # Export to CSV
        df.to_csv(output_file, index=False)

        # Calculate statistics
        duration = time.time() - start_time
        size_mb = Path(output_file).stat().st_size / (1024 * 1024)

        # Print results
        print(f"✓ Successfully exported {len(df):,} rows")
        print(f"  File size: {size_mb:.2f}MB")
        print(f"  Duration: {duration:.2f} seconds")

        # Check for null values
        null_counts = df.isna().sum()
        if null_counts.any():
            print("\n  Null value counts:")
            for col, count in null_counts[null_counts > 0].items():
                print(f"    - {col}: {count:,} nulls")

        return df
    except Exception as e:
        print(f"Error exporting {table_name}: {e}")
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

    # Print summary
    print("\nExport Summary:")
    print("-" * 50)
    total_rows = sum(len(df) for df in results.values())
    total_size = sum(
        Path(file).stat().st_size / (1024 * 1024)
        for file in output_files.values()
    )
    print(f"Total rows exported: {total_rows:,}")
    print(f"Total file size: {total_size:.2f}MB")
    print("\nExport completed successfully!")

except Exception as e:
    print(f"\nError during export process: {e}")
    raise
finally:
    engine.dispose()
    print("\nDatabase connection closed")
