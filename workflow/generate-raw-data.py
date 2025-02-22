import pandas as pd
from pathlib import Path
import time
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

# Database connection parameters
db_params = {
    'dbname': 'openalex', # set the database name here
    'user': 'skojaku', # set the username for the db
    'password': 'xxx', # set the password for the db
    'host': 'localhost',
    'port': '5432' # change the port appropriately
}

engine = create_engine(
    f"postgresql://{db_params['user']}:{quote_plus(db_params['password'])}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}"
)


SETUP_QUERIES = [
    """
    -- First create temporary table of Binghamton-affiliated authors
    CREATE TEMP TABLE binghamton_affiliated_authors AS
    SELECT DISTINCT openalex_author_id
    FROM (
        SELECT author_id as openalex_author_id
        FROM openalex.works_authorships
        WHERE institution_id = 'https://openalex.org/I123946342'
    ) AS bu_authors;
    
    CREATE INDEX ON binghamton_affiliated_authors(openalex_author_id);
    """,
    """
    -- Create temporary table for author mapping
    CREATE TEMP TABLE author_mapping AS
    SELECT 
        a.id as openalex_author_id,
        ROW_NUMBER() OVER (ORDER BY a.id) - 1 as author_id
    FROM openalex.authors a
    WHERE a.id IN (SELECT openalex_author_id FROM binghamton_affiliated_authors);
    
    CREATE INDEX ON author_mapping(openalex_author_id);
    CREATE INDEX ON author_mapping(author_id);
    """,
    """
    -- Create temporary table for paper mapping
    -- This includes ALL papers by Binghamton-affiliated authors
    CREATE TEMP TABLE paper_mapping AS
    SELECT id as openalex_paper_id,
           ROW_NUMBER() OVER (ORDER BY id) - 1 as paper_id
    FROM (
        SELECT DISTINCT w.id
        FROM openalex.works w
        INNER JOIN openalex.works_authorships wa ON w.id = wa.work_id
        WHERE wa.author_id IN (SELECT openalex_author_id FROM binghamton_affiliated_authors)
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
        check_query = """
        SELECT COUNT(*) as count
        FROM openalex.works_authorships
        WHERE institution_id = 'https://openalex.org/I123946342';
        """
        with engine.connect() as conn:
            result = pd.read_sql_query(check_query, conn)
            count = result['count'].iloc[0]
            if count == 0:
                print("No Binghamton University authors found!")
                return False
            print(f"Found {count:,} author-paper relationships for Binghamton University")
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

def export_to_csv(table_name, query, engine, output_dir):
    """Execute query and export results to CSV with timing and stats"""
    print(f"\nExporting {table_name}...")
    start_time = time.time()
    try:
        # Execute query and load into DataFrame
        df = pd.read_sql_query(query, engine)
        
        # Export to CSV
        output_file = output_dir / f"{table_name}.csv"
        df.to_csv(output_file, index=False)
        
        # Calculate statistics
        duration = time.time() - start_time
        size_mb = output_file.stat().st_size / (1024 * 1024)
        
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

def main():
    """Main execution function"""
    # Create output directory
    output_dir = Path('openalex_export')
    output_dir.mkdir(exist_ok=True)
    
    print(f"Starting export process...")
    print(f"Output directory: {output_dir.absolute()}")
    
    try:
        # Validate data first
        if not validate_data(engine):
            print("Validation failed. Exiting...")
            return
            
        # Setup temporary tables
        if not setup_temp_tables(engine):
            print("Failed to create temporary tables. Exiting...")
            return
        
        # Export each table
        results = {}
        for table_name, query in QUERIES.items():
            try:
                df = export_to_csv(table_name, query, engine, output_dir)
                results[table_name] = df
            except Exception as e:
                print(f"Failed to export {table_name}: {e}")
                raise
        
        # Print summary
        print("\nExport Summary:")
        print("-" * 50)
        total_rows = sum(len(df) for df in results.values())
        total_size = sum(
            (output_dir / f"{name}.csv").stat().st_size / (1024 * 1024)
            for name in QUERIES.keys()
        )
        print(f"Total rows exported: {total_rows:,}")
        print(f"Total file size: {total_size:.2f}MB")
        print(f"Files created:")
        for name in QUERIES.keys():
            file_size = (output_dir / f"{name}.csv").stat().st_size / (1024 * 1024)
            print(f"  - {name}.csv: {file_size:.2f}MB")
        
        print("\nExport completed successfully!")
    
    except Exception as e:
        print(f"\nError during export process: {e}")
        raise
    finally:
        engine.dispose()
        print("\nDatabase connection closed")

if __name__ == "__main__":
    main()
