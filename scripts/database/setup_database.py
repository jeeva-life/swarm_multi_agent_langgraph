#!/usr/bin/env python3
"""
Database setup script for the Multi-Agent Swarm system.
Creates the Chinook sample database and required tables.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.infrastructure.database_client import DatabaseClient
from src.core.config import config


async def setup_database():
    """Set up the database with sample data."""
    try:
        print("Setting up database...")
        
        # Initialize database client
        db_client = DatabaseClient()
        
        if not await db_client.initialize():
            print("Failed to connect to database")
            return False
        
        print("Connected to database successfully")
        
        # Create Chinook sample database tables
        await create_chinook_tables(db_client)
        
        # Insert sample data
        await insert_sample_data(db_client)
        
        print("Database setup completed successfully")
        return True
        
    except Exception as e:
        print(f"Error setting up database: {str(e)}")
        return False
    finally:
        if 'db_client' in locals():
            await db_client.close()


async def create_chinook_tables(db_client: DatabaseClient):
    """Create Chinook sample database tables."""
    try:
        print("Creating tables...")
        
        # Create customers table
        customers_table = """
        CREATE TABLE IF NOT EXISTS customers (
            customer_id SERIAL PRIMARY KEY,
            first_name VARCHAR(40) NOT NULL,
            last_name VARCHAR(20) NOT NULL,
            company VARCHAR(80),
            address VARCHAR(70),
            city VARCHAR(40),
            state VARCHAR(40),
            country VARCHAR(40),
            postal_code VARCHAR(10),
            phone VARCHAR(24),
            fax VARCHAR(24),
            email VARCHAR(60) NOT NULL,
            support_rep_id INTEGER
        );
        """
        
        # Create employees table
        employees_table = """
        CREATE TABLE IF NOT EXISTS employees (
            employee_id SERIAL PRIMARY KEY,
            last_name VARCHAR(20) NOT NULL,
            first_name VARCHAR(20) NOT NULL,
            title VARCHAR(30),
            reports_to INTEGER,
            birth_date DATE,
            hire_date DATE,
            address VARCHAR(70),
            city VARCHAR(40),
            state VARCHAR(40),
            country VARCHAR(40),
            postal_code VARCHAR(10),
            phone VARCHAR(24),
            fax VARCHAR(24),
            email VARCHAR(60)
        );
        """
        
        # Create invoices table
        invoices_table = """
        CREATE TABLE IF NOT EXISTS invoices (
            invoice_id SERIAL PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            invoice_date TIMESTAMP NOT NULL,
            billing_address VARCHAR(70),
            billing_city VARCHAR(40),
            billing_state VARCHAR(40),
            billing_country VARCHAR(40),
            billing_postal_code VARCHAR(10),
            total DECIMAL(10,2) NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        );
        """
        
        # Create invoice_lines table
        invoice_lines_table = """
        CREATE TABLE IF NOT EXISTS invoice_lines (
            invoice_line_id SERIAL PRIMARY KEY,
            invoice_id INTEGER NOT NULL,
            track_id INTEGER NOT NULL,
            unit_price DECIMAL(10,2) NOT NULL,
            quantity INTEGER NOT NULL,
            FOREIGN KEY (invoice_id) REFERENCES invoices(invoice_id)
        );
        """
        
        # Create tracks table
        tracks_table = """
        CREATE TABLE IF NOT EXISTS tracks (
            track_id SERIAL PRIMARY KEY,
            name VARCHAR(200) NOT NULL,
            album_id INTEGER,
            media_type_id INTEGER NOT NULL,
            genre_id INTEGER,
            composer VARCHAR(220),
            milliseconds INTEGER NOT NULL,
            bytes INTEGER,
            unit_price DECIMAL(10,2) NOT NULL
        );
        """
        
        # Create albums table
        albums_table = """
        CREATE TABLE IF NOT EXISTS albums (
            album_id SERIAL PRIMARY KEY,
            title VARCHAR(160) NOT NULL,
            artist_id INTEGER NOT NULL
        );
        """
        
        # Create artists table
        artists_table = """
        CREATE TABLE IF NOT EXISTS artists (
            artist_id SERIAL PRIMARY KEY,
            name VARCHAR(120)
        );
        """
        
        # Create genres table
        genres_table = """
        CREATE TABLE IF NOT EXISTS genres (
            genre_id SERIAL PRIMARY KEY,
            name VARCHAR(120)
        );
        """
        
        # Create media_types table
        media_types_table = """
        CREATE TABLE IF NOT EXISTS media_types (
            media_type_id SERIAL PRIMARY KEY,
            name VARCHAR(120)
        );
        """
        
        # Execute table creation
        tables = [
            ("customers", customers_table),
            ("employees", employees_table),
            ("artists", artists_table),
            ("albums", albums_table),
            ("genres", genres_table),
            ("media_types", media_types_table),
            ("tracks", tracks_table),
            ("invoices", invoices_table),
            ("invoice_lines", invoice_lines_table)
        ]
        
        for table_name, table_sql in tables:
            await db_client.execute_command(table_sql)
            print(f"Created table: {table_name}")
        
        print("All tables created successfully")
        
    except Exception as e:
        print(f"Error creating tables: {str(e)}")
        raise


async def insert_sample_data(db_client: DatabaseClient):
    """Insert sample data into the database."""
    try:
        print("Inserting sample data...")
        
        # Insert sample customers
        customers_data = [
            (1, "Luís", "Gonçalves", "Embraer - Empresa Brasileira de Aeronáutica S.A.", "Av. Brigadeiro Faria Lima, 2170", "São José dos Campos", "SP", "Brazil", "12227-000", "+55 (12) 3923-5555", "+55 (12) 3923-5566", "luisg@embraer.com.br", 3),
            (2, "Leonie", "Köhler", None, "Theodor-Heuss-Straße 287", "Stuttgart", None, "Germany", "70174", "+49 0711 2842222", None, "leonekohler@surfeu.de", 5),
            (3, "François", "Tremblay", None, "1498 rue Bélanger", "Montréal", "QC", "Canada", "H2G 1A7", "+1 (514) 721-4711", None, "ftremblay@gmail.com", 3),
            (4, "Bjørn", "Hansen", None, "Ullevålsveien 14", "Oslo", None, "Norway", "0171", "+47 22 44 22 22", None, "bjorn.hansen@yahoo.no", 4),
            (5, "František", "Wichterlová", "JetBrains s.r.o.", "Klanova 9/506", "Prague", None, "Czech Republic", "14700", "+420 2 4172 5555", None, "frantisekw@jetbrains.com", 4)
        ]
        
        for customer in customers_data:
            await db_client.execute_command(
                """INSERT INTO customers (customer_id, first_name, last_name, company, address, city, state, country, postal_code, phone, fax, email, support_rep_id) 
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)""",
                customer
            )
        
        print(f"Inserted {len(customers_data)} customers")
        
        # Insert sample employees
        employees_data = [
            (1, "Adams", "Andrew", "General Manager", None, "1962-02-18", "2002-08-14", "11120 Jasper Ave NW", "Edmonton", "AB", "Canada", "T5K 2N1", "+1 (780) 428-9482", "+1 (780) 428-3457", "andrew@chinookcorp.com"),
            (2, "Edwards", "Nancy", "Sales Manager", 1, "1958-12-08", "2002-05-01", "825 8 Ave SW", "Calgary", "AB", "Canada", "T2P 2T3", "+1 (403) 262-3443", "+1 (403) 262-3322", "nancy@chinookcorp.com"),
            (3, "Peacock", "Jane", "Sales Support Agent", 2, "1973-08-29", "2002-04-01", "1111 6 Ave SW", "Calgary", "AB", "Canada", "T2P 5M5", "+1 (403) 262-3443", "+1 (403) 262-6712", "jane@chinookcorp.com")
        ]
        
        for employee in employees_data:
            await db_client.execute_command(
                """INSERT INTO employees (employee_id, last_name, first_name, title, reports_to, birth_date, hire_date, address, city, state, country, postal_code, phone, fax, email) 
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)""",
                employee
            )
        
        print(f"Inserted {len(employees_data)} employees")
        
        # Insert sample artists
        artists_data = [
            (1, "AC/DC"),
            (2, "Accept"),
            (3, "Aerosmith"),
            (4, "Alanis Morissette"),
            (5, "Alice In Chains")
        ]
        
        for artist in artists_data:
            await db_client.execute_command(
                "INSERT INTO artists (artist_id, name) VALUES ($1, $2)",
                artist
            )
        
        print(f"Inserted {len(artists_data)} artists")
        
        # Insert sample genres
        genres_data = [
            (1, "Rock"),
            (2, "Jazz"),
            (3, "Metal"),
            (4, "Alternative & Punk"),
            (5, "Rock And Roll")
        ]
        
        for genre in genres_data:
            await db_client.execute_command(
                "INSERT INTO genres (genre_id, name) VALUES ($1, $2)",
                genre
            )
        
        print(f"Inserted {len(genres_data)} genres")
        
        # Insert sample media types
        media_types_data = [
            (1, "MPEG audio file"),
            (2, "Protected AAC audio file"),
            (3, "Protected MPEG-4 video file"),
            (4, "Purchased AAC audio file"),
            (5, "AAC audio file")
        ]
        
        for media_type in media_types_data:
            await db_client.execute_command(
                "INSERT INTO media_types (media_type_id, name) VALUES ($1, $2)",
                media_type
            )
        
        print(f"Inserted {len(media_types_data)} media types")
        
        # Insert sample albums
        albums_data = [
            (1, "For Those About To Rock We Salute You", 1),
            (2, "Balls to the Wall", 2),
            (3, "Restless and Wild", 2),
            (4, "Let There Be Rock", 1),
            (5, "Big Ones", 3)
        ]
        
        for album in albums_data:
            await db_client.execute_command(
                "INSERT INTO albums (album_id, title, artist_id) VALUES ($1, $2, $3)",
                album
            )
        
        print(f"Inserted {len(albums_data)} albums")
        
        # Insert sample tracks
        tracks_data = [
            (1, "For Those About To Rock (We Salute You)", 1, 1, 1, "Angus Young, Malcolm Young, Brian Johnson", 343719, 11170334, 0.99),
            (2, "Balls to the Wall", 2, 2, 1, None, 342562, 5510424, 0.99),
            (3, "Fast As a Shark", 3, 2, 1, "F. Baltes, S. Kaufman, U. Dirkscneider & W. Hoffman", 230619, 3990994, 0.99),
            (4, "Restless and Wild", 3, 2, 1, "F. Baltes, R.A. Smith-Diesel, S. Kaufman, U. Dirkscneider & W. Hoffman", 252051, 4331779, 0.99),
            (5, "Princess of the Dawn", 3, 2, 1, "Deaffy & R.A. Smith-Diesel", 375418, 6290521, 0.99)
        ]
        
        for track in tracks_data:
            await db_client.execute_command(
                """INSERT INTO tracks (track_id, name, album_id, media_type_id, genre_id, composer, milliseconds, bytes, unit_price) 
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)""",
                track
            )
        
        print(f"Inserted {len(tracks_data)} tracks")
        
        # Insert sample invoices
        invoices_data = [
            (1, 1, "2009-01-01 00:00:00", "Theodor-Heuss-Straße 287", "Stuttgart", None, "Germany", "70174", 1.98),
            (2, 2, "2009-01-02 00:00:00", "1498 rue Bélanger", "Montréal", "QC", "Canada", "H2G 1A7", 3.96),
            (3, 3, "2009-01-03 00:00:00", "Ullevålsveien 14", "Oslo", None, "Norway", "0171", 5.94),
            (4, 4, "2009-01-06 00:00:00", "Klanova 9/506", "Prague", None, "Czech Republic", "14700", 8.91),
            (5, 5, "2009-01-11 00:00:00", "Av. Brigadeiro Faria Lima, 2170", "São José dos Campos", "SP", "Brazil", "12227-000", 13.86)
        ]
        
        for invoice in invoices_data:
            await db_client.execute_command(
                """INSERT INTO invoices (invoice_id, customer_id, invoice_date, billing_address, billing_city, billing_state, billing_country, billing_postal_code, total) 
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)""",
                invoice
            )
        
        print(f"Inserted {len(invoices_data)} invoices")
        
        # Insert sample invoice lines
        invoice_lines_data = [
            (1, 1, 1, 0.99, 1),
            (2, 1, 2, 0.99, 1),
            (3, 2, 3, 0.99, 1),
            (4, 2, 4, 0.99, 1),
            (5, 2, 5, 0.99, 1),
            (6, 2, 6, 0.99, 1),
            (7, 3, 7, 0.99, 1),
            (8, 3, 8, 0.99, 1),
            (9, 3, 9, 0.99, 1),
            (10, 3, 10, 0.99, 1),
            (11, 3, 11, 0.99, 1),
            (12, 3, 12, 0.99, 1)
        ]
        
        for invoice_line in invoice_lines_data:
            await db_client.execute_command(
                """INSERT INTO invoice_lines (invoice_line_id, invoice_id, track_id, unit_price, quantity) 
                   VALUES ($1, $2, $3, $4, $5)""",
                invoice_line
            )
        
        print(f"Inserted {len(invoice_lines_data)} invoice lines")
        
        print("Sample data insertion completed successfully")
        
    except Exception as e:
        print(f"Error inserting sample data: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(setup_database())
