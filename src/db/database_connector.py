"""
Database connector for pharmaceutical package and sheet recognition.
"""

import json
import os
import sqlite3
from loguru import logger
import threading
from contextlib import contextmanager

class DatabaseConnector:
    """
    Database connector for storing and retrieving pharmaceutical product information.
    """
    
    def __init__(self, config_path='config/system_config.json', db_path=None):
        """
        Initialize database connector.
        
        Args:
            config_path (str): Path to configuration file
            db_path (str, optional): Path to database file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Determine database path
        if db_path is None:
            db_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
            os.makedirs(db_dir, exist_ok=True)
            self.db_path = os.path.join(db_dir, 'products.db')
        else:
            self.db_path = db_path
        
        # Initialize database
        self.init_database()
        
        # Thread lock for database access
        self.lock = threading.Lock()
        
        logger.info(f"Database connector initialized with database: {self.db_path}")
    
    @contextmanager
    def get_connection(self):
        """
        Get a database connection with thread safety.
        
        Yields:
            sqlite3.Connection: Database connection
        """
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.row_factory = sqlite3.Row
                yield conn
            finally:
                conn.close()
    
    def init_database(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create products table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                commercial_name TEXT NOT NULL,
                scientific_name TEXT,
                manufacturer TEXT,
                dosage TEXT,
                template_id TEXT,
                is_package INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Create recognition_results table for tracking OCR results
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS recognition_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER,
                commercial_name TEXT,
                commercial_name_confidence REAL,
                scientific_name TEXT,
                scientific_name_confidence REAL,
                manufacturer TEXT,
                manufacturer_confidence REAL,
                dosage TEXT,
                dosage_confidence REAL,
                is_package INTEGER,
                template_id TEXT,
                image_path TEXT,
                processing_time_ms REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (product_id) REFERENCES products (id)
            )
            ''')
            
            # Create index on commercial_name for faster lookups
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_commercial_name ON products (commercial_name)')
            
            conn.commit()
            
            logger.info("Database schema initialized")
    
    def add_product(self, commercial_name, scientific_name=None, manufacturer=None, dosage=None, template_id=None, is_package=True):
        """
        Add a new product to the database.
        
        Args:
            commercial_name (str): Commercial name
            scientific_name (str, optional): Scientific name
            manufacturer (str, optional): Manufacturer
            dosage (str, optional): Dosage information
            template_id (str, optional): Template ID
            is_package (bool): True if it's a package, False if it's a sheet
            
        Returns:
            int: Product ID
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if product already exists
            cursor.execute(
                'SELECT id FROM products WHERE commercial_name=?',
                (commercial_name,)
            )
            existing = cursor.fetchone()
            
            if existing:
                # Update existing product
                cursor.execute('''
                UPDATE products
                SET scientific_name=?, manufacturer=?, dosage=?, template_id=?, is_package=?
                WHERE id=?
                ''', (
                    scientific_name, manufacturer, dosage, template_id, 
                    1 if is_package else 0, existing['id']
                ))
                product_id = existing['id']
                logger.debug(f"Updated existing product: {commercial_name} (ID: {product_id})")
            else:
                # Insert new product
                cursor.execute('''
                INSERT INTO products 
                (commercial_name, scientific_name, manufacturer, dosage, template_id, is_package)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    commercial_name, scientific_name, manufacturer, dosage,
                    template_id, 1 if is_package else 0
                ))
                product_id = cursor.lastrowid
                logger.debug(f"Added new product: {commercial_name} (ID: {product_id})")
            
            conn.commit()
            
            return product_id
    
    def get_product(self, product_id):
        """
        Get product by ID.
        
        Args:
            product_id (int): Product ID
            
        Returns:
            dict: Product data
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM products WHERE id=?', (product_id,))
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            
            return None
    
    def get_product_by_name(self, commercial_name):
        """
        Get product by commercial name.
        
        Args:
            commercial_name (str): Commercial name
            
        Returns:
            dict: Product data
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM products WHERE commercial_name=?', (commercial_name,))
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            
            return None
    
    def get_product_by_template(self, template_id):
        """
        Get product by template ID.
        
        Args:
            template_id (str): Template ID
            
        Returns:
            dict: Product data
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM products WHERE template_id=?', (template_id,))
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            
            return None
    
    def update_product(self, product_id, **fields):
        """
        Update product fields.
        
        Args:
            product_id (int): Product ID
            **fields: Fields to update
            
        Returns:
            bool: True if successful, False otherwise
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Build SET clause for SQL
            set_clause = ', '.join(f"{key}=?" for key in fields.keys())
            values = list(fields.values())
            values.append(product_id)
            
            cursor.execute(f'UPDATE products SET {set_clause} WHERE id=?', values)
            conn.commit()
            
            return cursor.rowcount > 0
    
    def delete_product(self, product_id):
        """
        Delete product by ID.
        
        Args:
            product_id (int): Product ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM products WHERE id=?', (product_id,))
            conn.commit()
            
            return cursor.rowcount > 0
    
    def add_recognition_result(self, result_data):
        """
        Add a recognition result to the database.
        
        Args:
            result_data (dict): Recognition result data
            
        Returns:
            int: Result ID
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Extract fields from result data
            product_id = result_data.get('product_id')
            commercial_name = result_data.get('commercial_name', {}).get('value')
            commercial_name_confidence = result_data.get('commercial_name', {}).get('confidence', 0.0)
            scientific_name = result_data.get('scientific_name', {}).get('value')
            scientific_name_confidence = result_data.get('scientific_name', {}).get('confidence', 0.0)
            manufacturer = result_data.get('manufacturer', {}).get('value')
            manufacturer_confidence = result_data.get('manufacturer', {}).get('confidence', 0.0)
            dosage = result_data.get('dosage', {}).get('value')
            dosage_confidence = result_data.get('dosage', {}).get('confidence', 0.0)
            is_package = result_data.get('is_package', True)
            template_id = result_data.get('template_id')
            image_path = result_data.get('image_path')
            processing_time_ms = result_data.get('processing_time_ms', 0.0)
            
            # Insert into database
            cursor.execute('''
            INSERT INTO recognition_results
            (product_id, commercial_name, commercial_name_confidence,
            scientific_name, scientific_name_confidence,
            manufacturer, manufacturer_confidence,
            dosage, dosage_confidence,
            is_package, template_id, image_path, processing_time_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                product_id, commercial_name, commercial_name_confidence,
                scientific_name, scientific_name_confidence,
                manufacturer, manufacturer_confidence,
                dosage, dosage_confidence,
                1 if is_package else 0, template_id, image_path, processing_time_ms
            ))
            
            result_id = cursor.lastrowid
            conn.commit()
            
            logger.debug(f"Added recognition result (ID: {result_id}) for product: {commercial_name}")
            
            return result_id
    
    def get_recognition_results(self, product_id=None, limit=10):
        """
        Get recognition results.
        
        Args:
            product_id (int, optional): Filter by product ID
            limit (int): Maximum number of results to return
            
        Returns:
            list: Recognition results
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if product_id is not None:
                cursor.execute(
                    'SELECT * FROM recognition_results WHERE product_id=? ORDER BY created_at DESC LIMIT ?',
                    (product_id, limit)
                )
            else:
                cursor.execute(
                    'SELECT * FROM recognition_results ORDER BY created_at DESC LIMIT ?',
                    (limit,)
                )
            
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows]
    
    def verify_template_ids(self):
        """
        Verify that all products have valid template IDs.
        
        Returns:
            dict: Verification results
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT id, commercial_name, template_id FROM products')
            rows = cursor.fetchall()
            
            results = {
                'total': len(rows),
                'with_template': 0,
                'without_template': 0,
                'products_without_template': []
            }
            
            for row in rows:
                if row['template_id']:
                    results['with_template'] += 1
                else:
                    results['without_template'] += 1
                    results['products_without_template'].append({
                        'id': row['id'],
                        'commercial_name': row['commercial_name']
                    })
            
            return results
    
    def get_performance_stats(self):
        """
        Get performance statistics for recognition.
        
        Returns:
            dict: Performance statistics
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get average processing time
            cursor.execute('SELECT AVG(processing_time_ms) as avg_time FROM recognition_results')
            avg_time = cursor.fetchone()['avg_time'] or 0
            
            # Get processing time percentiles
            cursor.execute('''
            SELECT processing_time_ms
            FROM recognition_results
            ORDER BY processing_time_ms
            ''')
            
            times = [row['processing_time_ms'] for row in cursor.fetchall()]
            
            # Calculate percentiles
            p50 = p90 = p95 = p99 = 0
            if times:
                times.sort()
                n = len(times)
                p50 = times[int(n * 0.5)] if n > 0 else 0
                p90 = times[int(n * 0.9)] if n > 0 else 0
                p95 = times[int(n * 0.95)] if n > 0 else 0
                p99 = times[int(n * 0.99)] if n > 0 else 0
            
            # Get average confidence scores
            cursor.execute('''
            SELECT 
                AVG(commercial_name_confidence) as avg_commercial_name_confidence,
                AVG(scientific_name_confidence) as avg_scientific_name_confidence,
                AVG(manufacturer_confidence) as avg_manufacturer_confidence,
                AVG(dosage_confidence) as avg_dosage_confidence
            FROM recognition_results
            ''')
            
            confidence_row = cursor.fetchone()
            
            # Count products
            cursor.execute('SELECT COUNT(*) as count FROM products')
            product_count = cursor.fetchone()['count']
            
            # Count recognition results
            cursor.execute('SELECT COUNT(*) as count FROM recognition_results')
            result_count = cursor.fetchone()['count']
            
            return {
                'avg_processing_time_ms': avg_time,
                'p50_processing_time_ms': p50,
                'p90_processing_time_ms': p90,
                'p95_processing_time_ms': p95,
                'p99_processing_time_ms': p99,
                'avg_commercial_name_confidence': confidence_row['avg_commercial_name_confidence'] or 0,
                'avg_scientific_name_confidence': confidence_row['avg_scientific_name_confidence'] or 0,
                'avg_manufacturer_confidence': confidence_row['avg_manufacturer_confidence'] or 0,
                'avg_dosage_confidence': confidence_row['avg_dosage_confidence'] or 0,
                'product_count': product_count,
                'result_count': result_count
            }
    
    def search_products(self, query, limit=10):
        """
        Search for products by name or description.
        
        Args:
            query (str): Search query
            limit (int): Maximum number of results to return
            
        Returns:
            list: Search results
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Build search query
            search_param = f"%{query}%"
            
            cursor.execute('''
            SELECT * FROM products
            WHERE commercial_name LIKE ?
               OR scientific_name LIKE ?
               OR manufacturer LIKE ?
               OR dosage LIKE ?
            ORDER BY commercial_name
            LIMIT ?
            ''', (search_param, search_param, search_param, search_param, limit))
            
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows]
