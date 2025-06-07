"""
BigQuery client utilities for Ember ML.

This module provides functions for connecting to and interacting with
Google BigQuery from Ember ML.
"""

import logging
from typing import Any, Dict, Optional

# Set up logging
logger = logging.getLogger(__name__)


def initialize_client(project_id: str, credentials_path: Optional[str] = None) -> Any:
    """
    Initialize a BigQuery client.
    
    Args:
        project_id: Google Cloud project ID
        credentials_path: Path to Google Cloud credentials file
        
    Returns:
        BigQuery client object
    """
    try:
        from google.cloud import bigquery
        from google.oauth2 import service_account
        
        # Initialize credentials if provided
        if credentials_path:
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path
            )
            client = bigquery.Client(
                project=project_id,
                credentials=credentials
            )
        else:
            # Use default credentials
            client = bigquery.Client(project=project_id)
            
        logger.info(f"Initialized BigQuery client for project: {project_id}")
        return client
    
    except ImportError:
        logger.error("Failed to import BigQuery libraries. Install with: "
                     "pip install google-cloud-bigquery")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize BigQuery client: {e}")
        raise


def execute_query(client: Any, query: str) -> Any:
    """
    Execute a SQL query on BigQuery and return results as a DataFrame.
    
    Args:
        client: BigQuery client
        query: SQL query string
        
    Returns:
        DataFrame containing query results
    """
    try:
        import pandas as pd
        
        # Execute the query
        query_job = client.query(query)
        
        # Wait for the query to complete
        results = query_job.result()
        
        # Convert to DataFrame
        df = results.to_dataframe()
        
        logger.info(f"Executed query with {len(df)} results")
        return df
    
    except Exception as e:
        logger.error(f"Failed to execute query: {e}")
        raise


def fetch_table_schema(client: Any, dataset_id: str, table_id: str) -> Dict[str, str]:
    """
    Fetch the schema of a BigQuery table.
    
    Args:
        client: BigQuery client
        dataset_id: BigQuery dataset ID
        table_id: BigQuery table ID
        
    Returns:
        Dictionary mapping column names to their data types
    """
    try:
        # Get table reference
        table_ref = client.dataset(dataset_id).table(table_id)
        
        # Get table
        table = client.get_table(table_ref)
        
        # Extract schema
        schema = {}
        for field in table.schema:
            schema[field.name] = field.field_type
            
        logger.info(f"Fetched schema for {dataset_id}.{table_id} with {len(schema)} columns")
        return schema
    
    except Exception as e:
        logger.error(f"Failed to fetch table schema: {e}")
        raise