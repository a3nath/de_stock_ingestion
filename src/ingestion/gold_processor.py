import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

import pandas as pd
import numpy as np
from google.cloud import storage
from google.cloud.storage import Client

logger = logging.getLogger(__name__)

class GoldProcessor:
    """
    Handles the processing of silver layer data into the gold layer.
    Implements business transformations and analytics optimizations.
    """
    
    def __init__(
        self,
        bucket_name: str,
        silver_prefix: str = "silver",
        gold_prefix: str = "gold",
        storage_client: Optional[Client] = None
    ):
        """
        Initialize the GoldProcessor.

        Args:
            bucket_name (str): Name of the GCS bucket
            silver_prefix (str): Prefix for silver data in the bucket
            gold_prefix (str): Prefix for gold data in the bucket
            storage_client (Optional[Client]): GCS client instance
        """
        self.bucket_name = bucket_name
        self.silver_prefix = silver_prefix
        self.gold_prefix = gold_prefix
        self.storage_client = storage_client or storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)
        
        # Initialize metadata
        self.metadata = {
            "processing_date": None,
            "source_files": [],
            "record_count": 0,
            "status": "initialized",
            "errors": []
        }

    def process_daily_data(self, date: str) -> bool:
        """
        Process silver layer data for a specific date into the gold layer.

        Args:
            date (str): Date to process in YYYY-MM-DD format

        Returns:
            bool: True if processing was successful, False otherwise
        """
        try:
            logger.info(f"Starting gold layer processing for date: {date}")
            self.metadata["processing_date"] = date
            self.metadata["status"] = "processing"

            # 1. Read silver data
            silver_data = self._read_silver_data(date)
            if silver_data.empty:
                logger.warning(f"No silver data found for date: {date}")
                self.metadata["status"] = "failed"
                self.metadata["errors"].append("No silver data found")
                return False

            # 2. Apply business transformations
            transformed_data = self._transform_data(silver_data)
            
            # 3. Store in gold layer
            storage_success = self._store_gold_data(transformed_data, date)
            if not storage_success:
                logger.error("Failed to store data in gold layer")
                self.metadata["status"] = "failed"
                return False

            # 4. Update metadata
            self._update_metadata(transformed_data)
            
            self.metadata["status"] = "success"
            logger.info(f"Successfully processed data for date: {date}")
            return True

        except Exception as e:
            self.metadata["status"] = "failed"
            self.metadata["errors"].append(str(e))
            logger.error(f"Error processing data for date {date}: {str(e)}")
            return False

    def _read_silver_data(self, date: str) -> pd.DataFrame:
        """
        Read silver layer data from GCS for a specific date.

        Args:
            date (str): Date to read in YYYY-MM-DD format

        Returns:
            pd.DataFrame: Silver layer data

        Raises:
            ValueError: If date format is invalid
            Exception: If there are issues reading from GCS
        """
        try:
            # Validate date format
            datetime.strptime(date, "%Y-%m-%d")
            
            # List all blobs in the silver prefix for the given date
            prefix = f"{self.silver_prefix}/{date}/"
            blobs = list(self.bucket.list_blobs(prefix=prefix))
            
            if not blobs:
                logger.warning(f"No files found in {prefix}")
                return pd.DataFrame()
            
            # Initialize list to store data from all files
            all_data = []
            
            # Read each file
            for blob in blobs:
                if not blob.name.endswith('.json'):
                    continue
                    
                logger.info(f"Reading file: {blob.name}")
                self.metadata["source_files"].append(blob.name)
                
                # Download and parse JSON
                content = blob.download_as_string()
                data = pd.read_json(content)
                all_data.append(data)
            
            if not all_data:
                logger.warning(f"No valid JSON files found in {prefix}")
                return pd.DataFrame()
            
            # Combine all data
            combined_data = pd.concat(all_data, ignore_index=True)
            self.metadata["record_count"] = len(combined_data)
            
            logger.info(f"Successfully read {len(combined_data)} records from silver layer")
            return combined_data
            
        except ValueError as e:
            logger.error(f"Invalid date format: {date}")
            raise ValueError(f"Date must be in YYYY-MM-DD format: {str(e)}")
        except Exception as e:
            logger.error(f"Error reading silver data: {str(e)}")
            raise

    def _transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply business transformations to the data.

        Args:
            data (pd.DataFrame): Silver layer data to transform

        Returns:
            pd.DataFrame: Transformed data ready for gold layer
        """
        try:
            logger.info("Starting business transformations")
            
            # Create a copy to avoid modifying original data
            transformed_data = data.copy()
            
            # 1. Calculate technical indicators
            # Moving averages
            transformed_data['sma_20'] = transformed_data['close_price'].rolling(window=20).mean()
            transformed_data['sma_50'] = transformed_data['close_price'].rolling(window=50).mean()
            
            # Volatility
            transformed_data['volatility'] = transformed_data['daily_return'].rolling(window=20).std()
            
            # 2. Add business metrics
            # Price momentum
            transformed_data['price_momentum'] = transformed_data['close_price'].pct_change(periods=5)
            
            # Volume trend
            transformed_data['volume_trend'] = transformed_data['volume'].pct_change(periods=5)
            
            # 3. Add derived features
            # Price range percentage
            transformed_data['price_range_pct'] = (
                (transformed_data['high_price'] - transformed_data['low_price']) / 
                transformed_data['close_price'] * 100
            )
            
            # 4. Add metadata
            transformed_data['data_layer'] = 'gold'
            transformed_data['last_updated'] = pd.Timestamp.now()
            
            # 5. Sort and clean up
            transformed_data = transformed_data.sort_values('date').reset_index(drop=True)
            
            logger.info(f"Business transformations completed. Shape: {transformed_data.shape}")
            return transformed_data
            
        except Exception as e:
            error_msg = f"Error during business transformations: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _store_gold_data(self, data: pd.DataFrame, date: str) -> bool:
        """
        Store processed data in the gold layer.

        Args:
            data (pd.DataFrame): Processed data to store
            date (str): Date of the data

        Returns:
            bool: True if storage was successful

        Raises:
            ValueError: If date format is invalid
            Exception: If there are issues storing data
        """
        try:
            logger.info(f"Storing gold layer data for date: {date}")
            
            # Validate date format
            datetime.strptime(date, "%Y-%m-%d")
            
            # Create directory structure
            year, month, day = date.split('-')
            blob_path = f"{self.gold_prefix}/{year}/{month}/{day}/data.parquet"
            
            # Convert DataFrame to Parquet
            parquet_data = data.to_parquet(index=False)
            
            # Create blob and upload
            blob = self.bucket.blob(blob_path)
            blob.upload_from_string(
                parquet_data,
                content_type='application/octet-stream'
            )
            
            # Set metadata
            metadata = {
                'date': date,
                'record_count': str(len(data)),
                'columns': ','.join(data.columns),
                'ingestion_date': datetime.now().isoformat(),
                'schema_version': '1.0',
                'format': 'parquet'
            }
            blob.metadata = metadata
            blob.patch()
            
            logger.info(f"Successfully stored gold data at gs://{self.bucket_name}/{blob_path}")
            return True
            
        except ValueError as e:
            error_msg = f"Invalid date format: {date}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Error storing gold data: {str(e)}"
            logger.error(error_msg)
            raise

    def _update_metadata(self, data: pd.DataFrame) -> None:
        """Update processing metadata with results."""
        try:
            logger.info("Updating processing metadata")
            
            # Update processing statistics
            self.metadata['total_records'] = len(data)
            self.metadata['columns'] = list(data.columns)
            self.metadata['last_updated'] = datetime.now().isoformat()
            
            # Store metadata
            metadata_blob = self.bucket.blob(
                f"{self.gold_prefix}/metadata/{self.metadata['processing_date']}/processing_metadata.json"
            )
            metadata_blob.upload_from_string(
                json.dumps(self.metadata, indent=2),
                content_type='application/json'
            )
            
            logger.info("Successfully updated processing metadata")
            
        except Exception as e:
            logger.error(f"Error updating metadata: {str(e)}")
            self.metadata['status'] = 'failed'
            self.metadata['error'] = str(e)
            raise 