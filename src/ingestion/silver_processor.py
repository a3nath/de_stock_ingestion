import logging
from datetime import datetime
from typing import Dict, List, Optional
import json

import pandas as pd
import numpy as np
from google.cloud import storage
from google.cloud.storage import Client

logger = logging.getLogger(__name__)

class SilverProcessor:
    """
    Handles the processing of raw data into the silver layer.
    Implements data quality checks, transformations, and metadata management.
    """
    
    def __init__(
        self,
        bucket_name: str,
        raw_prefix: str = "raw",
        silver_prefix: str = "silver",
        storage_client: Optional[Client] = None
    ):
        """
        Initialize the SilverProcessor.

        Args:
            bucket_name (str): Name of the GCS bucket
            raw_prefix (str): Prefix for raw data in the bucket
            silver_prefix (str): Prefix for silver data in the bucket
            storage_client (Optional[Client]): GCS client instance
        """
        self.bucket_name = bucket_name
        self.raw_prefix = raw_prefix
        self.silver_prefix = silver_prefix
        self.storage_client = storage_client or storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)
        
        # Initialize metadata
        self.metadata = {
            "processing_date": None,
            "source_files": [],
            "record_count": 0,
            "quality_score": 0.0,
            "errors": [],
            "status": "initialized"
        }

    def process_daily_data(self, date: str) -> bool:
        """
        Process raw data for a specific date into the silver layer.

        Args:
            date (str): Date to process in YYYY-MM-DD format

        Returns:
            bool: True if processing was successful, False otherwise
        """
        try:
            logger.info(f"Starting silver layer processing for date: {date}")
            self.metadata["processing_date"] = date
            self.metadata["status"] = "processing"

            # Read raw data
            raw_data = self._read_raw_data(date)
            if raw_data.empty:
                logger.warning(f"No data found for date: {date}")
                return False

            # TODO: Implement remaining steps:
            # 2. Apply data quality checks
            # 3. Transform data if needed
            # 4. Store in silver layer
            # 5. Update metadata
            # 6. Handle errors

            self.metadata["status"] = "completed"
            logger.info(f"Successfully processed data for date: {date}")
            return True

        except Exception as e:
            self.metadata["status"] = "failed"
            self.metadata["errors"].append(str(e))
            logger.error(f"Error processing data for date {date}: {str(e)}")
            return False

    def _read_raw_data(self, date: str) -> pd.DataFrame:
        """
        Read raw data from GCS for a specific date.

        Args:
            date (str): Date to read in YYYY-MM-DD format

        Returns:
            pd.DataFrame: Raw data

        Raises:
            ValueError: If date format is invalid
            Exception: If there are issues reading from GCS
        """
        try:
            # Validate date format
            datetime.strptime(date, "%Y-%m-%d")
            
            # List all blobs in the raw prefix for the given date
            prefix = f"{self.raw_prefix}/{date}/"
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
                data = json.loads(content)
                
                # Convert to DataFrame
                df = pd.DataFrame([data])
                all_data.append(df)
            
            if not all_data:
                logger.warning(f"No valid JSON files found in {prefix}")
                return pd.DataFrame()
            
            # Combine all data
            combined_data = pd.concat(all_data, ignore_index=True)
            self.metadata["record_count"] = len(combined_data)
            
            logger.info(f"Successfully read {len(combined_data)} records from raw layer")
            return combined_data
            
        except ValueError as e:
            logger.error(f"Invalid date format: {date}")
            raise ValueError(f"Date must be in YYYY-MM-DD format: {str(e)}")
        except Exception as e:
            logger.error(f"Error reading raw data: {str(e)}")
            raise

    def _apply_quality_checks(self, data: pd.DataFrame) -> Dict:
        """
        Apply data quality checks to the dataset.

        Args:
            data (pd.DataFrame): Data to check

        Returns:
            Dict: Quality check results including:
                - completeness: Percentage of non-null values
                - validity: Results of business rule validations
                - uniqueness: Duplicate check results
                - consistency: Data type and format checks
                - quality_score: Overall quality score (0-100)
        """
        try:
            results = {
                "completeness": {},
                "validity": {},
                "uniqueness": {},
                "consistency": {},
                "quality_score": 0.0,
                "errors": []
            }

            # 1. Completeness Check
            for column in data.columns:
                null_count = data[column].isnull().sum()
                total_count = len(data)
                completeness = ((total_count - null_count) / total_count) * 100
                results["completeness"][column] = {
                    "null_count": int(null_count),
                    "completeness_percentage": float(completeness)
                }

            # 2. Validity Check (Business Rules)
            # Check for valid price ranges
            price_columns = ['Open', 'High', 'Low', 'Close']
            for col in price_columns:
                if col in data.columns:
                    #Check if the column is numeric
                    results["validity"][f"{col}_validity_type"] = {
                        "is_valid": data[col].dtype == 'float64'
                    }
            
            for col in price_columns:
                if col in data.columns:
                    invalid_prices = data[data[col] <= 0].shape[0]
                    results["validity"][f"{col}_validity_range"] = {
                        "invalid_count": int(invalid_prices),
                        "is_valid": invalid_prices == 0
                    }

            # Check High >= Open, Close, Low
            if all(col in data.columns for col in ['High', 'Open', 'Close', 'Low']):
                invalid_highs = data[
                    (data['High'] < data['Open']) | 
                    (data['High'] < data['Close']) | 
                    (data['High'] < data['Low'])
                ].shape[0]
                results["validity"]["high_price_validity"] = {
                    "invalid_count": int(invalid_highs),
                    "is_valid": invalid_highs == 0
                }
                
            #Check Low <= Open, Close, High
            if all(col in data.columns for col in ['High', 'Open', 'Low', 'Close']):
                invalid_lows = data[
                    (data['Low'] > data['Open']) |
                    (data['Low'] > data['Close']) |
                    (data['Low'] > data['High'])
                ].shape[0]
                results["validity"]["low_price_validity"] = {
                    "invalid_count": int(invalid_lows),
                    "is_valid": invalid_lows == 0
                }

            # 3. Uniqueness Check
            duplicate_rows = data.duplicated().sum()
            results["uniqueness"] = {
                "duplicate_count": int(duplicate_rows),
                "is_unique": duplicate_rows == 0
            }

            # 4. Consistency Check
            # Check date format consistency
            if 'date' in data.columns:
                try:
                    data['date'] = pd.to_datetime(data['date'])
                    results["consistency"]["date_format"] = {
                        "is_consistent": True
                    }
                except Exception as e:
                    results["consistency"]["date_format"] = {
                        "is_consistent": False,
                        "error": str(e)
                    }

            # Calculate overall quality score
            weights = {
                "completeness": 0.25,
                "validity": 0.25,
                "uniqueness": 0.25,
                "consistency": 0.25
            }

            # Calculate completeness score
            completeness_scores = [
                result["completeness_percentage"] 
                for result in results["completeness"].values()
            ]
            completeness_score = sum(completeness_scores) / len(completeness_scores)

            # Calculate validity score
            validity_scores = [
                100 if result["is_valid"] else 0 
                for result in results["validity"].values()
            ]
            validity_score = sum(validity_scores) / len(validity_scores) if validity_scores else 100

            # Calculate uniqueness score
            uniqueness_score = 100 if results["uniqueness"]["is_unique"] else 0

            # Calculate consistency score
            consistency_scores = [
                100 if result.get("is_consistent", True) else 0 
                for result in results["consistency"].values()
            ]
            consistency_score = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 100

            # Calculate weighted average
            quality_score = (
                completeness_score * weights["completeness"] +
                validity_score * weights["validity"] +
                uniqueness_score * weights["uniqueness"] +
                consistency_score * weights["consistency"]
            )

            results["quality_score"] = float(quality_score)
            
            # Log quality check results
            logger.info(f"Data quality check completed. Overall score: {quality_score:.2f}")
            if quality_score < 90:
                logger.warning(f"Data quality score below threshold: {quality_score:.2f}")

            return results

        except Exception as e:
            error_msg = f"Error during quality checks: {str(e)}"
            logger.error(error_msg)
            results["errors"].append(error_msg)
            results["quality_score"] = 0.0
            return results

    def _transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data according to silver layer requirements.

        Args:
            data (pd.DataFrame): Raw data to transform

        Returns:
            pd.DataFrame: Transformed data ready for silver layer
        """
        try:
            logger.info("Starting data transformation")
            
            # Create a copy to avoid modifying original data
            transformed_data = data.copy()
            
            # 1. Standardize column names
            column_mapping = {
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
                'date': 'Date',
                'symbol': 'Symbol'
            }
            transformed_data.columns = [column_mapping.get(col.lower(), col) for col in transformed_data.columns]
            
            # 2. Convert data types
            # Convert date to datetime
            if 'Date' in transformed_data.columns:
                transformed_data['Date'] = pd.to_datetime(transformed_data['Date'])
            
            # Convert price columns to float
            price_columns = ['Open', 'High', 'Low', 'Close']
            for col in price_columns:
                if col in transformed_data.columns:
                    transformed_data[col] = pd.to_numeric(transformed_data[col], errors='coerce')
            
            # Convert volume to integer
            if 'Volume' in transformed_data.columns:
                transformed_data['Volume'] = pd.to_numeric(transformed_data['Volume'], errors='coerce').fillna(0).astype(int)
            
            # 3. Add derived columns
            # Calculate daily returns
            if all(col in transformed_data.columns for col in ['Close', 'Open']):
                transformed_data['Daily_Return'] = (
                    (transformed_data['Close'] - transformed_data['Open']) / transformed_data['Open'] * 100
                ).round(2)
            
            # Calculate price range
            if all(col in transformed_data.columns for col in ['High', 'Low']):
                transformed_data['Price_Range'] = (
                    transformed_data['High'] - transformed_data['Low']
                ).round(2)
            
            # 4. Handle missing values
            # For price columns, forward fill missing values
            for col in price_columns:
                if col in transformed_data.columns:
                    transformed_data[col] = transformed_data[col].fillna(method='ffill')
            
            # For volume, fill with 0
            if 'Volume' in transformed_data.columns:
                transformed_data['Volume'] = transformed_data['Volume'].fillna(0)
            
            # 5. Add metadata columns
            transformed_data['Ingestion_Date'] = datetime.now().strftime('%Y-%m-%d')
            transformed_data['Data_Source'] = 'Yahoo Finance'
            transformed_data['Data_Layer'] = 'Silver'
            
            # 6. Sort by date
            if 'Date' in transformed_data.columns:
                transformed_data = transformed_data.sort_values('Date')
            
            # 7. Reset index
            transformed_data = transformed_data.reset_index(drop=True)
            
            logger.info(f"Data transformation completed. Shape: {transformed_data.shape}")
            return transformed_data
            
        except Exception as e:
            error_msg = f"Error during data transformation: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _store_silver_data(self, data: pd.DataFrame, date: str) -> bool:
        """
        Store processed data in the silver layer.

        Args:
            data (pd.DataFrame): Processed data to store
            date (str): Date of the data

        Returns:
            bool: True if storage was successful
        """
        # TODO: Implement silver data storage logic
        pass

    def _update_metadata(self, quality_results: Dict) -> None:
        """
        Update processing metadata with results.

        Args:
            quality_results (Dict): Results from quality checks
        """
        # TODO: Implement metadata update logic
        pass 