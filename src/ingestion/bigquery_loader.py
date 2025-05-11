import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import json
import pandas as pd
import numpy as np

from google.cloud import bigquery
from google.cloud import storage
from google.cloud.storage import Client

logger = logging.getLogger(__name__)

class BigQueryLoader:
    """
    Handles loading gold layer data from GCS into BigQuery for SQL querying.
    Includes data quality validation and monitoring.
    """
    
    # Default validation configuration
    DEFAULT_VALIDATION_CONFIG = {
        "min_quality_score": 90,
        "required_fields": [
            "date", "open_price", "high_price", "low_price", 
            "close_price", "volume", "daily_return"
        ],
        "value_ranges": {
            "open_price": {"min": 0},
            "high_price": {"min": 0},
            "low_price": {"min": 0},
            "close_price": {"min": 0},
            "volume": {"min": 0},
            "daily_return": {"min": -1, "max": 1}
        },
        "data_types": {
            "date": "DATE",
            "open_price": "FLOAT64",
            "high_price": "FLOAT64",
            "low_price": "FLOAT64",
            "close_price": "FLOAT64",
            "volume": "INTEGER",
            "daily_return": "FLOAT64",
            "price_range": "FLOAT64",
            "sma_20": "FLOAT64",
            "sma_50": "FLOAT64",
            "volatility": "FLOAT64",
            "price_momentum": "FLOAT64",
            "volume_trend": "FLOAT64",
            "price_range_pct": "FLOAT64"
        }
    }
    
    def __init__(
        self,
        project_id: str,
        dataset_id: str,
        table_id: str,
        bucket_name: str,
        gold_prefix: str = "gold",
        storage_client: Optional[Client] = None,
        bigquery_client: Optional[bigquery.Client] = None,
        validation_config: Optional[Dict] = None
    ):
        """
        Initialize the BigQueryLoader.

        Args:
            project_id (str): Google Cloud project ID
            dataset_id (str): BigQuery dataset ID
            table_id (str): BigQuery table ID
            bucket_name (str): GCS bucket name
            gold_prefix (str): Prefix for gold data in the bucket
            storage_client (Optional[Client]): GCS client instance
            bigquery_client (Optional[bigquery.Client]): BigQuery client instance
            validation_config (Optional[Dict]): Custom validation configuration
        """
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.bucket_name = bucket_name
        self.gold_prefix = gold_prefix
        
        # Initialize clients
        self.storage_client = storage_client or storage.Client()
        self.bigquery_client = bigquery_client or bigquery.Client(project=project_id)
        self.bucket = self.storage_client.bucket(bucket_name)
        
        # Set validation configuration
        self.validation_config = validation_config or self.DEFAULT_VALIDATION_CONFIG
        
        # Initialize metadata with enhanced quality tracking
        self.metadata = {
            "last_loaded_date": None,
            "total_records_loaded": 0,
            "status": "initialized",
            "errors": [],
            "quality_metrics": {
                "current": {
                    "quality_score": 0.0,
                    "validation_results": {},
                    "timestamp": None
                },
                "history": []
            }
        }

    def _validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data before loading to BigQuery.

        Args:
            data (pd.DataFrame): Data to validate

        Returns:
            Dict[str, Any]: Validation results including:
                - schema_validation: Schema compatibility results
                - data_type_validation: Data type validation results
                - value_range_validation: Value range check results
                - required_field_validation: Required field check results
                - duplicate_validation: Duplicate check results
                - quality_score: Overall quality score (0-100)
        """
        try:
            results = {
                "schema_validation": self._validate_schema(data),
                "data_type_validation": self._validate_data_types(data),
                "value_range_validation": self._validate_value_ranges(data),
                "required_field_validation": self._validate_required_fields(data),
                "duplicate_validation": self._validate_duplicates(data),
                "quality_score": 0.0,
                "errors": []
            }

            # Calculate quality score
            weights = {
                "schema_validation": 0.2,
                "data_type_validation": 0.2,
                "value_range_validation": 0.2,
                "required_field_validation": 0.2,
                "duplicate_validation": 0.2
            }

            # Calculate weighted average
            quality_score = sum(
                results[component]["score"] * weight
                for component, weight in weights.items()
                if "score" in results[component]
            )

            results["quality_score"] = float(quality_score)
            
            # Log validation results
            logger.info(f"Data validation completed. Quality score: {quality_score:.2f}")
            if quality_score < self.validation_config["min_quality_score"]:
                logger.warning(f"Data quality score below threshold: {quality_score:.2f}")

            return results

        except Exception as e:
            error_msg = f"Error during data validation: {str(e)}"
            logger.error(error_msg)
            return {
                "quality_score": 0.0,
                "errors": [error_msg]
            }

    def _validate_schema(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data schema against expected schema."""
        try:
            expected_columns = set(self.validation_config["data_types"].keys())
            actual_columns = set(data.columns)
            
            missing_columns = expected_columns - actual_columns
            extra_columns = actual_columns - expected_columns
            
            is_valid = len(missing_columns) == 0
            score = 100 if is_valid else 0
            
            return {
                "is_valid": is_valid,
                "score": score,
                "missing_columns": list(missing_columns),
                "extra_columns": list(extra_columns)
            }
        except Exception as e:
            logger.error(f"Error in schema validation: {str(e)}")
            return {"is_valid": False, "score": 0, "error": str(e)}

    def _validate_data_types(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data types match expected BigQuery types."""
        try:
            type_mapping = {
                "DATE": ["datetime64[ns]", "date"],
                "FLOAT64": ["float64", "float32"],
                "INTEGER": ["int64", "int32"]
            }
            
            invalid_types = {}
            for column, expected_type in self.validation_config["data_types"].items():
                if column in data.columns:
                    actual_type = str(data[column].dtype)
                    if expected_type in type_mapping:
                        if actual_type not in type_mapping[expected_type]:
                            invalid_types[column] = {
                                "expected": expected_type,
                                "actual": actual_type
                            }
            
            is_valid = len(invalid_types) == 0
            score = 100 if is_valid else 0
            
            return {
                "is_valid": is_valid,
                "score": score,
                "invalid_types": invalid_types
            }
        except Exception as e:
            logger.error(f"Error in data type validation: {str(e)}")
            return {"is_valid": False, "score": 0, "error": str(e)}

    def _validate_value_ranges(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate value ranges for numeric columns."""
        try:
            invalid_ranges = {}
            for column, ranges in self.validation_config["value_ranges"].items():
                if column in data.columns:
                    if "min" in ranges and data[column].min() < ranges["min"]:
                        invalid_ranges[column] = {"min": data[column].min(), "expected_min": ranges["min"]}
                    if "max" in ranges and data[column].max() > ranges["max"]:
                        invalid_ranges[column] = {"max": data[column].max(), "expected_max": ranges["max"]}
            
            is_valid = len(invalid_ranges) == 0
            score = 100 if is_valid else 0
            
            return {
                "is_valid": is_valid,
                "score": score,
                "invalid_ranges": invalid_ranges
            }
        except Exception as e:
            logger.error(f"Error in value range validation: {str(e)}")
            return {"is_valid": False, "score": 0, "error": str(e)}

    def _validate_required_fields(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate presence of required fields."""
        try:
            missing_fields = [
                field for field in self.validation_config["required_fields"]
                if field not in data.columns
            ]
            
            is_valid = len(missing_fields) == 0
            score = 100 if is_valid else 0
            
            return {
                "is_valid": is_valid,
                "score": score,
                "missing_fields": missing_fields
            }
        except Exception as e:
            logger.error(f"Error in required fields validation: {str(e)}")
            return {"is_valid": False, "score": 0, "error": str(e)}

    def _validate_duplicates(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate for duplicate records."""
        try:
            duplicate_count = data.duplicated().sum()
            is_valid = duplicate_count == 0
            score = 100 if is_valid else 0
            
            return {
                "is_valid": is_valid,
                "score": score,
                "duplicate_count": int(duplicate_count)
            }
        except Exception as e:
            logger.error(f"Error in duplicate validation: {str(e)}")
            return {"is_valid": False, "score": 0, "error": str(e)}

    def _update_metadata(self, job: bigquery.LoadJob, validation_results: Dict[str, Any]) -> None:
        """Update loading metadata with job results and validation metrics."""
        try:
            logger.info("Updating loading metadata")
            
            # Update current quality metrics
            self.metadata["quality_metrics"]["current"] = {
                "quality_score": float(validation_results["quality_score"]),
                "validation_results": validation_results,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add to history
            self.metadata["quality_metrics"]["history"].append(
                self.metadata["quality_metrics"]["current"]
            )
            
            # Keep only last 30 days of history
            if len(self.metadata["quality_metrics"]["history"]) > 30:
                self.metadata["quality_metrics"]["history"] = \
                    self.metadata["quality_metrics"]["history"][-30:]
            
            # Update statistics
            self.metadata["total_records_loaded"] = int(job.output_rows)
            self.metadata["last_updated"] = datetime.now().isoformat()

            # Helper to convert NumPy types to native Python types
            def convert(obj):
                if isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert(i) for i in obj]
                elif isinstance(obj, (np.integer, np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                else:
                    return obj

            serializable_metadata = convert(self.metadata)
            
            # Store metadata
            metadata_blob = self.bucket.blob(
                f"{self.gold_prefix}/bigquery_metadata/{self.metadata['last_loaded_date']}/loading_metadata.json"
            )
            metadata_blob.upload_from_string(
                json.dumps(serializable_metadata, indent=2),
                content_type='application/json'
            )
            
            logger.info("Successfully updated loading metadata")
            
        except Exception as e:
            logger.error(f"Error updating metadata: {str(e)}")
            self.metadata["status"] = "failed"
            self.metadata["errors"].append(str(e))
            raise

    def load_daily_data(self, date: str) -> bool:
        """
        Load gold layer data for a specific date into BigQuery.

        Args:
            date (str): Date to load in YYYY-MM-DD format

        Returns:
            bool: True if loading was successful, False otherwise
        """
        try:
            logger.info(f"Starting BigQuery load for date: {date}")
            self.metadata["last_loaded_date"] = date
            self.metadata["status"] = "loading"

            # 1. Get GCS file path
            year, month, day = date.split('-')
            gcs_path = f"gs://{self.bucket_name}/{self.gold_prefix}/{year}/{month}/{day}/data.parquet"
            
            # 2. Read and validate data
            try:
                data = pd.read_parquet(gcs_path)
                validation_results = self._validate_data(data)
                
                # Check quality score
                if validation_results["quality_score"] < self.validation_config["min_quality_score"]:
                    logger.warning(
                        f"Data quality score {validation_results['quality_score']:.2f} "
                        f"below threshold {self.validation_config['min_quality_score']}"
                    )
                    self.metadata["status"] = "failed"
                    self.metadata["errors"].append("Data quality score below threshold")
                    return False
                    
            except Exception as e:
                logger.error(f"Error reading or validating data: {str(e)}")
                self.metadata["status"] = "failed"
                self.metadata["errors"].append(str(e))
                return False
            
            # 3. Create or get table reference
            table_ref = self._get_or_create_table()
            
            # 4. Load data
            job_config = bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.PARQUET,
                write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
                schema_update_options=[
                    bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION
                ]
            )
            
            load_job = self.bigquery_client.load_table_from_uri(
                gcs_path,
                table_ref,
                job_config=job_config
            )
            
            # Wait for job to complete
            load_job.result()
            
            # 5. Update metadata
            self._update_metadata(load_job, validation_results)
            
            self.metadata["status"] = "success"
            logger.info(f"Successfully loaded data for date: {date}")
            return True

        except Exception as e:
            self.metadata["status"] = "failed"
            self.metadata["errors"].append(str(e))
            logger.error(f"Error loading data for date {date}: {str(e)}")
            return False

    def _get_or_create_table(self) -> bigquery.TableReference:
        """
        Get or create the BigQuery table with appropriate schema.

        Returns:
            bigquery.TableReference: Reference to the BigQuery table
        """
        dataset_ref = self.bigquery_client.dataset(self.dataset_id)
        table_ref = dataset_ref.table(self.table_id)
        
        try:
            # Try to get existing table
            self.bigquery_client.get_table(table_ref)
            logger.info(f"Using existing table: {self.table_id}")
            return table_ref
            
        except Exception:
            # Create new table with schema
            logger.info(f"Creating new table: {self.table_id}")
            
            schema = [
                bigquery.SchemaField("date", "DATE"),
                bigquery.SchemaField("open_price", "FLOAT64"),
                bigquery.SchemaField("high_price", "FLOAT64"),
                bigquery.SchemaField("low_price", "FLOAT64"),
                bigquery.SchemaField("close_price", "FLOAT64"),
                bigquery.SchemaField("volume", "INTEGER"),
                bigquery.SchemaField("daily_return", "FLOAT64"),
                bigquery.SchemaField("price_range", "FLOAT64"),
                bigquery.SchemaField("sma_20", "FLOAT64"),
                bigquery.SchemaField("sma_50", "FLOAT64"),
                bigquery.SchemaField("volatility", "FLOAT64"),
                bigquery.SchemaField("price_momentum", "FLOAT64"),
                bigquery.SchemaField("volume_trend", "FLOAT64"),
                bigquery.SchemaField("price_range_pct", "FLOAT64"),
                bigquery.SchemaField("data_source", "STRING"),
                bigquery.SchemaField("data_layer", "STRING"),
                bigquery.SchemaField("last_updated", "TIMESTAMP")
            ]
            
            table = bigquery.Table(table_ref, schema=schema)
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field="date"
            )
            
            return self.bigquery_client.create_table(table)
            
    def load_incremental_data(self, end_date: str = None, start_date: str = None):
        """
        Load only new stock data since the last processed date.
        
        Args:
            end_date (str, optional): The end date to process up to (defaults to today)
            start_date (str, optional): The start date to process from (defaults to day after last loaded date)
            
        Returns:
            bool: True if loading was successful
        """
        # Determine the date range to process
        if not start_date:
            last_loaded_date = self._get_last_processed_date()
            
            if not last_loaded_date:
                logger.info("No previous load found. Starting with initial load.")
                # You might want to define how far back to go for initial load
                start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            else:
                # Start from the day after the last loaded date
                start_date = (datetime.strptime(last_loaded_date, "%Y-%m-%d") + 
                            timedelta(days=1)).strftime("%Y-%m-%d")
        
        # Default end_date to today if not specified
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        # Skip if already up to date
        if start_date > end_date:
            logger.info(f"Data already up to date. Last loaded: {last_loaded_date if 'last_loaded_date' in locals() else 'None'}")
            return True
        
        logger.info(f"Loading incremental data from {start_date} to {end_date}")
        
        # Generate list of dates to process
        dates_to_process = self._generate_date_range(start_date, end_date)
        
        # Filter out weekends and holidays
        trading_dates = self._filter_trading_days(dates_to_process)
        
        if not trading_dates:
            logger.info("No trading dates to process in the specified range")
            return True
            
        success = True
        processed_dates = []
        
        for date in trading_dates:
            logger.info(f"Processing date: {date}")
            if self.load_daily_data(date):
                processed_dates.append(date)
            else:
                logger.error(f"Failed to load data for {date}")
                success = False
                # Continue with other dates instead of stopping
        
        logger.info(f"Incremental load completed. Processed {len(processed_dates)} out of {len(trading_dates)} dates.")
        return success

    def _get_last_processed_date(self):
        """Get the last date for which data was successfully loaded."""
        
        # Query BigQuery for the most recent date
        query = f"""
            SELECT MAX(date) as last_date 
            FROM `{self.project_id}.{self.dataset_id}.{self.table_id}`
        """
        try:
            query_job = self.bigquery_client.query(query)
            results = query_job.result()
            for row in results:
                if row.last_date:
                    return row.last_date.strftime("%Y-%m-%d")
        except Exception as e:
            logger.warning(f"Could not query last processed date: {str(e)}")
        
        return None

    def _generate_date_range(self, start_date, end_date):
        """Generate a list of dates between start_date and end_date inclusive."""
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        date_list = []
        current_dt = start_dt
        while current_dt <= end_dt:
            date_list.append(current_dt.strftime("%Y-%m-%d"))
            current_dt += timedelta(days=1)
        
        return date_list

    def _get_us_market_holidays(self, year):
        """
        Generate US stock market holidays for a given year.
        
        Args:
            year (int): The year to generate holidays for
            
        Returns:
            list: List of holiday dates in YYYY-MM-DD format
        """
        holidays = []
        
        # New Year's Day (January 1st)
        new_years = datetime(year, 1, 1)
        # If New Year's falls on a weekend, observe on Monday after or Friday before
        if new_years.weekday() == 5:  # Saturday
            holidays.append(datetime(year-1, 12, 31).strftime("%Y-%m-%d"))  # Observe on Friday before
        elif new_years.weekday() == 6:  # Sunday
            holidays.append(datetime(year, 1, 2).strftime("%Y-%m-%d"))  # Observe on Monday after
        else:
            holidays.append(new_years.strftime("%Y-%m-%d"))
        
        # Martin Luther King Jr. Day (3rd Monday in January)
        mlk_day = datetime(year, 1, 1)
        while mlk_day.weekday() != 0:  # Find first Monday
            mlk_day += timedelta(days=1)
        mlk_day += timedelta(days=14)  # Advance to 3rd Monday
        holidays.append(mlk_day.strftime("%Y-%m-%d"))
        
        # Presidents' Day (3rd Monday in February)
        presidents_day = datetime(year, 2, 1)
        while presidents_day.weekday() != 0:  # Find first Monday
            presidents_day += timedelta(days=1)
        presidents_day += timedelta(days=14)  # Advance to 3rd Monday
        holidays.append(presidents_day.strftime("%Y-%m-%d"))
        
        # Memorial Day (Last Monday in May)
        memorial_day = datetime(year, 6, 1) - timedelta(days=1)
        while memorial_day.weekday() != 0:  # Find last Monday in May
            memorial_day -= timedelta(days=1)
        holidays.append(memorial_day.strftime("%Y-%m-%d"))
        
        # Juneteenth (June 19th)
        juneteenth = datetime(year, 6, 19)
        # If Juneteenth falls on a weekend, observe on Monday after or Friday before
        if juneteenth.weekday() == 5:  # Saturday
            holidays.append(datetime(year, 6, 18).strftime("%Y-%m-%d"))  # Observe on Friday before
        elif juneteenth.weekday() == 6:  # Sunday
            holidays.append(datetime(year, 6, 20).strftime("%Y-%m-%d"))  # Observe on Monday after
        else:
            holidays.append(juneteenth.strftime("%Y-%m-%d"))
        
        # Independence Day (July 4th)
        independence_day = datetime(year, 7, 4)
        # If July 4th falls on a weekend, observe on Monday after or Friday before
        if independence_day.weekday() == 5:  # Saturday
            holidays.append(datetime(year, 7, 3).strftime("%Y-%m-%d"))  # Observe on Friday before
        elif independence_day.weekday() == 6:  # Sunday
            holidays.append(datetime(year, 7, 5).strftime("%Y-%m-%d"))  # Observe on Monday after
        else:
            holidays.append(independence_day.strftime("%Y-%m-%d"))
        
        # Labor Day (1st Monday in September)
        labor_day = datetime(year, 9, 1)
        while labor_day.weekday() != 0:  # Find first Monday
            labor_day += timedelta(days=1)
        holidays.append(labor_day.strftime("%Y-%m-%d"))
        
        # Thanksgiving Day (4th Thursday in November)
        thanksgiving = datetime(year, 11, 1)
        while thanksgiving.weekday() != 3:  # Find first Thursday
            thanksgiving += timedelta(days=1)
        thanksgiving += timedelta(days=21)  # Advance to 4th Thursday
        holidays.append(thanksgiving.strftime("%Y-%m-%d"))
        
        # Christmas Day (December 25th)
        christmas = datetime(year, 12, 25)
        # If Christmas falls on a weekend, observe on Monday after or Friday before
        if christmas.weekday() == 5:  # Saturday
            holidays.append(datetime(year, 12, 24).strftime("%Y-%m-%d"))  # Observe on Friday before
        elif christmas.weekday() == 6:  # Sunday
            holidays.append(datetime(year, 12, 26).strftime("%Y-%m-%d"))  # Observe on Monday after
        else:
            holidays.append(christmas.strftime("%Y-%m-%d"))
        
        return holidays

    def _filter_trading_days(self, date_list):
        """Filter out weekends and holidays from the date list."""
        trading_days = []
        
        # Generate holiday list for all years in the date range
        years = set()
        holidays = []
        
        # Collect years from date_list
        for date_str in date_list:
            date_dt = datetime.strptime(date_str, "%Y-%m-%d")
            years.add(date_dt.year)
        
        # Generate holidays for all years in the range
        for year in years:
            holidays.extend(self._get_us_market_holidays(year))
        
        for date_str in date_list:
            date_dt = datetime.strptime(date_str, "%Y-%m-%d")
            
            # Skip weekends (5=Saturday, 6=Sunday)
            if date_dt.weekday() >= 5:
                logger.debug(f"Skipping weekend: {date_str}")
                continue
            
            # Skip holidays
            if date_str in holidays:
                logger.debug(f"Skipping holiday: {date_str}")
                continue
            
            trading_days.append(date_str)
        
        return trading_days 