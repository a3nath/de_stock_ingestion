import logging
import os
from datetime import datetime, timedelta

# import boto3  # Commented out AWS SDK
from google.cloud import storage  # Added Google Cloud Storage
import pandas as pd
import yfinance as yf
# from botocore.exceptions import ClientError  # Commented out AWS exception
from google.cloud.exceptions import GoogleCloudError  # Added Google Cloud exception
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class FinanceIngestor:
    def __init__(self, bucket_name=None, project_id=None):
        """
        Initialize the FinanceIngestor with Google Cloud Storage bucket name.
        
        Args:
            bucket_name: GCS bucket name (defaults to env var GCS_BUCKET_NAME)
            project_id: Google Cloud project ID (defaults to env var GCS_PROJECT_ID)
        """
        # Use bucket_name if provided, otherwise get from environment variable
        self.bucket_name = bucket_name or os.getenv("GCS_BUCKET_NAME")
        if not self.bucket_name:
            raise ValueError("GCS bucket name must be provided or set in GCS_BUCKET_NAME environment variable")
            
        # Use project_id if provided, otherwise get from environment variable
        self.project_id = project_id or os.getenv("GCS_PROJECT_ID")
        if not self.project_id:
            raise ValueError("Google Cloud project ID must be provided or set in GCS_PROJECT_ID environment variable")
            
        # Create a Google Cloud Storage client
        self.storage_client = storage.Client(project=self.project_id)
        
        # Get the bucket (or create it if it doesn't exist)
        try:
            self.bucket = self.storage_client.get_bucket(self.bucket_name)
            logger.info(f"Using existing bucket: {self.bucket_name}")
        except GoogleCloudError:
            logger.info(f"Creating new bucket: {self.bucket_name}")
            self.bucket = self.storage_client.create_bucket(self.bucket_name)

    def fetch_data(self, symbol, start_date, end_date):
        """
        Fetch financial data from Yahoo Finance.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame containing the financial data
        """
        try:
            # Create a Yahoo Finance ticker object
            ticker = yf.Ticker(symbol)
            
            # Get historical data
            df = ticker.history(start=start_date, end=end_date)
            
            # Check if we got any data
            if df.empty:
                logger.warning(f"No data received for {symbol}")
                raise ValueError(f"No data received for {symbol}")
            
            logger.info(f"Successfully fetched {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise

    def validate_data(self, df, symbol):
        """
        Validate the DataFrame to ensure data quality.
        
        Args:
            df: DataFrame to validate
            symbol: Stock symbol for error messages
            
        Returns:
            True if validation passes
            
        Raises:
            ValueError: If validation fails with detailed error message
        """
        validation_errors = []
        
        # 1. Basic Structure Validation
        # Check if DataFrame is empty
        if df.empty:
            validation_errors.append("DataFrame is empty")
            error_message = f"Validation failed for {symbol} data: DataFrame is empty"
            logger.error(error_message)
            raise ValueError(error_message)
        
        # Log the actual columns we received for debugging
        logger.info(f"Columns received for {symbol}: {', '.join(df.columns)}")
        
        # Check for required columns - updated to match current Yahoo Finance API response
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_errors.append(f"Missing required columns: {', '.join(missing_columns)}")
        
        # 2. Data Type Validation
        # Only check columns that exist in the DataFrame
        numeric_columns = [col for col in required_columns if col in df.columns]
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                validation_errors.append(f"Column '{col}' is not numeric")
        
        # 3. Completeness Validation
        # Check for missing values in required columns that exist
        for col in numeric_columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                validation_errors.append(f"Column '{col}' has {missing_count} missing values")
        
        # If we have any validation errors, raise an exception
        if validation_errors:
            error_message = f"Validation failed for {symbol} data:\n" + "\n".join(validation_errors)
            logger.error(error_message)
            raise ValueError(error_message)
        
        logger.info(f"Validation passed for {symbol} data with {len(df)} records")
        return True

    def upload_to_gcs(self, df, symbol):
        """
        Upload DataFrame to Google Cloud Storage.
        
        Args:
            df: DataFrame to upload
            symbol: Stock symbol for file naming
            
        Returns:
            GCS blob path where the data was uploaded
        """
        try:
            # Add metadata
            df = df.copy()  # Create a copy to avoid modifying the original
            df['ingestion_date'] = datetime.now().isoformat()
            df['source'] = 'yahoo_finance'
            
            # Convert to JSON
            json_data = df.to_json(orient='records')
            
            # Generate GCS blob path (file path in GCS)
            date_str = datetime.now().strftime('%Y/%m/%d')
            blob_path = f"raw/finance/{symbol}/{date_str}/data.json"
            
            # Upload to GCS
            blob = self.bucket.blob(blob_path)
            blob.upload_from_string(
                json_data,
                content_type='application/json'
            )
            
            # Set metadata
            metadata = {
                'symbol': symbol,
                'records': str(len(df)),
                'ingestion_date': datetime.now().isoformat()
            }
            blob.metadata = metadata
            blob.patch()
            
            logger.info(f"Successfully uploaded data for {symbol} to gs://{self.bucket_name}/{blob_path}")
            return blob_path
            
        except GoogleCloudError as e:
            logger.error(f"Error uploading to Google Cloud Storage: {str(e)}")
            raise

    def process_daily_data(self, symbol):
        """
        Process daily data for a given symbol.
        
        Args:
            symbol: Stock symbol to process
        """
        try:
            # Calculate date range (yesterday to today)
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            
            logger.info(f"Processing data for {symbol} from {start_date} to {end_date}")
            
            # Fetch data
            df = self.fetch_data(symbol, start_date, end_date)
            
            # Validate data
            self.validate_data(df, symbol)
            
            # Upload to Google Cloud Storage
            blob_path = self.upload_to_gcs(df, symbol)
            
            return blob_path
            
        except Exception as e:
            logger.error(f"Error processing daily data for {symbol}: {str(e)}")
            raise

def main():
    """Main function to run the ingestor."""
    # Load environment variables
    load_dotenv()
    
    # Get bucket name and project ID from environment variables
    bucket_name = os.getenv("GCS_BUCKET_NAME")
    project_id = os.getenv("GCS_PROJECT_ID")
    
    # Get stock symbol from environment variable or use default
    symbol = os.getenv("STOCK_SYMBOL", "AAPL")
    
    # Create ingestor
    ingestor = FinanceIngestor(bucket_name=bucket_name, project_id=project_id)
    
    # Process data
    try:
        blob_path = ingestor.process_daily_data(symbol)
        print(f"Successfully processed {symbol} data. Stored at gs://{bucket_name}/{blob_path}")
    except Exception as e:
        print(f"Error processing {symbol} data: {str(e)}")

if __name__ == "__main__":
    main() 