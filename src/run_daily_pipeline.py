#!/usr/bin/env python
import os
import sys
import logging
import traceback
from datetime import datetime, timedelta
from typing import List, Optional

# Set up proper Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import pipeline components
from src.ingestion.finance_ingestor import FinanceIngestor
from src.ingestion.silver_processor import SilverProcessor
from src.ingestion.gold_processor import GoldProcessor
from src.ingestion.bigquery_loader import BigQueryLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/pipeline_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("pipeline_runner")

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

def run_pipeline(date: Optional[str] = None, symbols: List[str] = None) -> bool:
    """
    Run the complete data pipeline from ingestion to BigQuery loading.
    
    Args:
        date: Optional date to process in YYYY-MM-DD format (defaults to today)
        symbols: List of stock symbols to process (defaults to predefined list)
        
    Returns:
        bool: True if pipeline completed successfully, False otherwise
    """
    start_time = datetime.now()
    logger.info(f"Starting pipeline run at {start_time}")
    
    # Default to processing today's data if not specified 
    # (assuming the pipeline runs after market close)
    if not date:
        today = datetime.now()
        date = today.strftime("%Y-%m-%d")
    
    # Default symbols if not provided
    if not symbols:
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "VT", "ITOT"]
    
    # Load environment variables
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    bucket_name = os.environ.get("GCS_BUCKET_NAME")
    
    if not project_id or not bucket_name:
        logger.error("Missing required environment variables: GOOGLE_CLOUD_PROJECT or GCS_BUCKET_NAME")
        return False
    
    logger.info(f"Processing date: {date}")
    logger.info(f"Processing symbols: {', '.join(symbols)}")
    
    try:
        # Step 1: Ingest raw data
        logger.info("Starting raw data ingestion")
        ingestor = FinanceIngestor(bucket_name=bucket_name)
        
        for symbol in symbols:
            try:
                logger.info(f"Ingesting data for {symbol}")
                success = ingestor.process_daily_data(symbol, date)
                if not success:
                    logger.warning(f"Failed to ingest data for {symbol} on {date}")
            except Exception as e:
                logger.error(f"Error ingesting {symbol}: {str(e)}")
                # Continue with other symbols
        
        # Step 2: Process silver layer
        logger.info("Starting silver layer processing")
        silver_processor = SilverProcessor(bucket_name=bucket_name)
        silver_success = silver_processor.process_daily_data(date)
        
        if not silver_success:
            logger.error("Silver layer processing failed")
            # Continue to next steps despite errors
        
        # Step 3: Process gold layer
        logger.info("Starting gold layer processing")
        gold_processor = GoldProcessor(bucket_name=bucket_name)
        gold_success = gold_processor.process_daily_data(date)
        
        if not gold_success:
            logger.error("Gold layer processing failed")
            # Continue to next steps despite errors
        
        # Step 4: Load to BigQuery
        logger.info("Starting BigQuery loading")
        bigquery_loader = BigQueryLoader(
            project_id=project_id,
            dataset_id="finance_data",
            table_id="stock_data",
            bucket_name=bucket_name
        )
        
        # Use incremental loading
        bq_success = bigquery_loader.load_incremental_data(
            start_date=date,
            end_date=date
        )
        
        if not bq_success:
            logger.error("BigQuery loading failed")
        
        # Calculate overall success
        pipeline_success = silver_success and gold_success and bq_success
        
        # Log completion
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Pipeline completed in {duration:.2f} seconds with status: {'SUCCESS' if pipeline_success else 'FAILURE'}")
        
        return pipeline_success
        
    except Exception as e:
        # Catch any unhandled exceptions
        logger.error(f"Unhandled error in pipeline: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def send_notification(success: bool, date: str, duration_secs: float) -> None:
    """
    Send a notification about pipeline completion.
    This is a simple placeholder - you can replace with email or SMS notifications.
    
    Args:
        success: Whether the pipeline succeeded
        date: The date processed
        duration_secs: Pipeline duration in seconds
    """
    status = "SUCCESS" if success else "FAILURE"
    message = f"Finance data pipeline for {date} completed with status {status} in {duration_secs:.2f} seconds."
    
    # Simple console notification for now
    logger.info(f"NOTIFICATION: {message}")
    
    # TODO: Add real notification logic here (email, Slack, etc.)

if __name__ == "__main__":
    # Allow date to be passed as command line argument
    date_arg = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Track start time for duration calculation
    start_time = datetime.now()
    
    # Run the pipeline
    success = run_pipeline(date=date_arg)
    
    # Calculate duration
    duration_secs = (datetime.now() - start_time).total_seconds()
    
    # Send notification
    if date_arg:
        send_notification(success, date_arg, duration_secs)
    else:
        today = datetime.now().strftime("%Y-%m-%d")
        send_notification(success, today, duration_secs)
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1) 