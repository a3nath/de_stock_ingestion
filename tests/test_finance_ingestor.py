import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from datetime import datetime, timedelta
# import boto3  # Commented out AWS SDK

from src.ingestion.finance_ingestor import FinanceIngestor

# Creating a mock GCS client
@pytest.fixture
def mock_storage():
    with patch('google.cloud.storage.Client') as mock_storage:
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        
        # Set up the mock bucket
        mock_storage.return_value.get_bucket.return_value = mock_bucket
        
        # Set up the mock blob
        mock_bucket.blob.return_value = mock_blob
        
        yield mock_storage, mock_bucket, mock_blob

@pytest.fixture
def finance_ingestor(mock_storage):
    _, mock_bucket, _ = mock_storage
    
    # Create a finance ingestor with mock storage
    ingestor = FinanceIngestor(bucket_name="test-bucket", project_id="test-project")
    
    # Replace the real bucket with our mock
    ingestor.bucket = mock_bucket
    
    return ingestor

@pytest.fixture
def sample_data():
    dates = pd.date_range(start='2024-01-01', end='2024-01-02', freq='D')
    return pd.DataFrame({
        'Open': [100.0, 101.0],
        'High': [102.0, 103.0],
        'Low': [99.0, 100.0],
        'Close': [101.0, 102.0],
        'Volume': [1000, 1100],
        'Dividends': [0.0, 0.0],
        'Stock Splits': [0.0, 0.0]
    }, index=dates)

def test_fetch_data(finance_ingestor, sample_data):
    with patch('yfinance.Ticker') as mock_ticker:
        mock_ticker.return_value.history.return_value = sample_data
        
        df = finance_ingestor.fetch_data('AAPL', '2024-01-01', '2024-01-02')
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'Open' in df.columns
        assert 'Close' in df.columns
        
def test_validate_data(finance_ingestor):
    """Test the data validation functionality."""
    # Create a valid DataFrame
    valid_df = pd.DataFrame({
        'Open': [100.0, 101.0],
        'High': [102.0, 103.0],
        'Low': [99.0, 100.0],
        'Close': [101.0, 102.0],
        'Volume': [1000, 1100],
        'Dividends': [0.0, 0.0],
        'Stock Splits': [0.0, 0.0]
    }, index=pd.date_range(start='2024-01-01', end='2024-01-02', freq='D'))
    
    # Test with valid data - should pass
    assert finance_ingestor.validate_data(valid_df, 'AAPL') is True
    
    # Test with empty DataFrame - should raise ValueError
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError) as excinfo:
        finance_ingestor.validate_data(empty_df, 'AAPL')
    assert "DataFrame is empty" in str(excinfo.value)
    
    # Test with missing columns - should raise ValueError
    missing_columns_df = pd.DataFrame({
        'Open': [100.0, 101.0],
        'High': [102.0, 103.0],
        # Missing 'Low'
        'Close': [101.0, 102.0],
        'Volume': [1000, 1100]
        # Missing 'Dividends' and 'Stock Splits'
    }, index=pd.date_range(start='2024-01-01', end='2024-01-02', freq='D'))
    
    with pytest.raises(ValueError) as excinfo:
        finance_ingestor.validate_data(missing_columns_df, 'AAPL')
    assert "Missing required columns" in str(excinfo.value)
    
    # Test with non-numeric data - should raise ValueError
    non_numeric_df = pd.DataFrame({
        'Open': [100.0, 101.0],
        'High': [102.0, 103.0],
        'Low': [99.0, 100.0],
        'Close': ['101.0', '102.0'],  # Strings instead of numbers
        'Volume': [1000, 1100],
        'Dividends': [0.0, 0.0],
        'Stock Splits': [0.0, 0.0]
    }, index=pd.date_range(start='2024-01-01', end='2024-01-02', freq='D'))
    
    with pytest.raises(ValueError) as excinfo:
        finance_ingestor.validate_data(non_numeric_df, 'AAPL')
    assert "is not numeric" in str(excinfo.value)
    
    # Test with missing values - should raise ValueError
    missing_values_df = pd.DataFrame({
        'Open': [100.0, None],  # None creates a missing value
        'High': [102.0, 103.0],
        'Low': [99.0, 100.0],
        'Close': [101.0, 102.0],
        'Volume': [1000, 1100],
        'Dividends': [0.0, 0.0],
        'Stock Splits': [0.0, 0.0]
    }, index=pd.date_range(start='2024-01-01', end='2024-01-02', freq='D'))
    
    with pytest.raises(ValueError) as excinfo:
        finance_ingestor.validate_data(missing_values_df, 'AAPL')
    assert "has 1 missing values" in str(excinfo.value)
   
def test_upload_to_gcs(finance_ingestor, sample_data, mock_storage):
    _, mock_bucket, mock_blob = mock_storage
    
    # Call the upload method
    finance_ingestor.upload_to_gcs(sample_data, 'AAPL')
    
    # Verify blob creation and upload were called
    mock_bucket.blob.assert_called_once()
    mock_blob.upload_from_string.assert_called_once()
    
    # Verify blob path format
    blob_path = mock_bucket.blob.call_args[0][0]
    assert blob_path.startswith('raw/finance/AAPL/')
    assert blob_path.endswith('/data.json')
    
    # Verify metadata was set
    assert mock_blob.metadata is not None
    assert mock_blob.patch.called

def test_process_daily_data(finance_ingestor, sample_data, mock_storage):
    _, mock_bucket, mock_blob = mock_storage
    
    with patch('yfinance.Ticker') as mock_ticker:
        # Mock yfinance
        mock_ticker.return_value.history.return_value = sample_data
        
        # Process data
        finance_ingestor.process_daily_data('AAPL')
        
        # Verify both fetch and upload were called
        mock_ticker.return_value.history.assert_called_once()
        mock_blob.upload_from_string.assert_called_once() 