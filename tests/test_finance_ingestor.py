import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from datetime import datetime, timedelta
from jsonschema import ValidationError
import json
# import boto3  # Commented out AWS SDK

from src.ingestion.finance_ingestor import FinanceIngestor

@pytest.fixture
def mock_storage():
    with patch('google.cloud.storage.Client') as mock_client:
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        yield {
            'client': mock_client,
            'bucket': mock_bucket,
            'blob': mock_blob
        }

@pytest.fixture
def finance_ingestor(mock_storage):
    ingestor = FinanceIngestor('test-bucket', 'test-project')
    ingestor.storage_client = mock_storage['client']
    ingestor.bucket = mock_storage['bucket']
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

@pytest.fixture
def valid_stock_data():
    return {
        "symbol": "AAPL",
        "date": datetime.now().isoformat(),
        "open": 150.0,
        "high": 155.0,
        "low": 149.0,
        "close": 152.0,
        "volume": 1000000,
        "ingestion_date": datetime.now().isoformat()
    }

def test_fetch_data(finance_ingestor, sample_data):
    with patch('yfinance.Ticker') as mock_ticker:
        mock_ticker.return_value.history.return_value = sample_data
        
        df = finance_ingestor.fetch_data('AAPL', '2024-01-01', '2024-01-02')
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'Open' in df.columns
        assert 'Close' in df.columns
        mock_ticker.return_value.history.assert_called_once()

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

def test_validate_json_schema(finance_ingestor):
    valid_data = json.dumps([{
        "Open": 150.0,
        "High": 155.0,
        "Low": 149.0,
        "Close": 152.0,
        "Volume": 1000000,
        "Dividends": 0.0,
        "Stock Splits": 0.0,
        "ingestion_date": datetime.now().isoformat(),
        "source": "yahoo_finance"
    }])
    assert finance_ingestor.validate_json_schema(valid_data, 'AAPL') is True

def test_validate_json_schema_invalid(finance_ingestor):
    invalid_data = json.dumps([{
        "Open": "not-a-number",  # Invalid type
        "High": 155.0,
        "Low": 149.0,
        "Close": 152.0,
        "Volume": 1000000,
        "Dividends": 0.0,
        "Stock Splits": 0.0,
        "ingestion_date": "invalid-date",  # Invalid format
        "source": "yahoo_finance"
    }])
    with pytest.raises(ValidationError):
        finance_ingestor.validate_json_schema(invalid_data, 'AAPL')

def test_upload_to_gcs(finance_ingestor, sample_data, mock_storage):
    # Call the upload method
    blob_path = finance_ingestor.upload_to_gcs(sample_data, 'AAPL')
    
    # Verify blob creation and upload were called
    mock_storage['bucket'].blob.assert_called_once()
    mock_storage['blob'].upload_from_string.assert_called_once()
    
    # Verify blob path format
    assert blob_path.startswith('raw/finance/AAPL/')
    assert blob_path.endswith('/data.json')
    
    # Verify metadata was set
    assert mock_storage['blob'].metadata is not None
    assert mock_storage['blob'].metadata['symbol'] == 'AAPL'
    assert mock_storage['blob'].metadata['schema_version'] == '1.0'

def test_process_daily_data(finance_ingestor, sample_data, mock_storage):
    with patch('yfinance.Ticker') as mock_ticker:
        # Mock yfinance
        mock_ticker.return_value.history.return_value = sample_data
        
        # Process data
        finance_ingestor.process_daily_data('AAPL')
        
        # Verify both fetch and upload were called
        mock_ticker.return_value.history.assert_called_once()
        mock_storage['blob'].upload_from_string.assert_called_once()

def test_upload_to_gcs_with_metadata(finance_ingestor, mock_storage, valid_stock_data):
    """Test that upload includes metadata and validates schema"""
    # Create DataFrame with correct column names
    df = pd.DataFrame([{
        'Open': 150.0,
        'High': 155.0,
        'Low': 149.0,
        'Close': 152.0,
        'Volume': 1000000,
        'Dividends': 0.0,
        'Stock Splits': 0.0,
        'ingestion_date': datetime.now().isoformat(),
        'source': 'yahoo_finance'
    }])
    
    # Call the upload method
    blob_path = finance_ingestor.upload_to_gcs(df, 'AAPL')
    
    # Verify blob creation and upload were called
    mock_storage['bucket'].blob.assert_called_once()
    mock_storage['blob'].upload_from_string.assert_called_once()
    
    # Verify metadata was set
    assert mock_storage['blob'].metadata is not None
    assert mock_storage['blob'].metadata['symbol'] == 'AAPL'
    assert mock_storage['blob'].metadata['schema_version'] == '1.0'

def test_process_daily_data_handles_validation(finance_ingestor, mock_storage):
    """Test that process_daily_data validates data before upload"""
    with patch('yfinance.Ticker') as mock_ticker:
        # Mock yfinance
        mock_df = pd.DataFrame({
            'Open': [150.0],
            'High': [155.0],
            'Low': [149.0],
            'Close': [152.0],
            'Volume': [1000000],
            'Dividends': [0.0],
            'Stock Splits': [0.0]
        }, index=[datetime.now()])
        mock_ticker.return_value.history.return_value = mock_df
        
        # Process data
        finance_ingestor.process_daily_data('AAPL')
        
        # Verify both fetch and upload were called
        mock_ticker.return_value.history.assert_called_once()
        mock_storage['blob'].upload_from_string.assert_called_once() 