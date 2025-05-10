import pytest
from datetime import datetime
import pandas as pd
from unittest.mock import Mock, patch
from google.cloud import storage
import numpy as np

from src.ingestion.gold_processor import GoldProcessor

@pytest.fixture
def mock_storage():
    """Create mock storage components."""
    mock = Mock()
    return {
        'client': mock,
        'bucket': mock.bucket(),
        'blob': mock.bucket().blob()
    }

@pytest.fixture
def gold_processor(mock_storage):
    """Create a test instance of GoldProcessor."""
    processor = GoldProcessor(
        bucket_name='test-bucket',
        silver_prefix='silver',
        gold_prefix='gold',
        storage_client=mock_storage['client']
    )
    processor.bucket = mock_storage['bucket']
    return processor

@pytest.fixture
def sample_silver_data():
    """Create sample silver layer data for testing."""
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
    return pd.DataFrame({
        'date': dates,
        'open_price': [100.0 + i for i in range(10)],
        'high_price': [102.0 + i for i in range(10)],
        'low_price': [98.0 + i for i in range(10)],
        'close_price': [101.0 + i for i in range(10)],
        'volume': [1000 + i*100 for i in range(10)],
        'daily_return': [0.01] * 10,
        'price_range': [4.0] * 10,
        'data_source': ['raw_layer'] * 10,
        'data_layer': ['silver'] * 10
    })

def test_process_daily_data_success(gold_processor, mock_storage, sample_silver_data):
    """Test successful processing of daily data."""
    # Mock silver data reading
    gold_processor._read_silver_data = Mock(return_value=sample_silver_data)
    
    # Process data
    result = gold_processor.process_daily_data('2024-01-01')
    
    # Verify result
    assert result is True
    assert gold_processor.metadata['status'] == 'success'

def test_process_daily_data_empty(gold_processor, mock_storage):
    """Test processing with empty data."""
    # Mock empty data
    gold_processor._read_silver_data = Mock(return_value=pd.DataFrame())
    
    # Process data
    result = gold_processor.process_daily_data('2024-01-01')
    
    # Verify result
    assert result is False
    assert gold_processor.metadata['status'] == 'failed'

def test_transform_data(gold_processor, sample_silver_data):
    """Test business transformations."""
    # Transform data
    transformed = gold_processor._transform_data(sample_silver_data)
    
    # Verify transformations
    expected_columns = [
        'date', 'open_price', 'high_price', 'low_price', 'close_price',
        'volume', 'daily_return', 'price_range', 'data_source', 'data_layer',
        'sma_20', 'sma_50', 'volatility', 'price_momentum', 'volume_trend',
        'price_range_pct', 'last_updated'
    ]
    
    for col in expected_columns:
        assert col in transformed.columns
    
    # Verify technical indicators
    assert pd.notnull(transformed['sma_20'].iloc[-1])  # Last row should have SMA
    assert pd.notnull(transformed['volatility'].iloc[-1])  # Last row should have volatility
    
    # Verify business metrics
    assert pd.notnull(transformed['price_momentum'].iloc[-1])  # Last row should have momentum
    assert pd.notnull(transformed['volume_trend'].iloc[-1])  # Last row should have volume trend
    
    # Verify derived features
    assert all(transformed['price_range_pct'] > 0)  # Price range percentage should be positive
    
    # Verify metadata
    assert transformed['data_layer'].iloc[0] == 'gold'
    assert pd.notnull(transformed['last_updated'].iloc[0])

def test_store_gold_data(gold_processor, mock_storage, sample_silver_data):
    """Test gold data storage."""
    # Transform data
    transformed_data = gold_processor._transform_data(sample_silver_data)
    
    # Store data
    gold_processor._store_gold_data(transformed_data, '2024-01-01')
    
    # Verify storage calls
    mock_storage['bucket'].blob.assert_called()
    mock_storage['blob'].upload_from_string.assert_called()
    
    # Verify metadata
    assert mock_storage['blob'].metadata is not None
    assert 'format' in mock_storage['blob'].metadata
    assert mock_storage['blob'].metadata['format'] == 'parquet'

def test_update_metadata(gold_processor, mock_storage, sample_silver_data):
    """Test metadata update."""
    # Transform data
    transformed_data = gold_processor._transform_data(sample_silver_data)
    
    # Update metadata
    gold_processor._update_metadata(transformed_data)
    
    # Verify metadata updates
    assert gold_processor.metadata['total_records'] == len(transformed_data)
    assert 'columns' in gold_processor.metadata
    assert 'last_updated' in gold_processor.metadata
    assert mock_storage['bucket'].blob.called
    assert mock_storage['blob'].upload_from_string.called 