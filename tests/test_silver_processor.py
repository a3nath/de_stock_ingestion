import pytest
from datetime import datetime
import pandas as pd
from unittest.mock import Mock, patch
from google.cloud import storage
import numpy as np

from src.ingestion.silver_processor import SilverProcessor

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
def silver_processor(mock_storage):
    """Create a test instance of SilverProcessor."""
    processor = SilverProcessor(
        bucket_name='test-bucket',
        raw_prefix='raw',
        silver_prefix='silver',
        storage_client=mock_storage['client']
    )
    processor.bucket = mock_storage['bucket']
    return processor

@pytest.fixture
def sample_raw_data():
    """Create sample raw data for testing."""
    return pd.DataFrame({
        'Date': ['2024-01-01', '2024-01-02'],
        'Open': [100.0, 101.0],
        'High': [102.0, 103.0],
        'Low': [99.0, 100.0],
        'Close': [101.0, 102.0],
        'Volume': [1000, 1100],
        'Dividends': [0.0, 0.0],
        'Stock Splits': [0.0, 0.0]
    })

def test_process_daily_data_success(silver_processor, mock_storage, sample_raw_data):
    """Test successful processing of daily data."""
    # Mock raw data reading and quality check
    silver_processor._read_raw_data = Mock(return_value=sample_raw_data)
    silver_processor._apply_quality_checks = Mock(return_value={'quality_score': 95.0})
    
    # Process data
    result = silver_processor.process_daily_data('2024-01-01')
    
    # Verify result
    assert result is True
    assert silver_processor.metadata['status'] == 'success'

def test_process_daily_data_empty(silver_processor, mock_storage):
    """Test processing with empty data."""
    # Mock empty data
    silver_processor._read_raw_data = Mock(return_value=pd.DataFrame())
    
    # Process data
    result = silver_processor.process_daily_data('2024-01-01')
    
    # Verify result
    assert result is False
    assert silver_processor.metadata['status'] == 'failed'

def test_process_daily_data_low_quality(silver_processor, mock_storage, sample_raw_data):
    """Test processing with low quality data."""
    # Mock raw data with low quality
    sample_raw_data.loc[0, 'High'] = 0  # Invalid high price
    silver_processor._read_raw_data = Mock(return_value=sample_raw_data)
    silver_processor._apply_quality_checks = Mock(return_value={'quality_score': 85.0})
    
    # Process data
    result = silver_processor.process_daily_data('2024-01-01')
    
    # Verify result
    assert result is False
    assert silver_processor.metadata['status'] == 'failed'
    assert silver_processor.metadata['quality_score'] < 90.0

def test_transform_data(silver_processor, sample_raw_data):
    """Test data transformation."""
    # Transform data
    transformed = silver_processor._transform_data(sample_raw_data)
    
    # Verify transformations
    expected_columns = [
        'date', 'open_price', 'high_price', 'low_price', 'close_price',
        'volume', 'dividends', 'stock_splits', 'daily_return', 'price_range',
        'ingestion_date', 'data_source', 'data_layer'
    ]
    
    for col in expected_columns:
        assert col in transformed.columns
    
    assert transformed['data_layer'].iloc[0] == 'silver'
    assert transformed['data_source'].iloc[0] == 'raw_layer'
    assert pd.notnull(transformed['daily_return'].iloc[1])  # Second row should have a return
    assert all(transformed['price_range'] >= 0)  # Price range should be non-negative

def test_store_silver_data(silver_processor, mock_storage, sample_raw_data):
    """Test silver data storage."""
    # Transform data
    transformed_data = silver_processor._transform_data(sample_raw_data)
    
    # Store data
    silver_processor._store_silver_data(transformed_data, '2024-01-01')
    
    # Verify storage calls
    mock_storage['bucket'].blob.assert_called()
    mock_storage['blob'].upload_from_string.assert_called()

def test_update_metadata(silver_processor, mock_storage):
    """Test metadata update."""
    # Create quality results
    quality_results = {
        'quality_score': 95.0,
        'total_records': 100,
        'processing_time': 1.5,
        'quality_threshold_met': True,
        'completeness': {'Open': 100.0},
        'validity': {'high_price_validity': True},
        'uniqueness': {'is_unique': True},
        'consistency': {'date_format': True}
    }
    
    # Update metadata
    silver_processor._update_metadata(quality_results)
    
    # Verify metadata updates
    assert silver_processor.metadata['quality_score'] == 95.0
    assert silver_processor.metadata['total_records'] == 100
    assert silver_processor.metadata['processing_time'] == 1.5
    assert silver_processor.metadata['quality_threshold_met'] is True
    assert isinstance(silver_processor.metadata['quality_results'], dict)
    assert mock_storage['bucket'].blob.called
    assert mock_storage['blob'].upload_from_string.called 