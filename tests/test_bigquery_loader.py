import pytest
from datetime import datetime
import pandas as pd
from unittest.mock import Mock, patch
from google.cloud import bigquery, storage
import numpy as np

from src.ingestion.bigquery_loader import BigQueryLoader

@pytest.fixture
def mock_clients():
    """Create mock GCS and BigQuery clients."""
    mock_storage = Mock()
    mock_bigquery = Mock()
    return {
        'storage': mock_storage,
        'bigquery': mock_bigquery
    }

@pytest.fixture
def bigquery_loader(mock_clients):
    """Create a test instance of BigQueryLoader."""
    loader = BigQueryLoader(
        project_id='test-project',
        dataset_id='test_dataset',
        table_id='test_table',
        bucket_name='test-bucket',
        storage_client=mock_clients['storage'],
        bigquery_client=mock_clients['bigquery']
    )
    loader.bucket = mock_clients['storage'].bucket()
    return loader

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=5),
        'open_price': [100.0, 101.0, 102.0, 103.0, 104.0],
        'high_price': [105.0, 106.0, 107.0, 108.0, 109.0],
        'low_price': [95.0, 96.0, 97.0, 98.0, 99.0],
        'close_price': [102.0, 103.0, 104.0, 105.0, 106.0],
        'volume': [1000, 1100, 1200, 1300, 1400],
        'daily_return': [0.02, 0.01, 0.01, 0.01, 0.01],
        'price_range': [10.0, 10.0, 10.0, 10.0, 10.0],
        'sma_20': [101.0, 102.0, 103.0, 104.0, 105.0],
        'sma_50': [100.0, 101.0, 102.0, 103.0, 104.0],
        'volatility': [0.01, 0.01, 0.01, 0.01, 0.01],
        'price_momentum': [0.02, 0.02, 0.02, 0.02, 0.02],
        'volume_trend': [0.1, 0.1, 0.1, 0.1, 0.1],
        'price_range_pct': [0.1, 0.1, 0.1, 0.1, 0.1]
    })

def test_validate_data_success(bigquery_loader, sample_data):
    """Test successful data validation."""
    results = bigquery_loader._validate_data(sample_data)
    
    assert results["quality_score"] == 100.0
    assert all(component["is_valid"] for component in results.values() 
              if isinstance(component, dict) and "is_valid" in component)

def test_validate_data_missing_columns(bigquery_loader):
    """Test validation with missing columns."""
    data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=5),
        'open_price': [100.0, 101.0, 102.0, 103.0, 104.0]
    })
    
    results = bigquery_loader._validate_data(data)
    
    assert results["quality_score"] < 100.0
    assert not results["schema_validation"]["is_valid"]
    assert len(results["schema_validation"]["missing_columns"]) > 0

def test_validate_data_invalid_types(bigquery_loader):
    """Test validation with invalid data types."""
    data = pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02', '2024-01-03'],  # String instead of date
        'open_price': ['100.0', '101.0', '102.0'],  # String instead of float
        'volume': [1000.5, 1100.5, 1200.5]  # Float instead of integer
    })
    
    results = bigquery_loader._validate_data(data)
    
    assert results["quality_score"] < 100.0
    assert not results["data_type_validation"]["is_valid"]
    assert len(results["data_type_validation"]["invalid_types"]) > 0

def test_validate_data_invalid_ranges(bigquery_loader):
    """Test validation with invalid value ranges."""
    data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=3),
        'open_price': [-100.0, 101.0, 102.0],  # Negative price
        'high_price': [105.0, 106.0, 107.0],
        'low_price': [95.0, 96.0, 97.0],
        'close_price': [102.0, 103.0, 104.0],
        'volume': [-1000, 1100, 1200],  # Negative volume
        'daily_return': [0.02, 1.5, -1.5]  # Returns outside [-1, 1] range
    })
    
    results = bigquery_loader._validate_data(data)
    
    assert results["quality_score"] < 100.0
    assert not results["value_range_validation"]["is_valid"]
    assert len(results["value_range_validation"]["invalid_ranges"]) > 0

def test_validate_data_duplicates(bigquery_loader):
    """Test validation with duplicate records."""
    data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=3).repeat(2),
        'open_price': [100.0, 100.0, 101.0, 101.0, 102.0, 102.0],
        'high_price': [105.0, 105.0, 106.0, 106.0, 107.0, 107.0],
        'low_price': [95.0, 95.0, 96.0, 96.0, 97.0, 97.0],
        'close_price': [102.0, 102.0, 103.0, 103.0, 104.0, 104.0],
        'volume': [1000, 1000, 1100, 1100, 1200, 1200],
        'daily_return': [0.02, 0.02, 0.01, 0.01, 0.01, 0.01]
    })
    
    results = bigquery_loader._validate_data(data)
    
    assert results["quality_score"] < 100.0
    assert not results["duplicate_validation"]["is_valid"]
    assert results["duplicate_validation"]["duplicate_count"] > 0

def test_load_daily_data_with_validation(bigquery_loader, mock_clients, sample_data):
    """Test load_daily_data with validation."""
    # Mock parquet file reading
    with patch('pandas.read_parquet', return_value=sample_data):
        # Mock BigQuery table
        mock_table = Mock()
        mock_clients['bigquery'].get_table.return_value = mock_table
        
        # Mock load job
        mock_job = Mock()
        mock_job.output_rows = len(sample_data)
        mock_clients['bigquery'].load_table_from_uri.return_value = mock_job
        
        # Load data
        result = bigquery_loader.load_daily_data('2024-01-01')
        
        # Verify result
        assert result is True
        assert bigquery_loader.metadata['status'] == 'success'
        assert bigquery_loader.metadata['quality_metrics']['current']['quality_score'] == 100.0

def test_load_daily_data_validation_failure(bigquery_loader, mock_clients):
    """Test load_daily_data with validation failure."""
    # Create invalid data
    invalid_data = pd.DataFrame({
        'date': ['2024-01-01'],  # String instead of date
        'open_price': [-100.0]  # Negative price
    })
    
    # Mock parquet file reading
    with patch('pandas.read_parquet', return_value=invalid_data):
        # Load data
        result = bigquery_loader.load_daily_data('2024-01-01')
        
        # Verify result
        assert result is False
        assert bigquery_loader.metadata['status'] == 'failed'
        assert 'Data quality score below threshold' in bigquery_loader.metadata['errors']

def test_update_metadata_with_validation(bigquery_loader, mock_clients):
    """Test metadata update with validation results."""
    # Create mock job
    mock_job = Mock()
    mock_job.output_rows = 100
    
    # Create validation results
    validation_results = {
        "quality_score": 95.0,
        "schema_validation": {"is_valid": True, "score": 100},
        "data_type_validation": {"is_valid": True, "score": 100},
        "value_range_validation": {"is_valid": True, "score": 100},
        "required_field_validation": {"is_valid": True, "score": 100},
        "duplicate_validation": {"is_valid": False, "score": 75}
    }
    
    # Update metadata
    bigquery_loader._update_metadata(mock_job, validation_results)
    
    # Verify metadata updates
    assert bigquery_loader.metadata['total_records_loaded'] == 100
    assert bigquery_loader.metadata['quality_metrics']['current']['quality_score'] == 95.0
    assert len(bigquery_loader.metadata['quality_metrics']['history']) == 1
    assert mock_clients['storage'].bucket().blob.called
    assert mock_clients['storage'].bucket().blob().upload_from_string.called

def test_load_daily_data_success(bigquery_loader, mock_clients):
    """Test successful loading of daily data."""
    # Mock BigQuery table
    mock_table = Mock()
    mock_clients['bigquery'].get_table.return_value = mock_table
    
    # Mock load job
    mock_job = Mock()
    mock_job.output_rows = 100
    mock_clients['bigquery'].load_table_from_uri.return_value = mock_job
    
    # Load data
    result = bigquery_loader.load_daily_data('2024-01-01')
    
    # Verify result
    assert result is True
    assert bigquery_loader.metadata['status'] == 'success'
    assert bigquery_loader.metadata['total_records_loaded'] == 100

def test_load_daily_data_create_table(bigquery_loader, mock_clients):
    """Test table creation when it doesn't exist."""
    # Mock table not found
    mock_clients['bigquery'].get_table.side_effect = Exception("Table not found")
    
    # Mock table creation
    mock_table = Mock()
    mock_clients['bigquery'].create_table.return_value = mock_table
    
    # Mock load job
    mock_job = Mock()
    mock_job.output_rows = 100
    mock_clients['bigquery'].load_table_from_uri.return_value = mock_job
    
    # Load data
    result = bigquery_loader.load_daily_data('2024-01-01')
    
    # Verify result
    assert result is True
    assert bigquery_loader.metadata['status'] == 'success'
    mock_clients['bigquery'].create_table.assert_called_once()

def test_load_daily_data_failure(bigquery_loader, mock_clients):
    """Test handling of load failure."""
    # Mock load job failure
    mock_clients['bigquery'].load_table_from_uri.side_effect = Exception("Load failed")
    
    # Load data
    result = bigquery_loader.load_daily_data('2024-01-01')
    
    # Verify result
    assert result is False
    assert bigquery_loader.metadata['status'] == 'failed'
    assert len(bigquery_loader.metadata['errors']) > 0

def test_update_metadata(bigquery_loader, mock_clients):
    """Test metadata update."""
    # Create mock job
    mock_job = Mock()
    mock_job.output_rows = 100
    
    # Update metadata
    bigquery_loader._update_metadata(mock_job)
    
    # Verify metadata updates
    assert bigquery_loader.metadata['total_records_loaded'] == 100
    assert 'last_updated' in bigquery_loader.metadata
    assert mock_clients['storage'].bucket().blob.called
    assert mock_clients['storage'].bucket().blob().upload_from_string.called 