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
def bigquery_loader(mock_storage, mock_bigquery):
    """Create a BigQueryLoader instance with mock clients."""
    return BigQueryLoader(
        project_id='test-project',
        dataset_id='test_dataset',
        table_id='stock_data',
        bucket_name='test-bucket',
        gold_prefix='gold',
        storage_client=mock_storage['client'],
        bigquery_client=mock_bigquery
    )

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

def test_load_daily_data_with_validation(bigquery_loader, mock_bigquery, sample_data):
    """Test load_daily_data with validation."""
    with patch('pandas.read_parquet', return_value=sample_data):
        # Mock BigQuery table
        mock_table = Mock()
        mock_bigquery.get_table.return_value = mock_table
        # Mock load job
        mock_job = Mock()
        mock_job.output_rows = len(sample_data)
        mock_job.result.return_value = None
        mock_bigquery.load_table_from_uri.return_value = mock_job
        # Patch loader to use mock_bigquery
        bigquery_loader.bigquery_client = mock_bigquery
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

def test_update_metadata(bigquery_loader, mock_storage):
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
    # Patch loader to use mock_storage
    bigquery_loader.storage_client = mock_storage['client']
    bigquery_loader.bucket = mock_storage['client'].bucket()
    # Update metadata
    bigquery_loader._update_metadata(mock_job, validation_results)
    # Verify metadata updates
    assert bigquery_loader.metadata['total_records_loaded'] == 100
    assert bigquery_loader.metadata['quality_metrics']['current']['quality_score'] == 95.0
    assert len(bigquery_loader.metadata['quality_metrics']['history']) == 1
    bucket = bigquery_loader.storage_client.bucket()
    bucket.blob.assert_called()
    bucket.blob().upload_from_string.assert_called()

@pytest.fixture
def mock_storage():
    """Create mock storage components."""
    mock_client = Mock(spec=storage.Client)
    mock_bucket = Mock(spec=storage.Bucket)
    mock_blob = Mock(spec=storage.Blob)
    
    # Configure mock blob to return a dummy parquet file path
    mock_blob.name = "silver/2024/01/01/data.parquet"
    mock_blob.download_as_bytes.return_value = b"dummy parquet data"
    
    # Configure mock bucket to return our mock blob
    mock_bucket.blob.return_value = mock_blob
    mock_bucket.get_blob.return_value = mock_blob
    
    # Configure mock client to return our mock bucket
    mock_client.bucket.return_value = mock_bucket
    
    return {
        'client': mock_client,
        'bucket': mock_bucket,
        'blob': mock_blob
    }

@pytest.fixture
def mock_bigquery():
    """Create mock BigQuery components."""
    mock_client = Mock(spec=bigquery.Client)
    mock_job = Mock(spec=bigquery.LoadJob)
    mock_job.result.return_value = None
    mock_client.load_table_from_file.return_value = mock_job
    return mock_client

@patch('pandas.read_parquet')
def test_load_daily_data_success(mock_read_parquet, bigquery_loader, sample_data, mock_storage):
    """Test successful loading of daily data into BigQuery."""
    mock_read_parquet.return_value = sample_data
    # Patch load_table_from_uri to return a mock job with output_rows as int
    mock_job = Mock()
    mock_job.output_rows = len(sample_data)
    mock_job.result.return_value = None
    bigquery_loader.bigquery_client.load_table_from_uri.return_value = mock_job
    # Test loading data
    date = "2024-01-01"
    result = bigquery_loader.load_daily_data(date)
    assert result is True
    assert bigquery_loader.metadata['status'] == 'success'
    # Verify BigQuery interactions
    bigquery_loader.bigquery_client.load_table_from_uri.assert_called_once()
    load_job_config = bigquery_loader.bigquery_client.load_table_from_uri.call_args[1]['job_config']
    assert load_job_config.write_disposition == bigquery.WriteDisposition.WRITE_APPEND
    # Verify metadata was stored
    bucket = bigquery_loader.storage_client.bucket()
    bucket.blob.assert_called()
    bucket.blob().upload_from_string.assert_called()

@patch('pandas.read_parquet')
def test_load_daily_data_create_table(mock_read_parquet, bigquery_loader, sample_data, mock_storage, mock_bigquery):
    """Test loading data with table creation."""
    mock_read_parquet.return_value = sample_data
    mock_bigquery.get_table.side_effect = Exception('Not found')
    # Patch load_table_from_uri to return a mock job with output_rows as int
    mock_job = Mock()
    mock_job.output_rows = len(sample_data)
    mock_job.result.return_value = None
    bigquery_loader.bigquery_client.load_table_from_uri.return_value = mock_job
    # Patch create_table to return a real Table object
    from google.cloud import bigquery as bq
    dataset_id = 'test_dataset'
    table_id = 'stock_data'
    # Patch dataset to return a real DatasetReference
    mock_bigquery.dataset.return_value = bq.DatasetReference('test-project', dataset_id)
    table_ref = bq.DatasetReference('test-project', dataset_id).table(table_id)
    schema = [bq.SchemaField('date', 'DATE')]
    real_table = bq.Table(table_ref, schema=schema)
    mock_bigquery.create_table.return_value = real_table
    # Test loading data
    date = "2024-01-01"
    result = bigquery_loader.load_daily_data(date)
    assert result is True
    assert bigquery_loader.metadata['status'] == 'success'
    # Verify table creation was attempted
    mock_bigquery.create_table.assert_called_once()
    table = mock_bigquery.create_table.call_args[0][0]
    assert table.dataset_id == dataset_id
    assert table.table_id == table_id
    # Verify data was loaded
    mock_bigquery.load_table_from_uri.assert_called_once()
    load_job_config = mock_bigquery.load_table_from_uri.call_args[1]['job_config']
    assert load_job_config.write_disposition == bq.WriteDisposition.WRITE_APPEND
    # Verify metadata was stored
    bucket = bigquery_loader.storage_client.bucket()
    bucket.blob.assert_called()
    bucket.blob().upload_from_string.assert_called()

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

@patch('pandas.read_parquet')
def test_load_incremental_data_success(mock_read_parquet, bigquery_loader, sample_data, mock_storage):
    """Test successful incremental loading of data."""
    # Set up mocks
    mock_read_parquet.return_value = sample_data
    mock_job = Mock()
    mock_job.output_rows = len(sample_data)
    mock_job.result.return_value = None
    bigquery_loader.bigquery_client.load_table_from_uri.return_value = mock_job
    
    # Mock _get_last_processed_date to return a specific date
    with patch.object(bigquery_loader, '_get_last_processed_date', return_value='2024-01-01'):
        # Mock filter_trading_days to return specific dates
        with patch.object(bigquery_loader, '_filter_trading_days', return_value=['2024-01-02', '2024-01-03']):
            # Call load_incremental_data with specific dates
            result = bigquery_loader.load_incremental_data('2024-01-03')
            
            # Verify result
            assert result is True
            # Verify load_daily_data was called for each trading day
            assert bigquery_loader.bigquery_client.load_table_from_uri.call_count == 2

@patch('pandas.read_parquet')
def test_load_incremental_data_partial_success(mock_read_parquet, bigquery_loader, sample_data, mock_storage):
    """Test incremental loading with some failures."""
    # Set up mocks for success and failure cases
    mock_read_parquet.return_value = sample_data
    mock_job = Mock()
    mock_job.output_rows = len(sample_data)
    mock_job.result.return_value = None
    
    # Make the first call succeed and the second fail
    bigquery_loader.bigquery_client.load_table_from_uri.side_effect = [
        mock_job,  # First call succeeds
        Exception("Load failed")  # Second call fails
    ]
    
    # Mock _get_last_processed_date
    with patch.object(bigquery_loader, '_get_last_processed_date', return_value='2024-01-01'):
        # Mock filter_trading_days
        with patch.object(bigquery_loader, '_filter_trading_days', return_value=['2024-01-02', '2024-01-03']):
            # Call load_incremental_data
            result = bigquery_loader.load_incremental_data('2024-01-03')
            
            # Verify partial success (overall result is False due to one failure)
            assert result is False
            # Verify both dates were attempted
            assert bigquery_loader.bigquery_client.load_table_from_uri.call_count == 2

def test_get_last_processed_date_from_metadata(bigquery_loader):
    """Test getting last processed date from metadata."""
    # Set metadata with last_loaded_date
    bigquery_loader.metadata['last_loaded_date'] = '2024-01-01'
    
    # Get last processed date
    result = bigquery_loader._get_last_processed_date()
    
    # Verify result
    assert result == '2024-01-01'

def test_get_last_processed_date_from_query(bigquery_loader):
    """Test getting last processed date from BigQuery query."""
    # Clear metadata
    bigquery_loader.metadata['last_loaded_date'] = None
    
    # Mock query result
    mock_result = Mock()
    mock_row = Mock()
    mock_row.last_date = datetime(2024, 1, 1)
    mock_result.__iter__ = lambda x: iter([mock_row])
    
    mock_query_job = Mock()
    mock_query_job.result.return_value = mock_result
    
    bigquery_loader.bigquery_client.query.return_value = mock_query_job
    
    # Get last processed date
    result = bigquery_loader._get_last_processed_date()
    
    # Verify result
    assert result == '2024-01-01'
    bigquery_loader.bigquery_client.query.assert_called_once()

def test_generate_date_range(bigquery_loader):
    """Test generating a date range."""
    # Generate date range
    result = bigquery_loader._generate_date_range('2024-01-01', '2024-01-05')
    
    # Verify result
    assert result == ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']
    assert len(result) == 5

def test_filter_trading_days(bigquery_loader):
    """Test filtering out weekends and holidays."""
    # Test with a known date range from 2023
    # December 22-26, 2023: includes a weekend (23-24) and Christmas (25)
    test_dates = ['2023-12-22', '2023-12-23', '2023-12-24', '2023-12-25', '2023-12-26']
    
    # Call the method
    result = bigquery_loader._filter_trading_days(test_dates)
    
    # Only Dec 22 and Dec 26 should be trading days
    assert len(result) == 2
    assert '2023-12-22' in result  # Friday - trading day
    assert '2023-12-23' not in result  # Saturday - weekend
    assert '2023-12-24' not in result  # Sunday - weekend
    assert '2023-12-25' not in result  # Christmas - holiday
    assert '2023-12-26' in result  # Tuesday - trading day

def test_get_us_market_holidays(bigquery_loader):
    """Test holiday generation for a specific year."""
    # Test for 2023
    holidays_2023 = bigquery_loader._get_us_market_holidays(2023)
    
    # Check a few key holidays
    assert '2023-01-02' in holidays_2023  # New Year's Day (observed on Monday)
    assert '2023-07-04' in holidays_2023  # Independence Day
    assert '2023-12-25' in holidays_2023  # Christmas
    
    # Make sure weekends are not in the holidays list
    weekend_dates = ['2023-01-07', '2023-01-08']  # A Saturday and Sunday
    for date in weekend_dates:
        assert date not in holidays_2023
    
    # Test that the method handles different years correctly
    holidays_2024 = bigquery_loader._get_us_market_holidays(2024)
    assert '2024-01-01' in holidays_2024  # New Year's Day
    assert '2024-07-04' in holidays_2024  # Independence Day
    assert '2024-12-25' in holidays_2024  # Christmas

def test_weekend_holiday_observances(bigquery_loader):
    """Test specific cases where holidays fall on weekends."""
    # In 2022, January 1st (New Year's Day) was a Saturday
    holidays_2022 = bigquery_loader._get_us_market_holidays(2022)
    assert '2021-12-31' in holidays_2022  # New Year's observed on Friday before
    
    # In 2022, December 25th (Christmas) was a Sunday
    assert '2022-12-26' in holidays_2022  # Christmas observed on Monday after
    
    # In 2021, July 4th (Independence Day) was a Sunday
    holidays_2021 = bigquery_loader._get_us_market_holidays(2021)
    assert '2021-07-05' in holidays_2021  # Independence Day observed on Monday after
    
    # In 2020, July 4th (Independence Day) was a Saturday
    holidays_2020 = bigquery_loader._get_us_market_holidays(2020)
    assert '2020-07-03' in holidays_2020  # Independence Day observed on Friday before 