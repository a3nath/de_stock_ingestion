# Data Engineering Project 01

This project implements a Medallion Architecture for data processing, starting with financial data ingestion from Yahoo Finance API and storing it in Google Cloud Storage.

## Project Structure

```
.
├── src/
│   └── ingestion/
│       └── finance_ingestor.py
├── tests/
│   └── test_finance_ingestor.py
├── setup_infrastructure.sh
├── pyproject.toml
└── README.md
```

## Prerequisites

- Python 3.9+
- Google Cloud SDK (gcloud)
- Google Cloud Platform account with appropriate permissions
- Poetry (Python package manager)

## Setup Instructions

1. **Install Google Cloud SDK**
   ```bash
   # macOS with Homebrew
   brew install --cask google-cloud-sdk
   
   # Initialize gcloud
   gcloud init
   ```

2. **Initialize Poetry Environment**
   ```bash
   poetry install
   ```

3. **Configure Google Cloud Credentials**
   - Create a service account in Google Cloud Console
   - Download the service account key file
   - Set the environment variable in your `.env` file:
   ```
   GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
   GCS_PROJECT_ID="your-gcp-project-id"
   GCS_BUCKET_NAME="your-gcs-bucket-name"
   STOCK_SYMBOL="AAPL"
   ```

4. **Set Up Infrastructure**
   ```bash
   # Make the setup script executable
   chmod +x setup_infrastructure.sh
   
   # Run the setup script
   ./setup_infrastructure.sh
   ```

5. **Run Tests**
   ```bash
   poetry run pytest
   ```

6. **Run the Finance Ingestor**
   ```bash
   poetry run python -m src.ingestion.finance_ingestor
   ```

## Development

- The project uses Poetry for dependency management
- Google Cloud SDK manages infrastructure
- pytest is used for testing

## Data Flow

1. Raw Layer (Google Cloud Storage):
   - Financial data from Yahoo Finance API
   - Stored as JSON with metadata
   - Versioned storage in the `raw/finance/{symbol}/{date}` path

2. Bronze Layer (Coming in Sprint 2):
   - Will be implemented using data transformation scripts
   - Basic data cleaning and standardization
   - Stored in the `bronze/` folder

3. Silver Layer (Coming in Sprint 2):
   - Will be implemented using data transformation scripts
   - Data quality checks
   - Transformed and cleaned data
   - Stored in the `silver/` folder

4. Gold Layer (Coming in Sprint 2):
   - Business-level aggregations
   - Final data models
   - Stored in the `gold/` folder

## Testing

Run the test suite:
```bash
poetry run pytest
```

## Contributing

1. Create a new branch for your feature
2. Make your changes
3. Run tests
4. Submit a pull request

## License

MIT License 