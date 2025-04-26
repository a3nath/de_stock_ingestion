#!/bin/bash

# Exit on error
set -e

echo "Setting up Google Cloud Storage infrastructure..."

# Create bucket
echo "Creating GCS bucket..."
gcloud storage buckets create gs://finance_stock_data \
    --location=northamerica-northeast2 \
    --storage-class=STANDARD

# Enable versioning
echo "Enabling versioning..."
gsutil versioning set on gs://finance_stock_data

# Create folders for medallion architecture
echo "Creating folder structure..."
for folder in raw bronze silver gold; do
    echo "Creating $folder folder..."
    gsutil mkdir gs://finance_stock_data/$folder
done

# Set IAM permissions
echo "Setting IAM permissions..."
gcloud storage buckets add-iam-policy-binding gs://finance_stock_data \
    --member=serviceAccount:finance-data-ingestor@finance-stock-453800.iam.gserviceaccount.com \
    --role=roles/storage.objectAdmin

echo "Infrastructure setup complete!" 