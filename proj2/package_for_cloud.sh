#!/bin/bash

# package_for_cloud.sh
# Purpose: Zips the necessary project files for upload to Vast.ai / Cloud Server.
# Excludes local datasets, caches, and venvs.

echo "Packaging project for Cloud Deployment..."

ZIP_NAME="brainded_deploy.zip"

# clean up old zip
rm -f $ZIP_NAME

# Zip proj2 folder, requirements, and setup scripts
# We exclude __pycache__, .git, and local outputs
zip -r $ZIP_NAME proj2 requirements.txt setup_vast.sh \
    -x "proj2/**/__pycache__/*" \
    -x "proj2/output/*" \
    -x "**/.DS_Store"

echo "Created $ZIP_NAME"
echo "You can now upload this file to your server."
