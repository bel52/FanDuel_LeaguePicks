#!/bin/bash
# Create all necessary directories
mkdir -p data/input data/output data/fantasypros data/weekly data/targets data/executed data/season data/bankroll logs

# Set proper permissions
chmod -R 755 data logs
touch logs/.gitkeep

echo "Directory structure created successfully"
