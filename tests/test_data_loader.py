import pytest
from src.data_loader import load_data

def test_load_data_valid():
    # Example test: loading valid data file
    data = load_data('example_data.csv')
    assert data is not None
    assert len(data) > 0

def test_load_data_missing_file():
    # Example test: loading missing file should raise error
    with pytest.raises(FileNotFoundError):
        load_data('missing_file.csv')

# Add more tests for preprocessing functions as needed
