# tests/test_backend_api.py
import requests
import pytest

API_BASE_URL = "http://localhost:8080"


def test_balance_endpoint():
    """TC-B-04: Test the /balance endpoint for correctness."""
    # Define the request payload
    payload = {"user_id": "silver_001"}

    # Make the POST request
    response = requests.post(f"{API_BASE_URL}/balance", json=payload)

    # Assert the status code is 200 (OK)
    assert response.status_code == 200

    # Assert the response is valid JSON
    try:
        response_data = response.json()
    except ValueError:
        pytest.fail("Response is not valid JSON")

    # Assert the response has the expected structure
    assert "status" in response_data
    assert response_data["status"] == "success"
    assert "balance" in response_data
    assert "transaction_count" in response_data

    # Assert the data types are correct
    assert isinstance(response_data["balance"], (int, float))
    assert isinstance(response_data["transaction_count"], int)
