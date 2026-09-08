from unittest.mock import patch

@patch("src.api.routes.ingest.pipeline.ingest")
def test_ingestion_with_admin_user(mock_ingest, client):
    """Test that ingestion works when authenticated and properly passes allowed_role to the pipeline."""
    # 1. Get token
    auth_response = client.post(
        "/auth/token",
        data={"username": "admin_user", "password": "password123"}
    )
    token = auth_response.json()["access_token"]
    
    # 2. Mock pipeline.ingest
    from src.services.rag.models import IngestResult
    mock_ingest.return_value = IngestResult(
        ingested_documents=1,
        total_chunks=10,
        embedding_model="mock-model",
        duration_sec=1.5,
        sources=["dummy.pdf"]
    )
    
    # 3. Make request
    response = client.post(
        "/ingest",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "sources": ["dummy.pdf"],
            "allowed_role": "admin"
        }
    )
    
    assert response.status_code == 200
    
    # 4. Verify the pipeline was called with the correct allowed_role
    mock_ingest.assert_called_once_with(sources=["dummy.pdf"], allowed_role="admin")

@patch("src.api.routes.ingest.pipeline.ingest")
def test_ingestion_with_public_role_default(mock_ingest, client):
    """Test that ingestion defaults to public role if not specified."""
    auth_response = client.post(
        "/auth/token",
        data={"username": "admin_user", "password": "password123"}
    )
    token = auth_response.json()["access_token"]
    
    from src.services.rag.models import IngestResult
    mock_ingest.return_value = IngestResult(
        ingested_documents=1,
        total_chunks=10,
        embedding_model="mock-model",
        duration_sec=1.5,
        sources=["dummy.pdf"]
    )
    
    response = client.post(
        "/ingest",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "sources": ["dummy.pdf"],
            # allowed_role omitted
        }
    )
    
    assert response.status_code == 200
    mock_ingest.assert_called_once_with(sources=["dummy.pdf"], allowed_role="public")
