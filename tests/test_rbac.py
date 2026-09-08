def test_authentication_success(client):
    """Test that valid credentials return a JWT token."""
    response = client.post(
        "/auth/token",
        data={"username": "public_user", "password": "password123"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

def test_authentication_failure(client):
    """Test that invalid credentials are rejected."""
    response = client.post(
        "/auth/token",
        data={"username": "public_user", "password": "wrongpassword"}
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect username or password"

def test_chat_requires_authentication(client):
    """Test that the /chat endpoint requires a valid JWT token."""
    response = client.post(
        "/chat",
        json={"query": "What is the capital of France?"}
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated"

def test_chat_with_public_user(client, db_session, mocker):
    """Test that a public user can access the chat endpoint, but the pipeline is only given public documents."""
    # 1. Get token
    auth_response = client.post(
        "/auth/token",
        data={"username": "public_user", "password": "password123"}
    )
    token = auth_response.json()["access_token"]
    
    # 2. Mock the pipeline.query to avoid needing a live Qdrant instance for this unit test
    # but still verify that RBAC logic in the route passes the correct allowed_doc_ids.
    mock_query = mocker.patch("src.api.routes.chat.pipeline.query")
    from src.services.rag.models import QueryResult
    mock_query.return_value = QueryResult(
        answer="Mocked answer",
        citations=[],
        latency_ms=100.0,
        confidence=1.0,
    )
    
    # 3. Create a dummy public document in the DB
    from src.storage.db.models import Document, DocumentAccess
    import uuid
    doc_id = uuid.uuid4()
    db_session.add(Document(id=doc_id, filename="public.txt", raw_text="public info"))
    db_session.add(DocumentAccess(document_id=doc_id, allowed_role="public"))
    db_session.commit()
    
    # 4. Make request
    response = client.post(
        "/chat",
        headers={"Authorization": f"Bearer {token}"},
        json={"query": "Test query"}
    )
    
    assert response.status_code == 200
    
    # 5. Verify the RBAC logic properly extracted the allowed_doc_ids
    mock_query.assert_called_once()
    kwargs = mock_query.call_args.kwargs
    assert str(doc_id) in kwargs["allowed_doc_ids"]

def test_chat_with_admin_user(client, db_session, mocker):
    """Test that an admin user can access the chat endpoint and gets admin document IDs."""
    auth_response = client.post(
        "/auth/token",
        data={"username": "admin_user", "password": "password123"}
    )
    token = auth_response.json()["access_token"]
    
    mock_query = mocker.patch("src.api.routes.chat.pipeline.query")
    from src.services.rag.models import QueryResult
    mock_query.return_value = QueryResult(
        answer="Mocked answer",
        citations=[],
        latency_ms=100.0,
        confidence=1.0,
    )
    
    from src.storage.db.models import Document, DocumentAccess
    import uuid
    admin_doc_id = uuid.uuid4()
    db_session.add(Document(id=admin_doc_id, filename="admin.txt", raw_text="admin info"))
    db_session.add(DocumentAccess(document_id=admin_doc_id, allowed_role="admin"))
    db_session.commit()
    
    response = client.post(
        "/chat",
        headers={"Authorization": f"Bearer {token}"},
        json={"query": "Test query"}
    )
    
    assert response.status_code == 200
    kwargs = mock_query.call_args.kwargs
    assert str(admin_doc_id) in kwargs["allowed_doc_ids"]
