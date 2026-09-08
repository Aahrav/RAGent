import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import uuid
import os
from typing import Generator

# Disable OpenTelemetry during tests to prevent connection errors
os.environ["OTEL_SDK_DISABLED"] = "true"

from src.main import app
from src.storage.db.database import get_db, Base
from src.storage.db.models import User, Document, DocumentAccess
from src.api.auth import get_password_hash

# Use an in-memory SQLite database for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="session", autouse=True)
def setup_database() -> Generator:
    # Create all tables in the test database
    Base.metadata.create_all(bind=engine)
    
    # Populate initial seed data for tests
    db = TestingSessionLocal()
    
    # 1. Create a Public User
    public_user = User(
        username="public_user",
        hashed_password=get_password_hash("password123"),
        role="public"
    )
    # 2. Create an Admin User
    admin_user = User(
        username="admin_user",
        hashed_password=get_password_hash("password123"),
        role="admin"
    )
    
    db.add(public_user)
    db.add(admin_user)
    db.commit()

    yield  # Run tests

    # Tear down database after tests
    Base.metadata.drop_all(bind=engine)
    import os
    if os.path.exists("./test.db"):
        os.remove("./test.db")

@pytest.fixture
def db_session() -> Generator:
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

@pytest.fixture
def client(db_session) -> Generator:
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
