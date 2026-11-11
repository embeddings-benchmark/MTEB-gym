import httpx2
import pytest


@pytest.fixture(autouse=True)
def no_hugging_face(monkeypatch):
    """Tests do not ask Hugging Face what a name is."""
    monkeypatch.setattr("mteb_gym.llm.repo_files", lambda name: None)
    monkeypatch.setattr("mteb_gym.models.repo_files", lambda name: None)

    def no_server(*args, **kwargs):
        raise httpx2.ConnectError("no server here")

    monkeypatch.setattr(httpx2, "get", no_server)
