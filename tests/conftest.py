import pytest


@pytest.fixture(autouse=True)
def no_hugging_face(monkeypatch):
    """Tests do not ask Hugging Face what a name is."""
    monkeypatch.setattr("mteb_gym.llm.repo_files", lambda name: None)
