import os
from unittest.mock import MagicMock
import pytest
from google.genai.errors import ClientError

from src.llm_client import GeminiLLMClient


class _BadModel:
    """A dummy model that raises a non-retryable error on generate_content."""

    def generate_content(self, *_, **__):  # pragma: no cover
        raise TypeError("non-retryable error for testing")


class BadLLMClient(GeminiLLMClient):
    """GeminiLLMClient with a model that always raises a TypeError."""

    def __init__(self):
        os.environ.setdefault('GEMINI_API_KEY', 'dummy')
        os.environ.setdefault('GEMINI_MODEL', 'gemini-1.0')
        super().__init__(delay_between_calls=0)
        self.client._models = _BadModel()


def test_generate_non_retryable_error():
    """Ensure that a non-retryable exception bubbles up instantly."""
    llm = BadLLMClient()
    with pytest.raises(TypeError) as exc:
        llm.generate("system prompt", "user prompt")
    assert "non-retryable" in str(exc.value)


def test_error_400_not_retried_or_cached(tmp_path, monkeypatch):
    """Ensure HTTP 400 error is raised immediately, not retried, and never cached."""
    os.environ.setdefault('GEMINI_API_KEY', 'dummy')
    os.environ.setdefault('GEMINI_MODEL', 'gemini-1.0')

    llm = GeminiLLMClient(delay_between_calls=0)
    llm.cache_dir = str(tmp_path)

    err_400 = ClientError(400, {"error": {"code": 400, "message": "API key not valid"}})

    mock_models = MagicMock()
    mock_models.generate_content.side_effect = err_400
    llm.client._models = mock_models

    with pytest.raises(ClientError) as exc_info:
        llm.generate("system prompt", "user prompt")

    assert exc_info.value.code == 400
    assert mock_models.generate_content.call_count == 1
    # Verify nothing was saved to cache
    assert len(list(tmp_path.glob("*.json"))) == 0


def test_error_429_retried_and_succeeds(tmp_path, monkeypatch):
    """Ensure HTTP 429 error is retried with backoff >= 15s and succeeds on subsequent try."""
    os.environ.setdefault('GEMINI_API_KEY', 'dummy')
    os.environ.setdefault('GEMINI_MODEL', 'gemini-1.0')

    llm = GeminiLLMClient(delay_between_calls=0)
    llm.cache_dir = str(tmp_path)

    err_429 = ClientError(429, {"error": {"code": 429, "message": "Resource exhausted"}})
    mock_response = MagicMock()
    mock_response.text = '{"answerable": true, "answer": "Success"}'

    mock_models = MagicMock()
    mock_models.generate_content.side_effect = [err_429, mock_response]
    llm.client._models = mock_models

    sleep_calls = []
    monkeypatch.setattr("time.sleep", lambda s: sleep_calls.append(s))

    result = llm.generate("system prompt", "user prompt")

    assert result == '{"answerable": true, "answer": "Success"}'
    assert mock_models.generate_content.call_count == 2
    # Verify retry waited at least 15 seconds on 429
    assert len(sleep_calls) >= 1
    assert sleep_calls[0] >= 15.0
    # Verify successful response was cached
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_cache_hit_skips_delay(tmp_path, monkeypatch):
    """Ensure cache hits return immediately without applying call delay or calling the API."""
    os.environ.setdefault('GEMINI_API_KEY', 'dummy')
    os.environ.setdefault('GEMINI_MODEL', 'gemini-1.0')

    llm = GeminiLLMClient(delay_between_calls=13.0)
    llm.cache_dir = str(tmp_path)

    cache_key = llm._cache_key("system prompt", "user prompt")
    llm._save_to_cache(cache_key, "cached content")

    mock_models = MagicMock()
    llm.client._models = mock_models

    sleep_called = False

    def fake_sleep(_):
        nonlocal sleep_called
        sleep_called = True

    monkeypatch.setattr("time.sleep", fake_sleep)

    result = llm.generate("system prompt", "user prompt")

    assert result == "cached content"
    mock_models.generate_content.assert_not_called()
    assert not sleep_called


def test_real_call_applies_delay(tmp_path, monkeypatch):
    """Ensure real API calls apply the configured delay after succeeding."""
    os.environ.setdefault('GEMINI_API_KEY', 'dummy')
    os.environ.setdefault('GEMINI_MODEL', 'gemini-1.0')

    llm = GeminiLLMClient(delay_between_calls=13.0)
    llm.cache_dir = str(tmp_path)

    mock_response = MagicMock()
    mock_response.text = '{"answer": "fresh"}'
    mock_models = MagicMock()
    mock_models.generate_content.return_value = mock_response
    llm.client._models = mock_models

    sleep_calls = []
    monkeypatch.setattr("time.sleep", lambda s: sleep_calls.append(s))

    result = llm.generate("system fresh", "user fresh")

    assert result == '{"answer": "fresh"}'
    assert mock_models.generate_content.call_count == 1
    assert sleep_calls == [13.0]


