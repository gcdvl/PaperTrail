import os
import json
import hashlib
import time
from typing import Optional

class LLMClient:
    """Base LLM client interface."""
    def generate(self, system: str, user: str) -> str:
        raise NotImplementedError

def _exponential_backoff(attempt: int, base: float = 1.0, cap: float = 60.0) -> float:
    delay = min(cap, base * (2 ** attempt))
    jitter = delay * 0.1 * (2 * (0.5 - time.time() % 1))
    return max(0.0, delay + jitter)

class GeminiLLMClient(LLMClient):
    def __init__(self, delay_between_calls: float = 13.0):
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.model_name = os.getenv('GEMINI_MODEL')
        if not self.api_key:
            raise EnvironmentError('Missing GEMINI_API_KEY environment variable')
        if not self.model_name:
            raise EnvironmentError('Missing GEMINI_MODEL environment variable')
        self.delay = delay_between_calls
        # Initialize new google-genai client
        from google import genai
        self.client = genai.Client(api_key=self.api_key)
        # model name is stored for later generate calls
        self.cache_dir = os.path.join(os.getcwd(), '.cache', 'llm')
        os.makedirs(self.cache_dir, exist_ok=True)
        self._last_call = 0.0

    def _cache_key(self, system: str, user: str) -> str:
        hasher = hashlib.sha256()
        hasher.update(self.model_name.encode('utf-8'))
        hasher.update(system.encode('utf-8'))
        hasher.update(user.encode('utf-8'))
        return hasher.hexdigest()

    def _load_from_cache(self, key: str) -> Optional[str]:
        path = os.path.join(self.cache_dir, key + '.json')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('response')
        return None

    def _save_to_cache(self, key: str, response: str):
        path = os.path.join(self.cache_dir, key + '.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'response': response}, f)

    def _respect_rate_limit(self):
        if self.delay > 0:
            time.sleep(self.delay)

    def generate(self, system: str, user: str) -> str:
        cache_key = self._cache_key(system, user)
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            return cached

        from google.genai import errors, types
        max_attempts = 6
        for attempt in range(max_attempts):
            try:
                config = types.GenerateContentConfig(system_instruction=system, temperature=0)
                response = self.client.models.generate_content(model=self.model_name, contents=user, config=config)
                text = response.text
                self._save_to_cache(cache_key, text)
                if self.delay > 0:
                    time.sleep(self.delay)
                return text
            except (errors.ClientError, errors.ServerError, errors.APIError) as e:
                code = getattr(e, 'code', None)
                if code in (429, 503):
                    if attempt >= max_attempts - 1:
                        raise RuntimeError(f'Exceeded max retry attempts ({max_attempts}) due to rate limiting') from e
                    if code == 429:
                        delay = max(15.0, _exponential_backoff(attempt, base=15.0, cap=60.0))
                    else:
                        delay = _exponential_backoff(attempt, base=1.0, cap=60.0)
                    time.sleep(delay)
                else:
                    raise

class StubLLMClient(LLMClient):
    def __init__(self, response: str):
        self.response = response
    def generate(self, system: str, user: str) -> str:
        return self.response
