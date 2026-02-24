import os
import ssl
import time
from functools import lru_cache
from typing import Any

import certifi
import requests
from loguru import logger

from energbench.utils.constants import HTTP_RETRIES_DEFAULT, HTTP_TIMEOUT_DEFAULT

# Well-known system CA bundle paths by platform.
_SYSTEM_CA_PATHS = (
    "/etc/ssl/certs/ca-certificates.crt",  # Debian / Ubuntu
    "/etc/pki/tls/certs/ca-bundle.crt",  # RHEL / CentOS / Fedora
    "/etc/ssl/ca-bundle.pem",  # openSUSE
    "/etc/pki/tls/cacert.pem",  # OpenELEC
    "/etc/ssl/cert.pem",  # macOS / Alpine
)


@lru_cache(maxsize=1)
def get_system_ca_bundle() -> str:
    """Return the best available CA certificate bundle path.

    Prefers the system CA store (which typically carries CAs that the
    ``certifi`` package may have removed) and falls back to ``certifi``
    if no system bundle is found.
    """
    # Python's compiled-in OpenSSL default comes first.
    openssl_cafile = ssl.get_default_verify_paths().openssl_cafile
    if openssl_cafile and os.path.isfile(openssl_cafile):
        return openssl_cafile

    for path in _SYSTEM_CA_PATHS:
        if os.path.isfile(path):
            return path

    return certifi.where()


class HTTPClient:
    """Simple HTTP client for API requests with common authentication patterns.

    Provides a unified interface for making HTTP requests with different
    authentication methods (header-based, parameter-based).
    """

    def __init__(
        self,
        auth_method: str = "header",
        auth_param_name: str = "x-api-key",
        timeout: int = HTTP_TIMEOUT_DEFAULT,
        retries: int = HTTP_RETRIES_DEFAULT,
    ):
        """Initialize HTTP client.

        Args:
            auth_method: Authentication method ("header" or "param")
            auth_param_name: Name of the header/param for the API key
            timeout: Request timeout in seconds
            retries: Number of retry attempts for failed requests
        """
        self.auth_method = auth_method
        self.auth_param_name = auth_param_name
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()

    def get(
        self,
        url: str,
        api_key: str | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Make a GET request.

        Args:
            url: URL to request
            api_key: API key for authentication
            params: Query parameters
            headers: Additional headers

        Returns:
            Parsed JSON response

        Raises:
            requests.exceptions.RequestException: If request fails
        """

        params = params or {}
        headers = headers or {}

        if api_key:
            if self.auth_method == "header":
                headers[self.auth_param_name] = api_key
            elif self.auth_method == "param":
                params[self.auth_param_name] = api_key

        last_error: requests.exceptions.RequestException | None = None
        for attempt in range(1 + self.retries):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                result: dict[str, Any] = response.json()
                return result
            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < self.retries:
                    delay = 2 ** attempt
                    logger.warning(f"HTTP GET failed (attempt {attempt + 1}/{1 + self.retries}), retrying in {delay}s: {e}")
                    time.sleep(delay)
        logger.error(f"HTTP request failed after {1 + self.retries} attempts: {last_error}")
        raise last_error  # type: ignore[misc]

    def post(
        self,
        url: str,
        api_key: str | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        json_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make a POST request.

        Args:
            url: URL to request
            api_key: API key for authentication
            params: Query parameters
            headers: Additional headers
            json_data: JSON data to send in body

        Returns:
            Parsed JSON response

        Raises:
            requests.exceptions.RequestException: If request fails
        """

        params = params or {}
        headers = headers or {}

        if api_key:
            if self.auth_method == "header":
                headers[self.auth_param_name] = api_key
            elif self.auth_method == "param":
                params[self.auth_param_name] = api_key

        last_error: requests.exceptions.RequestException | None = None
        for attempt in range(1 + self.retries):
            try:
                response = self.session.post(
                    url,
                    params=params,
                    headers=headers,
                    json=json_data,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                result: dict[str, Any] = response.json()
                return result
            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < self.retries:
                    delay = 2 ** attempt
                    logger.warning(f"HTTP POST failed (attempt {attempt + 1}/{1 + self.retries}), retrying in {delay}s: {e}")
                    time.sleep(delay)
        logger.error(f"HTTP request failed after {1 + self.retries} attempts: {last_error}")
        raise last_error  # type: ignore[misc]
