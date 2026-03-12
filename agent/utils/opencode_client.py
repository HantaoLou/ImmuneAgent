"""
utils.opencode_client
=====================
Thin wrapper around the OpenCode REST API.

Endpoints used:
    POST   /session                         create a new session
    POST   /session/:id/prompt_async        send a prompt (non-blocking)
    GET    /session/:id/message             list all messages in a session
    GET    /session/status                  map of session_id -> status object
    GET    /skill                           list loaded skills
    PATCH  /global/config                   patch the global OpenCode config
    GET    /session                         list all sessions
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

HTTP_TIMEOUT: int = 30
HTTP_CONNECT_TIMEOUT: int = 5


class OpenCodeError(RuntimeError):
    """Raised when an OpenCode API call fails."""


class OpenCodeClient:
    """
    Synchronous HTTP client for the OpenCode REST API.

    Parameters
    ----------
    base_url : str
        Root URL of the OpenCode server (default: http://127.0.0.1:4096).
    timeout : int
        Per-request timeout in seconds.
    connect_timeout : int
        TCP connect timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:4096",
        timeout: int = HTTP_TIMEOUT,
        connect_timeout: int = HTTP_CONNECT_TIMEOUT,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._connect_timeout = connect_timeout
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=connect_timeout),
            follow_redirects=True,
            trust_env=False,  # bypass HTTP_PROXY for localhost connections
        )

    # ------------------------------------------------------------------
    # Internal request helpers (retry once on keep-alive drop)
    # ------------------------------------------------------------------

    def _recreate_client(self) -> None:
        """Close and recreate the httpx client (handles WinError 10053)."""
        try:
            self._client.close()
        except Exception:
            pass
        self._client = httpx.Client(
            timeout=httpx.Timeout(self._timeout, connect=self._connect_timeout),
            follow_redirects=True,
            trust_env=False,  # bypass HTTP_PROXY for localhost connections
        )

    def _get(self, url: str) -> httpx.Response:
        """GET with one automatic retry on ReadError (keep-alive reset)."""
        try:
            return self._client.get(url)
        except httpx.ReadError:
            logger.debug("GET ReadError, recreating client and retrying: %s", url)
            self._recreate_client()
            return self._client.get(url)

    def _post(self, url: str, **kwargs: Any) -> httpx.Response:
        """POST with one automatic retry on ReadError (keep-alive reset)."""
        try:
            return self._client.post(url, **kwargs)
        except httpx.ReadError:
            logger.debug("POST ReadError, recreating client and retrying: %s", url)
            self._recreate_client()
            return self._client.post(url, **kwargs)

    def _patch(self, url: str, **kwargs: Any) -> httpx.Response:
        """PATCH with one automatic retry on ReadError (keep-alive reset)."""
        try:
            return self._client.patch(url, **kwargs)
        except httpx.ReadError:
            logger.debug("PATCH ReadError, recreating client and retrying: %s", url)
            self._recreate_client()
            return self._client.patch(url, **kwargs)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def check_health(self) -> bool:
        """Return True if the OpenCode server is reachable."""
        try:
            resp = self._get(f"{self._base}/session/status")
            return resp.status_code == 200
        except httpx.RequestError:
            return False

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def create_session(self) -> str:
        """
        Create a new primary agent session.

        Returns
        -------
        str
            The new session ID.
        """
        resp = self._post(f"{self._base}/session", json={})
        self._raise_for_status(resp, "create_session")
        data = resp.json()
        session_id: str = data["id"]
        logger.debug("Created session: %s", session_id)
        return session_id

    def list_sessions(self) -> list[dict]:
        """Return a list of all session objects."""
        resp = self._get(f"{self._base}/session")
        self._raise_for_status(resp, "list_sessions")
        data = resp.json()
        if isinstance(data, dict):
            return data.get("value", [])
        return data if isinstance(data, list) else []

    def send_prompt(self, session_id: str, text: str) -> None:
        """
        Send a user prompt to an existing session (async, non-blocking).

        The prompt is sent using the ``parts`` array format required by
        OpenCode's ``/session/:id/prompt_async`` endpoint.

        Parameters
        ----------
        session_id : str
            Target session ID.
        text : str
            Plain-text task description.
        """
        payload = {"parts": [{"type": "text", "text": text}]}
        resp = self._post(
            f"{self._base}/session/{session_id}/prompt_async",
            json=payload,
        )
        self._raise_for_status(resp, "send_prompt")
        logger.debug("Prompt sent to session %s", session_id)

    def get_messages(self, session_id: str) -> list[dict]:
        """
        Fetch all messages for a session.

        Returns
        -------
        list[dict]
            List of message objects, each containing an ``info`` dict and a
            ``parts`` list.
        """
        resp = self._get(f"{self._base}/session/{session_id}/message")
        self._raise_for_status(resp, "get_messages")
        data = resp.json()
        if isinstance(data, dict):
            return data.get("value", [])
        return data if isinstance(data, list) else []

    def get_session_status(self) -> dict[str, Any]:
        """
        Return the status map for all active sessions.

        Returns
        -------
        dict
            Keys are session IDs; values are status objects.  A session whose
            value has ``type == "busy"`` is still processing.
        """
        resp = self._get(f"{self._base}/session/status")
        self._raise_for_status(resp, "get_session_status")
        return resp.json()

    # ------------------------------------------------------------------
    # Skills
    # ------------------------------------------------------------------

    def list_skills(self) -> list[str]:
        """Return the names of all skills currently loaded in OpenCode."""
        try:
            resp = self._get(f"{self._base}/skill")
            if resp.status_code != 200:
                return []
            data = resp.json()
            if isinstance(data, dict):
                items = data.get("value", [])
            else:
                items = data if isinstance(data, list) else []
            return [s.get("name", "") for s in items if isinstance(s, dict)]
        except httpx.RequestError:
            return []

    # ------------------------------------------------------------------
    # Config patching (runtime injection, survives until server restart)
    # ------------------------------------------------------------------

    def patch_config(self, patch: dict) -> None:
        """
        Apply a partial configuration patch to the running OpenCode server.

        The patch is merged into the global config.  A server restart is NOT
        required; changes take effect immediately.

        Parameters
        ----------
        patch : dict
            Partial config dict, e.g. ``{"skills": {"paths": ["/abs/path"]}}``.
        """
        resp = self._patch(f"{self._base}/global/config", json=patch)
        self._raise_for_status(resp, "patch_config")
        logger.debug("Config patched: %s", list(patch.keys()))

    def inject_skills_path(self, skills_path: str) -> None:
        """
        Register a skills directory with OpenCode via config PATCH.

        Parameters
        ----------
        skills_path : str
            Absolute path to the directory containing skill sub-directories.
        """
        self.patch_config({"skills": {"paths": [skills_path]}})
        logger.info("Skills path injected: %s", skills_path)

    def inject_mcp_servers(self, mcp_cfg: dict) -> None:
        """
        Register MCP server definitions via config PATCH.

        Parameters
        ----------
        mcp_cfg : dict
            Dict matching OpenCode's ``mcp`` config schema, e.g.::

                {
                    "igblast": {"type": "remote", "url": "http://..."},
                }
        """
        self.patch_config({"mcp": mcp_cfg})
        logger.info("MCP servers injected: %s", list(mcp_cfg.keys()))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _raise_for_status(self, resp: httpx.Response, operation: str) -> None:
        if resp.status_code >= 400:
            raise OpenCodeError(
                f"OpenCode API error [{operation}]: "
                f"HTTP {resp.status_code} - {resp.text[:300]}"
            )

    def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        self._client.close()

    def __enter__(self) -> "OpenCodeClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
