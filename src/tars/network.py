from __future__ import annotations

from dataclasses import dataclass
import hashlib
import http.client
from ipaddress import ip_address
import socket
import ssl
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


_SENSITIVE_QUERY_KEYS = {
    "authorization", "credential", "credentials", "password", "secret", "token",
    "access_token", "api_key", "apikey", "private_key",
}


def _sensitive_query_key(value: str) -> bool:
    normalized = str(value).casefold().replace("-", "_")
    return normalized in _SENSITIVE_QUERY_KEYS or any(
        marker in normalized
        for marker in ("authorization", "credential", "password", "private_key",
                       "secret", "token", "api_key")
    )


def _format_host(host: str) -> str:
    return f"[{host}]" if ":" in host else host


def _normalized_ip(value: str) -> str:
    return str(ip_address(value.split("%", 1)[0]))


def _classify_addresses(addresses, *, allow_loopback: bool) -> tuple[str, ...]:
    normalized = tuple(sorted({_normalized_ip(value) for value in addresses}))
    if not normalized:
        raise ValueError("network target resolved to no addresses")
    parsed = tuple(ip_address(value) for value in normalized)
    if all(address.is_global for address in parsed):
        return normalized
    if allow_loopback and all(address.is_loopback for address in parsed):
        return normalized
    raise ValueError(
        "network target resolves to a non-public address; private, loopback, "
        "link-local and reserved targets are denied"
    )


@dataclass(frozen=True)
class NetworkDestination:
    request_url: str
    policy_url: str
    scheme: str
    host: str
    port: int
    origin: str
    addresses: tuple[str, ...]
    url_sha256: str

    def same_origin(self, other: "NetworkDestination") -> bool:
        return self.origin == other.origin


def network_destination(value: str, *, resolve_dns: bool = True,
                        allow_loopback: bool = False) -> NetworkDestination:
    raw = str(value)
    parsed = urlsplit(raw if "://" in raw else "https://" + raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("network target must be an HTTP(S) URL or host")
    if parsed.username or parsed.password:
        raise ValueError("credentials in network targets are forbidden")
    try:
        host = parsed.hostname.rstrip(".").encode("idna").decode("ascii").casefold()
        explicit_port = parsed.port
    except (UnicodeError, ValueError) as exc:
        raise ValueError("network target has an invalid host or port") from exc
    if not host:
        raise ValueError("network target requires a host")
    default_port = 443 if parsed.scheme == "https" else 80
    port = explicit_port or default_port
    if not 1 <= port <= 65535:
        raise ValueError("network target has an invalid port")

    literal = None
    try:
        literal = _normalized_ip(host)
    except ValueError:
        pass
    if literal is None:
        try:
            legacy = socket.inet_aton(host)
        except OSError:
            legacy = None
        if legacy is not None:
            literal = _normalized_ip(socket.inet_ntoa(legacy))
    if host == "localhost" or host.endswith(".localhost"):
        literal = "127.0.0.1"

    addresses = ()
    if literal is not None:
        addresses = _classify_addresses((literal,), allow_loopback=allow_loopback)
    elif resolve_dns:
        try:
            answers = socket.getaddrinfo(
                host, port, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP,
            )
        except socket.gaierror as exc:
            raise ValueError(f"network target cannot be resolved: {host}") from exc
        addresses = _classify_addresses(
            (answer[4][0] for answer in answers), allow_loopback=allow_loopback,
        )

    authority = _format_host(host)
    if explicit_port is not None and explicit_port != default_port:
        authority += f":{port}"
    request_url = urlunsplit(
        (parsed.scheme, authority, parsed.path or "/", parsed.query, "")
    )
    safe_query = urlencode([
        (key, "[REDACTED]" if _sensitive_query_key(key) else item)
        for key, item in parse_qsl(parsed.query, keep_blank_values=True)
    ])
    policy_url = urlunsplit(
        (parsed.scheme, authority, parsed.path or "/", safe_query, "")
    )
    origin = f"{parsed.scheme}://{_format_host(host)}:{port}"
    return NetworkDestination(
        request_url=request_url,
        policy_url=policy_url,
        scheme=parsed.scheme,
        host=host,
        port=port,
        origin=origin,
        addresses=addresses,
        url_sha256=hashlib.sha256(request_url.encode("utf-8")).hexdigest(),
    )


def tcp_destination(host: str, port: int, *, scheme="tcp", resolve_dns=True,
                    allow_loopback=False) -> NetworkDestination:
    if not str(scheme).isalpha():
        raise ValueError("network transport scheme is invalid")
    raw_host = str(host)
    if raw_host.startswith("[") and raw_host.endswith("]"):
        raw_host = raw_host[1:-1]
    seed = network_destination(
        f"https://{_format_host(raw_host)}:{int(port)}",
        resolve_dns=resolve_dns, allow_loopback=allow_loopback,
    )
    authority = f"{str(scheme).casefold()}://{_format_host(seed.host)}:{seed.port}"
    url = authority + "/"
    return NetworkDestination(
        request_url=url, policy_url=url, scheme=str(scheme).casefold(),
        host=seed.host, port=seed.port, origin=authority,
        addresses=seed.addresses,
        url_sha256=hashlib.sha256(url.encode("utf-8")).hexdigest(),
    )


def _connect(destination: NetworkDestination, timeout):
    if not destination.addresses:
        raise ValueError("network destination was not resolved before connection")
    last_error = None
    for address in destination.addresses:
        parsed = ip_address(address)
        family = socket.AF_INET6 if parsed.version == 6 else socket.AF_INET
        sock = socket.socket(family, socket.SOCK_STREAM, socket.IPPROTO_TCP)
        sock.settimeout(timeout)
        try:
            endpoint = ((address, destination.port, 0, 0) if parsed.version == 6
                        else (address, destination.port))
            sock.connect(endpoint)
            peer = _normalized_ip(sock.getpeername()[0])
            if peer not in destination.addresses:
                sock.close()
                raise PermissionError("connected peer differs from the authorized destination")
            return sock, peer
        except OSError as exc:
            sock.close()
            last_error = exc
    if last_error is None:
        raise OSError("network destination has no connectable address")
    raise last_error


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, destination, *, timeout):
        super().__init__(destination.host, destination.port, timeout=timeout)
        self.destination = destination
        self.peer_ip = ""

    def connect(self):
        self.sock, self.peer_ip = _connect(self.destination, self.timeout)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, destination, *, timeout, context=None):
        super().__init__(
            destination.host, destination.port, timeout=timeout,
            context=context or ssl.create_default_context(),
        )
        self.destination = destination
        self.peer_ip = ""

    def connect(self):
        raw, peer = _connect(self.destination, self.timeout)
        try:
            self.sock = self._context.wrap_socket(
                raw, server_hostname=self.destination.host,
            )
        except Exception:
            raw.close()
            raise
        actual = _normalized_ip(self.sock.getpeername()[0])
        if actual != peer or actual not in self.destination.addresses:
            self.sock.close()
            raise PermissionError("TLS peer differs from the authorized destination")
        self.peer_ip = actual


class BoundHTTPResponse:
    def __init__(self, connection, response, destination):
        self._connection = connection
        self._response = response
        self.destination = destination
        self.status = response.status
        self.headers = response.headers
        self.peer_ip = connection.peer_ip

    def read(self, amount=None):
        return self._response.read(amount)

    def getcode(self):
        return self.status

    def geturl(self):
        return self.destination.request_url

    def __iter__(self):
        return iter(self._response)

    def close(self):
        try:
            self._response.close()
        finally:
            self._connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def open_bound(destination: NetworkDestination, *, method="GET", headers=None,
               body=None, timeout=30, ssl_context=None) -> BoundHTTPResponse:
    if not isinstance(destination, NetworkDestination) or not destination.addresses:
        raise TypeError("a resolved NetworkDestination is required")
    if destination.scheme not in {"http", "https"}:
        raise ValueError("bound HTTP transport requires an HTTP(S) destination")
    connection_type = (
        _PinnedHTTPSConnection if destination.scheme == "https" else _PinnedHTTPConnection
    )
    options = {"timeout": timeout}
    if connection_type is _PinnedHTTPSConnection:
        options["context"] = ssl_context
    connection = connection_type(destination, **options)
    parsed = urlsplit(destination.request_url)
    target = parsed.path or "/"
    if parsed.query:
        target += "?" + parsed.query
    safe_headers = {
        str(key): str(value) for key, value in dict(headers or {}).items()
        if str(key).casefold() not in {
            "connection", "host", "proxy-authorization", "proxy-connection",
            "transfer-encoding",
        }
    }
    try:
        connection.request(str(method).upper(), target, body=body, headers=safe_headers)
        response = connection.getresponse()
    except Exception:
        connection.close()
        raise
    return BoundHTTPResponse(connection, response, destination)
