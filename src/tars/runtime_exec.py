from __future__ import annotations

import argparse
import os

from .model_integrity import SHA256_RE, open_model_artifact


def _expected_digest(value, *, label):
    text = str(value)
    if not SHA256_RE.fullmatch(text):
        raise ValueError(f"{label} requires a valid SHA-256 identity")
    return text.casefold()


def _parser():
    parser = argparse.ArgumentParser(
        description="Execute llama-server against exact hash-verified file descriptors")
    parser.add_argument("--server", required=True)
    parser.add_argument("--server-sha256", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    server_digest = _expected_digest(
        args.server_sha256, label="llama-server")
    model_digest = _expected_digest(args.model_sha256, label="model")
    trailing = list(args.arguments)
    if trailing[:1] == ["--"]:
        trailing = trailing[1:]

    with open_model_artifact(args.server) as (server, server_handle):
        if server.sha256 != server_digest:
            raise RuntimeError(
                "llama-server bytes no longer match the calibrated SHA-256 identity")
        with open_model_artifact(args.model) as (model, model_handle):
            if model.sha256 != model_digest:
                raise RuntimeError(
                    "model bytes no longer match the verified SHA-256 identity")
            for handle in (server_handle, model_handle):
                os.set_inheritable(handle.fileno(), True)
            server_ref = f"/proc/self/fd/{server_handle.fileno()}"
            model_ref = f"/proc/self/fd/{model_handle.fileno()}"
            os.execve(
                server_ref,
                [server_ref, "-m", model_ref, *trailing],
                dict(os.environ),
            )


if __name__ == "__main__":
    main()
