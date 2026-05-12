"""
Pytest configuration: stub AWS SDK modules before any test imports them.
This lets the test suite run without boto3/botocore installed locally.
"""
import sys
import types
from unittest.mock import MagicMock

# Only stub if not already installed
if "boto3" not in sys.modules:
    boto3_stub = MagicMock()
    boto3_stub.client.return_value = MagicMock()
    sys.modules["boto3"] = boto3_stub

if "botocore" not in sys.modules:
    botocore_stub = types.ModuleType("botocore")
    sys.modules["botocore"] = botocore_stub

    config_stub = types.ModuleType("botocore.config")
    config_stub.Config = MagicMock()
    sys.modules["botocore.config"] = config_stub

    exc_stub = types.ModuleType("botocore.exceptions")
    # Keep as real Exception subclasses so `except` clauses work
    class _BotoCoreError(Exception):
        pass
    class _ClientError(Exception):
        pass
    class _ReadTimeoutError(Exception):
        pass
    exc_stub.BotoCoreError = _BotoCoreError
    exc_stub.ClientError = _ClientError
    exc_stub.ReadTimeoutError = _ReadTimeoutError
    sys.modules["botocore.exceptions"] = exc_stub
