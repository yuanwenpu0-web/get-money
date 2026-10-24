from cryptography.fernet import Fernet, InvalidToken

from .config import FERNET_KEY_FILE


def _get_fernet() -> Fernet:
    if not FERNET_KEY_FILE.exists():
        FERNET_KEY_FILE.write_bytes(Fernet.generate_key())
    return Fernet(FERNET_KEY_FILE.read_bytes().strip())


def encrypt(value: str) -> str:
    if not value:
        return ""
    return _get_fernet().encrypt(value.encode("utf-8")).decode("utf-8")


def decrypt(value: str) -> str:
    if not value:
        return ""
    try:
        return _get_fernet().decrypt(value.encode("utf-8")).decode("utf-8")
    except (InvalidToken, ValueError):
        return ""


def mask(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}{'*' * 8}{value[-4:]}"
