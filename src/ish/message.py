from dataclasses import dataclass
from hashlib import sha256
from typing import Optional


@dataclass(frozen=True)
class Message:
    uid: Optional[int]
    from_addr: str
    to_addr: Optional[str]
    subject: str
    body: str

    def preview(self, length: int = 100) -> str:
        return self.body[:length]

    def hash(self) -> str:
        return sha256(self.body.encode("utf-8")).hexdigest()[:12]