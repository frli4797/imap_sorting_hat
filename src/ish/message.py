from dataclasses import dataclass
from hashlib import sha256
import hashlib
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
        # Generate a short hash based on the message body and all the other fields
        data = (self.from_addr or "") + (self.to_addr or "") + (self.subject or "") + (self.body or "")
        return hashlib.blake2b(data.encode("utf-8"), digest_size=8).hexdigest()