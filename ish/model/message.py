import datetime
from hashlib import sha256
import json

from sqlalchemy import JSON, Column, DateTime, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Message(Base):
    __tablename__ = "email"
    id = Column(Integer, primary_key=True)
    from_ = Column(String(255))
    to = Column(String(255), nullable=False)
    message_id = Column(String(255))
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    hash = Column(String(64))
    subject = Column(String(256))
    body = Column(Text)
    _vector = Column(JSON, name="vector")

    @property
    def vector(self) -> list:
        return list(json.loads(self._vector))

    @vector.setter
    def vector(self, values: list):
        self._vector = json.dumps(values)

    def equals(self, mesg) -> bool:
        if mesg.message_id == self.message_id:
            return True
        elif mesg.hash == self.message_id:
            return True
        return False


def mesg_hash(self, mesg: Message) -> str:
    return sha256(mesg.body.encode("utf-8")).hexdigest()[:12]
