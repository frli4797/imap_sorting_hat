import datetime
import json
from hashlib import sha256

from sqlalchemy import JSON, Column, DateTime, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase

# Base = declarative_base()


class Base(DeclarativeBase):
    pass


class Email(Base):
    __tablename__ = "email"
    id = Column(Integer, primary_key=True, nullable=False)

    # Columns
    from_ = Column(String(255))
    to = Column(String(255), nullable=False)
    message_id = Column(String(255), index=True)
    timestamp = Column(DateTime, default=datetime.datetime.now, nullable=False)
    hash = Column(String(64), nullable=False, index=True)
    subject = Column(String(256))
    body = Column(Text)
    folder = Column(String(255), nullable=False, index=True)
    uid = Column(Integer, index=True)
    vector_json = Column(JSON, name="vector")

    # __table_args__ = (Index("idx_uid_folder", folder.name, uid.name),)

    @property
    def emdeddings(self) -> list:
        return list(json.loads(self.vector_json))

    @emdeddings.setter
    def emdeddings(self, values: list):
        self.vector_json = json.dumps(values)

    def equals(self, mesg) -> bool:
        if mesg.message_id == self.message_id:
            return True
        elif mesg.hash == self.message_id:
            return True
        return False

    def mesg_hash(self) -> str:
        return sha256(self.body.encode("utf-8")).hexdigest()[:12]

    def __eq__(self, other):
        return (
            isinstance(other, Email)
            and other.id == self.id
            and other.hash == self.hash
            and other.message_id == self.message_id
        )
