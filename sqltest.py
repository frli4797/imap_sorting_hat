import json
from sqlalchemy import DateTime, String, create_engine, Column, Integer, JSON, select
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import Session

Base = declarative_base()


class Message(Base):
    __tablename__ = "email"
    id = Column(Integer, primary_key=True)
    from_ = Column(String)
    to = Column(String)
    message_id = Column(String)
    date = Column(DateTime)
    subject = Column(String)
    payload = Column(String)
    _vector = Column(JSON, name="vector")

    @property
    def vector(self) -> list:
        return list(json.loads(self._vector))

    @vector.setter
    def vector(self, values: list):
        self._vector = json.dumps(values)


engine = create_engine("sqlite:///your-database.db")
Base.metadata.create_all(engine)
engine.connect()
jsonvec = json.dumps([22.2, 11.1])
test_mess = Message(
    from_="from@from.se", to="to@to.se", subject="A subject", vector=[22.2, 11.1]
)

# session = sessionmaker(autoflush=False, autocommit=False, bind=engine)

# db = session()

# db.add(test_mess)
# db.commit()
stmt = select(Message)
with Session(engine) as session:
    result = session.execute(select(Message).order_by(Message.id))
    # print(result.all())
    # db.commit()

    for mess in result:
        # vec = mess.vector
        # print(mess.id, vec)
        print(mess.Message.id, mess.Message.vector)
