import pytest

from carepilot.auth import Principal, Role
from carepilot.store import Ticket, TicketStore, seed_demo_users


class FakeClock:
    def __init__(self, now: float = 1_000_000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def store(tmp_path, clock: FakeClock) -> TicketStore:
    return TicketStore(tmp_path / "tickets.sqlite", clock=clock)


@pytest.fixture
def seeded(store: TicketStore) -> TicketStore:
    seed_demo_users(store)
    return store


@pytest.fixture
def c1() -> Principal:
    return Principal("c1", Role.CUSTOMER, frozenset({"S1", "S2"}))


@pytest.fixture
def c2() -> Principal:
    return Principal("c2", Role.CUSTOMER, frozenset({"S3"}))


@pytest.fixture
def a1() -> Principal:
    return Principal("a1", Role.AGENT)


@pytest.fixture
def a2() -> Principal:
    return Principal("a2", Role.AGENT)


@pytest.fixture
def admin() -> Principal:
    return Principal("admin1", Role.ADMIN)


def make_ticket(store: TicketStore, customer: Principal, *, shop_id: str | None = "S1", **kwargs) -> Ticket:
    kwargs.setdefault("subject", "店铺授权过期")
    kwargs.setdefault("thread_id", "thread-1")
    return store.create_ticket(customer, shop_id=shop_id, **kwargs)


def make_open_ticket(store: TicketStore, customer: Principal, agent: Principal, **kwargs) -> Ticket:
    ticket = make_ticket(store, customer, **kwargs)
    store.claim(agent, ticket.ticket_id)
    return store.transition(agent, ticket.ticket_id, "open")
