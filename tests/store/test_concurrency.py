import threading

from carepilot.auth import Principal
from carepilot.store import Conflict, TicketStore

from conftest import make_ticket


def _race_claim(store: TicketStore, ticket_id: str, agents: list[Principal]) -> list[str]:
    barrier = threading.Barrier(len(agents))
    outcomes: list[str] = []
    lock = threading.Lock()

    def worker(agent: Principal) -> None:
        barrier.wait()
        try:
            store.claim(agent, ticket_id)
            result = "ok"
        except Conflict:
            result = "conflict"
        with lock:
            outcomes.append(result)

    threads = [threading.Thread(target=worker, args=(agent,)) for agent in agents]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return outcomes


def test_two_agents_race_to_claim(store: TicketStore, c1: Principal, a1: Principal, a2: Principal):
    for _ in range(50):
        ticket = make_ticket(store, c1)
        outcomes = _race_claim(store, ticket.ticket_id, [a1, a2])
        assert sorted(outcomes) == ["conflict", "ok"]
        assert store.get_ticket(a1, ticket.ticket_id).owner_id in {"a1", "a2"}
