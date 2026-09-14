"""A schema file's claims: tables, their columns, and what rests on what -- each quoted from its line."""
from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.entities.affected import affected
from scone_memory.ingestion.code_graph import record_claims
from scone_memory.ingestion.schema_claims import is_schema, schema_claims

pytestmark = pytest.mark.asyncio

SCHEMA = '''-- the shop's tables; a comment naming CREATE TABLE ghosts must not count
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,  -- trailing comment
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS "orders" (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    placed_at TIMESTAMP,
    CHECK (placed_at IS NOT NULL)
);

create table order_items (
    order_id integer,
    sku text,
    quantity numeric(10, 2),
    constraint fk_order foreign key (order_id) references orders (id),
    primary key (order_id, sku)
);

/* a view over the lot */
CREATE VIEW customer_totals AS
  SELECT c.id, sum(i.quantity) AS units
  FROM customers c
  JOIN orders o ON o.customer_id = c.id
  JOIN order_items i ON i.order_id = o.id
  GROUP BY c.id;

ALTER TABLE orders ADD CONSTRAINT fk_customer FOREIGN KEY (customer_id) REFERENCES customers (id);
'''


def test_a_schema_is_known_by_its_suffix():
    assert is_schema("db/schema.sql") and is_schema("MIGRATIONS/001_init.DDL") and not is_schema("schema.py") and not is_schema("")


def test_tables_columns_and_dependencies_are_claimed_from_their_lines():
    claims = schema_claims(SCHEMA, "db/schema.sql")
    said = [(c.subject, c.predicate, c.object, c.first_line) for c in claims]
    assert ("db/schema.sql", "defines", "db/schema.sql:customers", 2) in said
    assert ("db/schema.sql:customers", "defines", "db/schema.sql:customers.email", 4) in said
    assert ("db/schema.sql", "defines", "db/schema.sql:orders", 8) in said, "quotes around a name are not part of it"
    assert ("db/schema.sql:orders", "defines", "db/schema.sql:orders.customer_id", 10) in said
    assert ("db/schema.sql:orders", "depends_on", "db/schema.sql:customers", 10) in said, "an inline REFERENCES"
    assert ("db/schema.sql:order_items", "depends_on", "db/schema.sql:orders", 19) in said, "a constraint's FOREIGN KEY"
    assert ("db/schema.sql", "defines", "db/schema.sql:customer_totals", 24) in said
    assert {o for s, p, o, _ in said if s == "db/schema.sql:customer_totals" and p == "depends_on"} == {
        "db/schema.sql:customers", "db/schema.sql:orders", "db/schema.sql:order_items"}
    assert ("db/schema.sql:orders", "depends_on", "db/schema.sql:customers", 31) in said, "an ALTER TABLE's foreign key"
    columns = [o for s, p, o, _ in said if p == "defines" and s == "db/schema.sql:order_items"]
    assert columns == ["db/schema.sql:order_items.order_id", "db/schema.sql:order_items.sku", "db/schema.sql:order_items.quantity"], \
        "constraints are not columns"
    assert not any("ghosts" in o for _, _, o, _ in said), "a table named in a comment is not a table"
    assert next(c.quote for c in claims if c.object == "db/schema.sql:customers.email") == "email TEXT NOT NULL UNIQUE,  -- trailing comment"
    assert schema_claims(SCHEMA, "notes.txt") == () and schema_claims("", "x.sql") == ()


def test_odd_spellings_are_read_and_a_table_never_rests_on_itself():
    text = 'CREATE TABLE `app`.`users` (\n  id int,\n  manager_id int REFERENCES [users]([id])\n);\nCREATE OR REPLACE VIEW v AS SELECT * FROM v_old, users;\n'
    said = [(c.subject, c.predicate, c.object) for c in schema_claims(text, "s.sql")]
    assert ("s.sql", "defines", "s.sql:app.users") in said and ("s.sql:app.users", "defines", "s.sql:app.users.manager_id") in said
    assert ("s.sql:app.users", "depends_on", "s.sql:users") in said, "a self-reference spelt another way is claimed as written"
    assert ("s.sql:v", "depends_on", "s.sql:v_old") in said and ("s.sql:v", "depends_on", "s.sql:users") in said


async def test_what_rests_on_a_table_is_answered_by_the_graph():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), code_graph=True).open()
    try:
        added = await engine.remember("default", SCHEMA, source="db/schema.sql")
        blast = await affected(engine, "default", "db/schema.sql:customers")
        assert blast.status == "found"
        assert {(r.label, r.depth, r.through) for r in blast.reached} >= {
            ("db/schema.sql:orders", 1, "depends_on"), ("db/schema.sql:customer_totals", 1, "depends_on"),
            ("db/schema.sql:order_items", 2, "depends_on")}
        nothing = await affected(engine, "default", "db/schema.sql:order_items")
        assert [r.label for r in nothing.reached] == ["db/schema.sql:customer_totals"]
        recorded: list = []
        said = await record_claims(engine, "default", episode_id=added.episode_id, content=SCHEMA, path="db/schema.sql",
                                   when="2026-01-01T00:00:00Z", _recorded=recorded)
        assert said == len(recorded) == len(schema_claims(SCHEMA, "db/schema.sql"))
    finally:
        await engine.close()
