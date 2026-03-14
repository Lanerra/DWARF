"""Populate the DWARF research knowledge graph in Neo4j."""

import csv
from pathlib import Path

from neo4j import GraphDatabase

URI = "bolt://localhost:7688"
USER = "neo4j"
PASSWORD = "dwarf-research"

RESULTS_TSV = Path(__file__).parent / "results.tsv"

COMMIT_TO_KERNEL = {
    "28271d2": "V5",
    "7a6d61e": "V3",
    "d466147": "V3",
    "54639c0": "d41j16d",
    "98b38ef": "d41j16d",
    "a2e6f05": "V3",
    "8530af6": "d41j16d",
    "43655f4": "d41j16d",
}

COMMIT_TO_OFFSETS = {
    "28271d2": "condU_44",
    "7a6d61e": "J16D",
    "d466147": "J16D",
    "54639c0": "J16D",
    "98b38ef": "J16D",
    "a2e6f05": "J16D",
    "8530af6": "J16D",
    "43655f4": "J16D",
}


def create_datasets(session):
    session.run(
        """
        MERGE (d:Dataset {id: 'fineweb_edu'})
        SET d.name = 'FineWeb-Edu',
            d.train_sequences = 52716,
            d.val_sequences = 5582,
            d.sequence_length = 2048,
            d.total_tokens_approx = 108000000
        """
    )
    session.run(
        """
        MERGE (d:Dataset {id: 'project_gutenberg'})
        SET d.name = 'Project Gutenberg',
            d.status = 'future'
        """
    )


def create_offset_sets(session):
    condu_offsets = list(range(0, 33)) + [48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536]
    session.run(
        """
        MERGE (o:OffsetSet {id: 'condU_44'})
        SET o.j = 44,
            o.max_delta = 1536,
            o.offsets = $offsets,
            o.source = 'condU architecture'
        """,
        offsets=condu_offsets,
    )

    j16d_offsets = [1, 3, 4, 13, 15, 21, 23, 28, 48, 64, 96, 192, 384, 512, 768, 1024]
    session.run(
        """
        MERGE (o:OffsetSet {id: 'J16D'})
        SET o.j = 16,
            o.offsets = $offsets,
            o.max_delta = 1024,
            o.max_hops = 2,
            o.full_coverage = true,
            o.relay_optimal = true
        """,
        offsets=j16d_offsets,
    )


def create_kernels(session):
    kernels = [
        {
            "id": "V3",
            "description": "pos_bias + scale_embed (Q-dynamic matched filter)",
            "has_movt": False,
            "has_qk_ovt": False,
            "has_npci": False,
        },
        {
            "id": "V4",
            "description": "V3 + condU base",
            "has_movt": False,
            "has_qk_ovt": False,
            "has_npci": False,
        },
        {
            "id": "V5",
            "description": "V4 + MOVT + QK-OVT + NPCI",
            "has_movt": True,
            "has_qk_ovt": True,
            "has_npci": True,
        },
        {
            "id": "V6",
            "description": "V5 + NaN guards + correctness fixes (Mar 12 2026)",
            "has_movt": True,
            "has_qk_ovt": True,
            "has_npci": True,
        },
        {
            "id": "d41j16d",
            "description": "V3 + J16D offsets (no MOVT/QK-OVT/NPCI)",
            "has_movt": False,
            "has_qk_ovt": False,
            "has_npci": False,
        },
        {
            "id": "v6j16d",
            "description": "V6 + J16D offsets (MOVT+QK-OVT+NPCI, no EMA)",
            "has_movt": True,
            "has_qk_ovt": True,
            "has_npci": True,
        },
    ]
    for kernel in kernels:
        session.run(
            """
            MERGE (k:Kernel {id: $id})
            SET k.description = $description,
                k.has_movt = $has_movt,
                k.has_qk_ovt = $has_qk_ovt,
                k.has_npci = $has_npci
            """,
            id=kernel["id"],
            description=kernel["description"],
            has_movt=kernel["has_movt"],
            has_qk_ovt=kernel["has_qk_ovt"],
            has_npci=kernel["has_npci"],
        )


def create_methods(session):
    methods = [
        {"id": "DSQG", "name": "Dyadic Sparse Query-Gate attention", "key_property": "O(1) KV cache"},
        {"id": "RelayChain", "name": "multi-hop long-range information relay via learned offset positions", "key_property": None},
        {"id": "ScaleEmbed", "name": "Q-weighted scale gains per offset per head (lr_mult unlock)", "key_property": None},
        {"id": "KalmanEMA", "name": "temporal smoothing via Kalman-inspired exponential moving average", "key_property": None},
        {"id": "KdVCorrection", "name": "soliton field correction post-EMA", "key_property": None},
        {"id": "AGC", "name": "automatic gain control normalization", "key_property": None},
        {"id": "HuygensIF", "name": "Huygens interference via interference factor layers", "key_property": None},
        {"id": "MOVT", "name": "Multi-plane Orthogonal Value Transport", "key_property": None},
        {"id": "QK_OVT", "name": "Query-Key Orthogonal Value Transport", "key_property": None},
        {"id": "NPCI", "name": "Non-Positive Causal Interference", "key_property": None},
    ]
    for method in methods:
        session.run(
            """
            MERGE (m:Method {id: $id})
            SET m.name = $name,
                m.key_property = $key_property
            """,
            id=method["id"],
            name=method["name"],
            key_property=method["key_property"],
        )


def parse_results_tsv():
    rows = []
    with open(RESULTS_TSV, newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            if row["status"] == "crash":
                continue
            rows.append(row)
    return rows


def create_runs_from_tsv(session):
    rows = parse_results_tsv()
    for row in rows:
        commit = row["commit"]
        run_id = f"ar_{commit}"

        ar_score = None
        try:
            ar_score = float(row["ar_score"])
        except (ValueError, TypeError):
            pass

        passkey = float(row["passkey_ep3"]) if row["passkey_ep3"] else None
        ppl = float(row["ppl_ep3"]) if row["ppl_ep3"] else None
        memory_mb = float(row["memory_mb"]) if row["memory_mb"] else None

        session.run(
            """
            MERGE (r:Run {id: $run_id})
            SET r.commit = $commit,
                r.ar_score = $ar_score,
                r.passkey_ep3 = $passkey,
                r.ppl_ep3 = $ppl,
                r.memory_mb = $memory_mb,
                r.status = $status,
                r.description = $description,
                r.source = 'autoresearch'
            """,
            run_id=run_id,
            commit=commit,
            ar_score=ar_score,
            passkey=passkey,
            ppl=ppl,
            memory_mb=memory_mb,
            status=row["status"],
            description=row["description"],
        )

        result_id = f"result_{commit}"
        session.run(
            """
            MERGE (res:Result {id: $result_id})
            SET res.ar_score = $ar_score,
                res.passkey = $passkey,
                res.ppl = $ppl
            """,
            result_id=result_id,
            ar_score=ar_score,
            passkey=passkey,
            ppl=ppl,
        )

        session.run(
            """
            MATCH (r:Run {id: $run_id})
            MATCH (res:Result {id: $result_id})
            MERGE (r)-[:PRODUCES]->(res)
            """,
            run_id=run_id,
            result_id=result_id,
        )

        session.run(
            """
            MATCH (r:Run {id: $run_id})
            MATCH (d:Dataset {id: 'fineweb_edu'})
            MERGE (r)-[:TRAINED_ON]->(d)
            """,
            run_id=run_id,
        )

        kernel_id = COMMIT_TO_KERNEL.get(commit)
        if kernel_id:
            session.run(
                """
                MATCH (r:Run {id: $run_id})
                MATCH (k:Kernel {id: $kernel_id})
                MERGE (r)-[:USES_KERNEL]->(k)
                """,
                run_id=run_id,
                kernel_id=kernel_id,
            )

        offset_id = COMMIT_TO_OFFSETS.get(commit)
        if offset_id:
            session.run(
                """
                MATCH (r:Run {id: $run_id})
                MATCH (o:OffsetSet {id: $offset_id})
                MERGE (r)-[:USES_OFFSETS]->(o)
                """,
                run_id=run_id,
                offset_id=offset_id,
            )


def create_memory_runs(session):
    memory_runs = [
        {
            "id": "condU_35M",
            "test_ppl": 38.293,
            "passkey": 90.0,
            "val_ppl": 38.293,
            "description": "condU 35M hybrid — beats standard 85M (39.447) by -1.154 PPL with 59% fewer params",
            "kernel": "V3",
            "offsets": "condU_44",
            "params_millions": 39,
        },
        {
            "id": "condM_85M",
            "test_ppl": 36.042,
            "passkey": None,
            "val_ppl": 37.0,
            "description": "condM 85M — beats standard 85M by -3.405 PPL with 13% fewer params",
            "kernel": None,
            "offsets": "condU_44",
            "params_millions": 88,
        },
        {
            "id": "standard_85M",
            "test_ppl": 39.447,
            "passkey": 96.7,
            "val_ppl": None,
            "description": "standard 85M baseline — full O(N²) causal attention",
            "kernel": None,
            "offsets": None,
            "params_millions": 101,
        },
        {
            "id": "condX_v2_35M",
            "test_ppl": 38.171,
            "val_ppl": 37.729,
            "passkey": 96.7,
            "description": "condX-v2 35M BF16 — current paper headline",
            "kernel": "V5",
            "offsets": "condU_44",
            "params_millions": 39,
        },
    ]

    for run in memory_runs:
        session.run(
            """
            MERGE (r:Run {id: $id})
            SET r.test_ppl = $test_ppl,
                r.passkey = $passkey,
                r.val_ppl = $val_ppl,
                r.description = $description,
                r.source = 'memory',
                r.params_millions = $params_millions
            """,
            id=run["id"],
            test_ppl=run["test_ppl"],
            passkey=run["passkey"],
            val_ppl=run["val_ppl"],
            description=run["description"],
            params_millions=run["params_millions"],
        )

        result_id = f"result_{run['id']}"
        session.run(
            """
            MERGE (res:Result {id: $result_id})
            SET res.test_ppl = $test_ppl,
                res.passkey = $passkey,
                res.val_ppl = $val_ppl
            """,
            result_id=result_id,
            test_ppl=run["test_ppl"],
            passkey=run["passkey"],
            val_ppl=run["val_ppl"],
        )

        session.run(
            """
            MATCH (r:Run {id: $run_id})
            MATCH (res:Result {id: $result_id})
            MERGE (r)-[:PRODUCES]->(res)
            """,
            run_id=run["id"],
            result_id=result_id,
        )

        session.run(
            """
            MATCH (r:Run {id: $run_id})
            MATCH (d:Dataset {id: 'fineweb_edu'})
            MERGE (r)-[:TRAINED_ON]->(d)
            """,
            run_id=run["id"],
        )

        if run["kernel"]:
            session.run(
                """
                MATCH (r:Run {id: $run_id})
                MATCH (k:Kernel {id: $kernel_id})
                MERGE (r)-[:USES_KERNEL]->(k)
                """,
                run_id=run["id"],
                kernel_id=run["kernel"],
            )

        if run["offsets"]:
            session.run(
                """
                MATCH (r:Run {id: $run_id})
                MATCH (o:OffsetSet {id: $offset_id})
                MERGE (r)-[:USES_OFFSETS]->(o)
                """,
                run_id=run["id"],
                offset_id=run["offsets"],
            )


def create_hypotheses(session):
    hypotheses = [
        {
            "id": "hyp_scale_embed_threshold",
            "statement": "scale_embed|max| must cross ~0.74 basin boundary for relay chain activation",
        },
        {
            "id": "hyp_ema_crossover",
            "statement": "EMA beneficial ≤14M, harmful ≥35M — EMA competes with relay chain gradient at scale",
        },
        {
            "id": "hyp_lr_mult_scaling",
            "statement": "lr_mult must scale with HD: HD≤32→10.0, HD=64→15.0, HD≈128→20.0 (hypothesis)",
        },
        {
            "id": "hyp_j16d_optimal",
            "statement": "J=16 relay-optimal offsets sufficient for full dense[0,1536] coverage via max_hops=2",
        },
        {
            "id": "hyp_relay_chain_inside_out",
            "statement": "relay chain forms inside-out, growing outward from core distances each epoch",
        },
        {
            "id": "hyp_chinchilla_invalid",
            "statement": "Chinchilla-optimal framing does not apply to DSQG; repeated data beneficial not harmful",
        },
    ]
    for hypothesis in hypotheses:
        session.run(
            """
            MERGE (h:Hypothesis {id: $id})
            SET h.statement = $statement
            """,
            id=hypothesis["id"],
            statement=hypothesis["statement"],
        )


def create_paper(session):
    session.run(
        """
        MERGE (p:Paper {id: 'dwarf_paper'})
        SET p.title = 'DWARF: Dyadic Wave And Resonant Field Attention',
            p.status = 'in_preparation'
        """
    )


def create_confirmation_relationships(session):
    # 30M ar_score=+12.84 (d466147) confirms scale_embed threshold
    session.run(
        """
        MATCH (res:Result {id: 'result_d466147'})
        MATCH (h:Hypothesis {id: 'hyp_scale_embed_threshold'})
        MERGE (res)-[:CONFIRMS]->(h)
        """
    )

    # 54639c0: EMA beneficial at 14M (+3.33 vs pure J16D)
    session.run(
        """
        MATCH (res:Result {id: 'result_54639c0'})
        MATCH (h:Hypothesis {id: 'hyp_ema_crossover'})
        MERGE (res)-[c:CONFIRMS]->(h)
        SET c.note = 'EMA beneficial at 14M: +3.33 vs pure J16D'
        """
    )

    # 98b38ef: EMA harmful at 35M (stalled relay chain)
    session.run(
        """
        MATCH (res:Result {id: 'result_98b38ef'})
        MATCH (h:Hypothesis {id: 'hyp_ema_crossover'})
        MERGE (res)-[c:CONFIRMS]->(h)
        SET c.note = 'EMA harmful at 35M: stalled relay chain at d=32'
        """
    )

    # 43655f4 (d41-J16D-35M) confirms EMA crossover at 35M
    session.run(
        """
        MATCH (res:Result {id: 'result_43655f4'})
        MATCH (h:Hypothesis {id: 'hyp_ema_crossover'})
        MERGE (res)-[c:CONFIRMS]->(h)
        SET c.note = 'ar_score=+3.84 at 35M with EMA, confirms crossover'
        """
    )

    # J16D runs confirm J16D optimality
    session.run(
        """
        MATCH (res:Result {id: 'result_d466147'})
        MATCH (h:Hypothesis {id: 'hyp_j16d_optimal'})
        MERGE (res)-[c:CONFIRMS]->(h)
        SET c.note = '11/12 distances at 20% with J=16 relay-optimal offsets'
        """
    )

    session.run(
        """
        MATCH (res:Result {id: 'result_a2e6f05'})
        MATCH (h:Hypothesis {id: 'hyp_j16d_optimal'})
        MERGE (res)-[c:CONFIRMS]->(h)
        SET c.note = 'ar_score=+14.96 NEW RECORD with J16D offsets'
        """
    )


def print_summary(session):
    result = session.run(
        """
        MATCH (n)
        RETURN labels(n)[0] AS label, count(n) AS count
        ORDER BY count DESC
        """
    )
    print("\n\U0001f4ca Node counts:")
    for record in result:
        print(f"  {record['label']}: {record['count']}")

    result = session.run(
        """
        MATCH ()-[r]->()
        RETURN type(r) AS relationship_type, count(r) AS count
        ORDER BY count DESC
        """
    )
    print("\n\U0001f517 Relationship counts:")
    for record in result:
        print(f"  {record['relationship_type']}: {record['count']}")


def main():
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

    with driver.session() as session:
        print("\U0001f527 Creating datasets...")
        create_datasets(session)

        print("\U0001f527 Creating offset sets...")
        create_offset_sets(session)

        print("\U0001f527 Creating kernels...")
        create_kernels(session)

        print("\U0001f527 Creating methods...")
        create_methods(session)

        print("\U0001f527 Creating runs from results.tsv...")
        create_runs_from_tsv(session)

        print("\U0001f527 Creating runs from MEMORY.md...")
        create_memory_runs(session)

        print("\U0001f527 Creating hypotheses...")
        create_hypotheses(session)

        print("\U0001f527 Creating paper node...")
        create_paper(session)

        print("\U0001f527 Creating confirmation relationships...")
        create_confirmation_relationships(session)

        print_summary(session)

    driver.close()
    print("\n\u2705 Research graph populated successfully.")


if __name__ == "__main__":
    main()
