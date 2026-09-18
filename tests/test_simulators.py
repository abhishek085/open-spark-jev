from open_spark_jev.data.simulators import SIMULATORS, generate, generate_all


def test_posteriors_normalised_and_records_valid():
    for name in SIMULATORS:
        recs = generate(name, 50, seed=1)
        for r in recs:
            q = r.question_obj()
            assert r.target["label"] in q.labels
            d = r.target_dist()
            assert abs(sum(d) - 1) < 1e-6
            r.label_index()


def test_generate_all_covers_domains_and_injection():
    recs = generate_all(100, seed=0, inject_frac=0.5)
    assert {r.domain for r in recs} == set(SIMULATORS)
    assert any(r.meta["injected"] for r in recs)
