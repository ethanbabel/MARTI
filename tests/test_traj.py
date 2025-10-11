import time
import pytest
from test_workflow import WorkflowTrajectory


# ----------------- Helpers ----------------- #
def record_step_with_parents(
    wt: WorkflowTrajectory,
    agent_name: str,
    turn_id: int,
    input_text: str,
    output_text: str,
    parents=None,
    **kw,
):
    meta = kw.pop("metadata", {}) or {}
    if parents is not None:
        meta = {**meta, "parents": parents}
    return wt.record_step(
        agent_name=agent_name,
        turn_id=turn_id,
        input_text=input_text,
        output_text=output_text,
        metadata=meta,
        **kw,
    )


def mermaid_graph_dag(wt, show_text=True):
    steps = wt.global_steps
    by_id = {s.global_turn_id: s for s in steps}
    parents_map = {}
    for s in steps:
        p = s.metadata.get("parents")
        if p is None:  # fallback: chain
            p = [s.global_turn_id - 1] if s.global_turn_id > 1 else []
        parents_map[s.global_turn_id] = p

    children_map = {sid: [] for sid in parents_map}
    for sid, plist in parents_map.items():
        for p in plist:
            if p in children_map:
                children_map[p].append(sid)

    roots = [sid for sid, ps in parents_map.items() if not ps]
    leaves = [sid for sid, ch in children_map.items() if not ch]

    lines = ["graph TD"]
    if roots:
        lines.append("  START([START])")
    if leaves:
        lines.append("  FINAL([FINAL])")

    def mk_label(s):
        if show_text:
            txt = (s.output_text or "")[:16].replace('"', "'")
            return f"{s.agent_name}#{s.turn_id}\\n{txt}"
        return f"{s.agent_name}#{s.turn_id}"

    for s in steps:
        lines.append(f'  N{s.global_turn_id}["{mk_label(s)}"]')

    for r in roots:
        lines.append(f"  START --> N{r}")
    for sid, plist in parents_map.items():
        for p in plist:
            if p in by_id:
                lines.append(f"  N{p} --> N{sid}")
    for l in leaves:
        lines.append(f"  N{l} --> FINAL")
    print("\n".join(lines))
    return "\n".join(lines)


def extract_edges(mermaid_txt):
    """
    Return set of tuples ('Na','Nb') from 'Na --> Nb' lines, ignoring START/FINAL text wrappers.
    """
    edges = set()
    for line in mermaid_txt.splitlines():
        line = line.strip()
        if "-->" in line:
            a, b = line.split("-->")
            a = a.strip()
            b = b.strip()
            # remove e.g. 'N1', 'FINAL([FINAL])'
            a = a.split("(")[0].strip()
            b = b.split("(")[0].strip()
            edges.add((a, b))
    return edges


# ----------------- Builders ----------------- #
def build_chain():
    wt = WorkflowTrajectory(
        workflow_id="chain",
        prompt="Chain demo",
        label="chain",
        start_time=time.time(),
    )
    # ─── Round-1 ───
    # step-1 : Exec
    record_step_with_parents(wt, "Exec",    1, "S1",  "S2",  parents=[])
    # step-2 : Planner 依赖 step-1
    record_step_with_parents(wt, "Planner", 2, "S2",  "S3",  parents=[1])
    # step-3 : Solver  依赖 step-2
    record_step_with_parents(wt, "Solver",  2, "S3",  "Done", parents=[2])
    # ─── Round-2 ───
    # step-4 : Planner 依赖 step-3
    record_step_with_parents(wt, "Planner", 3, "Done", "New-S3", parents=[3])
    # step-5 : Solver  依赖 step-4
    record_step_with_parents(wt, "Solver",  3, "New-S3", "Done-2", parents=[4])
    wt.finalize("Done-2", 1.0)
    return wt

def build_mad():
    # wt = WorkflowTrajectory(
    #     workflow_id="mad",
    #     prompt="Debate",
    #     label="MAD",
    #     start_time=time.time(),
    # )
    # record_step_with_parents(wt, "Pro", 1, "topic", "clean", parents=[])
    # record_step_with_parents(wt, "Con", 1, "topic", "risk", parents=[])
    # record_step_with_parents(wt, "Judge", 1, "r1", "go r2", parents=[1, 2])
    # record_step_with_parents(wt, "Pro", 2, "r2", "manage", parents=[3])
    # record_step_with_parents(wt, "Con", 2, "r2", "waste", parents=[3])
    # record_step_with_parents(wt, "Judge", 2, "final", "Pro wins", parents=[4, 5])
    # record_step_with_parents(wt, "Pro", 3, "r2", "manage", parents=[6])
    # record_step_with_parents(wt, "Con", 3, "r2", "waste", parents=[6])
    # record_step_with_parents(wt, "Judge", 3, "final", "Pro wins", parents=[7, 8])
    # wt.finalize("Pro wins", 0.8)

    wt = WorkflowTrajectory(
        workflow_id="mad_3r",
        prompt="Three-round MAD",
        label="MAD",
        start_time=time.time(),
    )

    # Round-1  ───────────────────────────────────────────
    s1 = record_step_with_parents(wt, "model1", 1, "START", "o11", parents=[])
    s2 = record_step_with_parents(wt, "model2", 1, "START", "o21", parents=[])
    s3 = record_step_with_parents(wt, "model3", 1, "START", "o31", parents=[])

    # 收集上一轮所有 id 供后面引用
    r1_ids = [s1.global_turn_id, s2.global_turn_id, s3.global_turn_id]

    # Round-2  ───────────────────────────────────────────
    s4 = record_step_with_parents(wt, "model1", 2, "o11+o21+o31", "o12", parents=r1_ids)
    s5 = record_step_with_parents(wt, "model2", 2, "o11+o21+o31", "o22", parents=r1_ids)
    s6 = record_step_with_parents(wt, "model3", 2, "o11+o21+o31", "o32", parents=r1_ids)

    r2_ids = [s4.global_turn_id, s5.global_turn_id, s6.global_turn_id]

    # Round-3  ───────────────────────────────────────────
    s7 = record_step_with_parents(wt, "model1", 3, "o12+o22+o32", "o13", parents=r2_ids)
    s8 = record_step_with_parents(wt, "model2", 3, "o12+o22+o32", "o23", parents=r2_ids)
    s9 = record_step_with_parents(wt, "model3", 3, "o12+o22+o32", "o33", parents=r2_ids)

    wt.finalize("o13 / o23 / o33", final_reward=0.9)
    return wt


def build_moa():
    wt = WorkflowTrajectory(
        workflow_id="moa",
        prompt="Summarize",
        label="MoA",
        start_time=time.time(),
    )
    record_step_with_parents(wt, "A", 1, "doc", "outA", parents=[])
    record_step_with_parents(wt, "B", 1, "doc", "outB", parents=[])
    record_step_with_parents(wt, "C", 1, "doc", "outC", parents=[])
    record_step_with_parents(wt, "Agg", 1, "merge", "merged", parents=[1, 2, 3])
    wt.finalize("merged", 0.95)
    return wt


def build_adaptive():
    wt = WorkflowTrajectory(
        workflow_id="adaptive",
        prompt="Classify",
        label="Adaptive",
        start_time=time.time(),
    )
    record_step_with_parents(wt, "Router", 1, "img", "use vision", parents=[])
    record_step_with_parents(wt, "Vision", 1, "img", "cat", parents=[1])
    record_step_with_parents(wt, "Router", 2, "cat", "return cat", parents=[2])
    wt.finalize("cat", 1.0)
    return wt


# ----------------- Tests ----------------- #
def test_chain_graph():
    wt = build_chain()
    txt = mermaid_graph_dag(wt)
    edges = extract_edges(txt)
    # START->N1; N1->N2->N3->N4->FINAL
    assert ("START", "N1") in edges
    assert ("N1", "N2") in edges and ("N2", "N3") in edges and ("N3", "N4") in edges


def test_mad_graph():
    wt = build_mad()
    txt = mermaid_graph_dag(wt)
    edges = extract_edges(txt)
    # two roots -> Judge1
    assert ("START", "N1") in edges and ("START", "N2") in edges
    assert ("N1", "N3") in edges and ("N2", "N3") in edges
    # Judge1 fanout -> Pro2, Con2
    assert ("N3", "N4") in edges and ("N3", "N5") in edges
    # Pro2+Con2 -> Judge2
    assert ("N4", "N6") in edges and ("N5", "N6") in edges


def test_moa_graph():
    wt = build_moa()
    txt = mermaid_graph_dag(wt)
    edges = extract_edges(txt)
    # 3 roots
    assert ("START", "N1") in edges and ("START", "N2") in edges and ("START", "N3") in edges
    # Agg merges all
    for p in ("N1", "N2", "N3"):
        assert (p, "N4") in edges


def test_adaptive_graph():
    wt = build_adaptive()
    txt = mermaid_graph_dag(wt)
    edges = extract_edges(txt)
    assert ("START", "N1") in edges
    assert ("N1", "N2") in edges
    assert ("N2", "N3") in edges
    
if __name__ == "__main__":
    test_chain_graph()
    test_mad_graph()
    test_moa_graph()
    test_adaptive_graph()