#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OMEGA_FORGE_V13.py

V13: "Harnessed Structural Discovery" edition.

Design goals (practical, enforceable):
- Separate concerns: Generator (mutation) vs Judge (execution + detection) vs Harness (logging + anti-replay).
- Make "success" a state transition, not a print statement.
- Enforce DISTINCTNESS: success requires new structural hash AND new genome id.
- Stream evidence to disk (JSONL) with flush+fsync to avoid 0-byte outputs on interruption.

This is a research-stage toy VM / discovery loop (NOT a practical algorithm discovery system yet).
It is engineered to make cheating harder and to make evidence auditable.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable, Set


# -----------------------------
# Utilities
# -----------------------------

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def stable_hash_obj(obj) -> str:
    return sha1(json.dumps(obj, sort_keys=True, ensure_ascii=False))


# -----------------------------
# Tiny VM
# -----------------------------
# The VM is intentionally minimal, but supports non-linear control flow.
# Instruction = (op, a, b)
# Registers: r0..r3, memory[0..31]

OPS = [
    "NOP",
    "INC", "DEC",
    "ADD", "SUB", "MUL",
    "MOV",         # MOV rA, rB  (rA <- rB)
    "LOAD",        # LOAD rA, mem[idx]
    "STORE",       # STORE mem[idx], rA
    "JLT", "JGT",  # if rA < rB jump +imm ; if rA > rB jump +imm
    "JMP",         # jump +imm
    "SWAP",        # swap rA, rB
    "HALT",
]

REGS = 4
MEM = 32

@dataclass
class ExecState:
    halted_cleanly: bool = False
    error: Optional[str] = None
    steps: int = 0
    pc_trace: List[int] = field(default_factory=list)
    edge_trace: List[Tuple[int,int,str]] = field(default_factory=list)  # (pc, npc, type)
    instr_trace: List[str] = field(default_factory=list)
    writes: int = 0
    max_depth: int = 0  # reserved, no call stack in this tiny VM
    # loop-related
    loops_count: int = 0
    scc_count: int = 0

    def coverage(self, prog_len: int) -> float:
        if prog_len <= 0:
            return 0.0
        if not self.pc_trace:
            return 0.0
        return len(set(p for p in self.pc_trace if 0 <= p < prog_len)) / float(prog_len)

@dataclass
class Program:
    id: str
    instrs: List[Tuple[str,int,int]]  # (op,a,b)
    parents: List[str] = field(default_factory=list)

    def clone(self, new_id: Optional[str] = None) -> "Program":
        return Program(
            id=new_id or (self.id + "_c"),
            instrs=[(op,a,b) for (op,a,b) in self.instrs],
            parents=list(self.parents),
        )

class TinyVM:
    def __init__(self, max_steps: int = 256):
        self.max_steps = max_steps

    def execute(self, prog: Program, inputs: List[int], seed: int = 0) -> ExecState:
        rnd = random.Random(seed)
        st = ExecState()
        regs = [0]*REGS
        mem = [0]*MEM
        # simple input feed: copy first values into mem
        for i, v in enumerate(inputs[:MEM]):
            mem[i] = int(v) & 0xFFFFFFFF

        pc = 0
        steps = 0
        prog_len = len(prog.instrs)

        def edge_type(op: str) -> str:
            if op in ("JLT","JGT","JMP"):
                return "J"
            return "S"

        while steps < self.max_steps:
            if pc < 0 or pc >= prog_len:
                st.error = f"PC_OOB:{pc}"
                break

            op, a, b = prog.instrs[pc]
            st.pc_trace.append(pc)
            st.instr_trace.append(op)

            npc = pc + 1
            et = "S"

            try:
                if op == "NOP":
                    pass
                elif op == "INC":
                    regs[a % REGS] = (regs[a % REGS] + 1) & 0xFFFFFFFF
                elif op == "DEC":
                    regs[a % REGS] = (regs[a % REGS] - 1) & 0xFFFFFFFF
                elif op == "ADD":
                    regs[a % REGS] = (regs[a % REGS] + regs[b % REGS]) & 0xFFFFFFFF
                elif op == "SUB":
                    regs[a % REGS] = (regs[a % REGS] - regs[b % REGS]) & 0xFFFFFFFF
                elif op == "MUL":
                    regs[a % REGS] = (regs[a % REGS] * regs[b % REGS]) & 0xFFFFFFFF
                elif op == "MOV":
                    regs[a % REGS] = regs[b % REGS]
                elif op == "SWAP":
                    ra = a % REGS
                    rb = b % REGS
                    regs[ra], regs[rb] = regs[rb], regs[ra]
                elif op == "LOAD":
                    regs[a % REGS] = mem[b % MEM]
                elif op == "STORE":
                    mem[a % MEM] = regs[b % REGS]
                    st.writes += 1
                elif op == "JMP":
                    et = "J"
                    npc = pc + int(b)
                elif op == "JLT":
                    et = "J"
                    if regs[a % REGS] < regs[b % REGS]:
                        npc = pc + int(rnd.choice([-2,-1,1,2]) if b == 0 else b)
                elif op == "JGT":
                    et = "J"
                    if regs[a % REGS] > regs[b % REGS]:
                        npc = pc + int(rnd.choice([-2,-1,1,2]) if b == 0 else b)
                elif op == "HALT":
                    st.halted_cleanly = True
                    st.edge_trace.append((pc, pc, "H"))
                    break
                else:
                    st.error = f"UNKNOWN_OP:{op}"
                    break
            except Exception as e:
                st.error = f"EXC:{type(e).__name__}:{e}"
                break

            st.edge_trace.append((pc, npc, et))
            pc = npc
            steps += 1

        st.steps = steps
        # derive loop + SCC from observed edges
        st.loops_count, st.scc_count = analyze_loops_from_edges(st.edge_trace)
        return st


# -----------------------------
# CFG / Structural analysis
# -----------------------------

def analyze_loops_from_edges(edges: List[Tuple[int,int,str]]) -> Tuple[int,int]:
    # Build adjacency from observed edges (ignore HALT self edge)
    adj: Dict[int, Set[int]] = {}
    nodes: Set[int] = set()
    for u,v,t in edges:
        if t == "H":
            continue
        nodes.add(u); nodes.add(v)
        adj.setdefault(u,set()).add(v)

    # Tarjan SCC
    index = 0
    stack: List[int] = []
    onstack: Set[int] = set()
    idx: Dict[int,int] = {}
    low: Dict[int,int] = {}
    sccs: List[List[int]] = []

    sys.setrecursionlimit(10000)

    def strongconnect(v: int):
        nonlocal index
        idx[v] = index
        low[v] = index
        index += 1
        stack.append(v)
        onstack.add(v)

        for w in adj.get(v, ()):
            if w not in idx:
                strongconnect(w)
                low[v] = min(low[v], low[w])
            elif w in onstack:
                low[v] = min(low[v], idx[w])

        if low[v] == idx[v]:
            comp = []
            while True:
                w = stack.pop()
                onstack.remove(w)
                comp.append(w)
                if w == v:
                    break
            sccs.append(comp)

    for v in list(nodes):
        if v not in idx:
            strongconnect(v)

    # loop SCC: size>1 or self-loop present
    loops = 0
    for comp in sccs:
        if len(comp) > 1:
            loops += 1
        elif len(comp) == 1:
            u = comp[0]
            if u in adj and u in adj[u]:
                loops += 1
    return loops, len(sccs)

def canonical_cfg_signature(edges: List[Tuple[int,int,str]]) -> Dict:
    # Normalize to relative deltas + edge types, plus SCC sizes.
    # This avoids raw PC dependence but preserves structural motifs.
    deltas: List[Tuple[int,str]] = []
    for u,v,t in edges:
        if t == "H":
            continue
        deltas.append((int(v) - int(u), t))
    deltas.sort()
    loops, scc_count = analyze_loops_from_edges(edges)
    return {
        "deltas": deltas[:256],  # cap
        "loops": loops,
        "scc_count": scc_count,
        "edge_count": len(deltas),
        "delta_hist": histogram([d for d,_ in deltas], topk=16),
        "type_hist": histogram([t for _,t in deltas], topk=8),
    }

def histogram(items: List, topk: int = 16) -> List[Tuple[str,int]]:
    counts: Dict[str,int] = {}
    for it in items:
        k = str(it)
        counts[k] = counts.get(k, 0) + 1
    pairs = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return pairs[:topk]

def cfg_hash(sig: Dict) -> str:
    return stable_hash_obj(sig)

def cfg_edit_distance(sig_a: Dict, sig_b: Dict) -> int:
    # Simple proxy distance using delta multiset symmetric difference + hist diffs.
    # Not perfect, but stable and cheap.
    a = sig_a["deltas"]
    b = sig_b["deltas"]
    ca: Dict[Tuple[int,str],int] = {}
    cb: Dict[Tuple[int,str],int] = {}
    for x in a: ca[x] = ca.get(x,0)+1
    for x in b: cb[x] = cb.get(x,0)+1
    keys = set(ca)|set(cb)
    dist = 0
    for k in keys:
        dist += abs(ca.get(k,0)-cb.get(k,0))
    dist += abs(sig_a.get("loops",0)-sig_b.get("loops",0))*5
    dist += abs(sig_a.get("scc_count",0)-sig_b.get("scc_count",0))*2
    return int(dist)

def extract_active_subsequences(op_trace: List[str], min_len: int) -> List[Tuple[str,...]]:
    # contiguous subsequences from op trace
    subs: List[Tuple[str,...]] = []
    if len(op_trace) < min_len:
        return subs
    # sample a few windows to limit blow-up
    max_windows = 128
    step = max(1, len(op_trace)//max_windows)
    for i in range(0, len(op_trace)-min_len+1, step):
        subs.append(tuple(op_trace[i:i+min_len]))
    return subs


# -----------------------------
# Generator (mutation operators)
# -----------------------------

@dataclass
class GenConfig:
    pop_size: int = 40
    elite_keep: int = 8
    max_prog_len: int = 64
    min_prog_len: int = 8
    seed: int = 0

class Generator:
    def __init__(self, cfg: GenConfig):
        self.cfg = cfg
        self.rnd = random.Random(cfg.seed)

    def random_program(self, pid: str) -> Program:
        L = self.rnd.randint(self.cfg.min_prog_len, self.cfg.max_prog_len//2)
        instrs = [self.random_instr() for _ in range(L-1)] + [("HALT",0,0)]
        return Program(id=pid, instrs=instrs, parents=[])

    def random_instr(self) -> Tuple[str,int,int]:
        op = self.rnd.choice(OPS[:-1])  # exclude HALT for random body
        a = self.rnd.randint(0, max(REGS, MEM)-1)
        b = self.rnd.randint(-6, 6)
        # bias jumps to small offsets
        if op in ("JMP","JLT","JGT"):
            b = self.rnd.choice([-5,-4,-3,-2,-1,1,2,3,4,5])
        return (op, a, b)

    def mutate(self, parent: Program, new_id: str) -> Program:
        child = parent.clone(new_id=new_id)
        child.parents = [parent.id]
        r = self.rnd.random()

        # Structural operators (more likely than micro-ops)
        if r < 0.20:
            self.op_insert_block(child)
        elif r < 0.40:
            self.op_delete_block(child)
        elif r < 0.60:
            self.op_relocate_block(child)
        elif r < 0.80:
            self.op_insert_loop_skeleton(child)
        else:
            self.op_point_mutation(child)

        # Keep HALT at end (safety)
        if not child.instrs:
            child.instrs = [("HALT",0,0)]
        if child.instrs[-1][0] != "HALT":
            child.instrs[-1] = ("HALT",0,0)

        # length clamp
        if len(child.instrs) > self.cfg.max_prog_len:
            child.instrs = child.instrs[:self.cfg.max_prog_len-1] + [("HALT",0,0)]
        return child

    def op_point_mutation(self, prog: Program):
        if len(prog.instrs) <= 1:
            return
        i = self.rnd.randint(0, len(prog.instrs)-2)
        op,a,b = prog.instrs[i]
        if self.rnd.random() < 0.5:
            prog.instrs[i] = self.random_instr()
        else:
            # tweak params
            prog.instrs[i] = (op, (a+self.rnd.randint(0,3)) % max(REGS,MEM), b + self.rnd.choice([-2,-1,1,2]))

    def op_insert_block(self, prog: Program):
        if len(prog.instrs) >= self.cfg.max_prog_len:
            return
        k = self.rnd.randint(2, 6)
        block = [self.random_instr() for _ in range(k)]
        pos = self.rnd.randint(0, max(0, len(prog.instrs)-1))
        prog.instrs[pos:pos] = block

    def op_delete_block(self, prog: Program):
        if len(prog.instrs) <= self.cfg.min_prog_len:
            return
        pos = self.rnd.randint(0, max(0, len(prog.instrs)-2))
        k = self.rnd.randint(1, min(6, len(prog.instrs)-1-pos))
        del prog.instrs[pos:pos+k]

    def op_relocate_block(self, prog: Program):
        if len(prog.instrs) <= 8:
            return
        pos = self.rnd.randint(0, len(prog.instrs)-4)
        k = self.rnd.randint(2, 6)
        block = prog.instrs[pos:pos+k]
        del prog.instrs[pos:pos+k]
        dst = self.rnd.randint(0, max(0, len(prog.instrs)-1))
        prog.instrs[dst:dst] = block

    def op_insert_loop_skeleton(self, prog: Program):
        # Insert a small loop using JLT or JMP back-edge.
        if len(prog.instrs) >= self.cfg.max_prog_len-6:
            return
        # skeleton: INC r0; DEC r1; JGT r0 r1 -k  (or JMP back)
        r0 = self.rnd.randint(0, REGS-1)
        r1 = self.rnd.randint(0, REGS-1)
        skeleton = [
            ("INC", r0, 0),
            ("DEC", r1, 0),
            ("JGT", r0, r1 if r1 != 0 else 1),
            ("JMP", 0, -2),
        ]
        # fix JGT offset (b field) to a small negative jump
        # In this VM JGT uses b as jump offset; if b==0 it's randomized. We want deterministic negative.
        skeleton[2] = ("JGT", r0, r1)  # compare regs
        # We'll set jump offset by inserting explicit JMP back-edge; JGT only gates forward.
        # This still creates SCC due to JMP back-edge.
        pos = self.rnd.randint(0, max(0, len(prog.instrs)-1))
        prog.instrs[pos:pos] = skeleton


# -----------------------------
# Judge (detection + anti-cheat)
# -----------------------------

@dataclass
class DetectConfig:
    # rarity / structure
    K_cfg: int = 8                 # min cfg edit distance to parent signature
    min_sub_len: int = 8           # length of subseq window
    f_rarity: float = 0.001        # <0.1% frequency
    repro_N: int = 4               # reproducibility inputs
    cov_min: float = 0.55          # coverage threshold
    require_both: bool = True      # CFG AND subseq must pass
    min_loops: int = 1             # ban linear-only cheats
    max_steps: int = 256           # VM max steps

class StructuralJudge:
    def __init__(self, cfg: DetectConfig):
        self.cfg = cfg
        self.global_success_hashes: Set[str] = set()
        self.global_success_ids: Set[str] = set()
        self.subseq_counts: Dict[str,int] = {}
        self.subseq_total: int = 0

    def _subseq_key(self, subseq: Tuple[str,...]) -> str:
        return "|".join(subseq)

    def _update_archive(self, subseqs: List[Tuple[str,...]]):
        for s in subseqs:
            k = self._subseq_key(s)
            self.subseq_counts[k] = self.subseq_counts.get(k, 0) + 1
            self.subseq_total += 1

    def _is_rare(self, subseq: Tuple[str,...]) -> bool:
        if self.subseq_total <= 0:
            return True
        k = self._subseq_key(subseq)
        c = self.subseq_counts.get(k, 0)
        freq = c / float(self.subseq_total)
        return freq < self.cfg.f_rarity

    def evaluate(self, vm: TinyVM, prog: Program, parent: Optional[Program], inputs_seeds: List[Tuple[List[int],int]]) -> Tuple[bool, Dict]:
        # Run reproducibility suite
        suite: List[Dict] = []
        for inp, sd in inputs_seeds:
            st = vm.execute(prog, inp, seed=sd)
            suite.append({
                "seed": sd,
                "halted_cleanly": st.halted_cleanly,
                "error": st.error,
                "steps": st.steps,
                "writes": st.writes,
                "loops": st.loops_count,
                "scc_count": st.scc_count,
                "coverage": st.coverage(len(prog.instrs)),
                "trace_fp": sha1("".join(st.instr_trace[:128]) + "|" + ",".join(map(str, st.pc_trace[:128]))),
                "op_trace": st.instr_trace,
                "edges": st.edge_trace,
            })

        # Anti-cheat gate (must pass for ALL runs)
        for r in suite:
            if r["error"] is not None:
                return False, {"fail": "error", "detail": r["error"]}
            if not r["halted_cleanly"]:
                return False, {"fail": "dirty_halt"}
            if r["coverage"] < self.cfg.cov_min:
                return False, {"fail": "low_coverage", "cov": r["coverage"]}
            if r["loops"] < self.cfg.min_loops:
                return False, {"fail": "no_loops_linear_cheat", "loops": r["loops"], "scc": r["scc_count"]}

        # Reproducibility: same structural fingerprints across N runs (coarse)
        fp0 = (suite[0]["loops"], suite[0]["scc_count"])
        for r in suite[1:]:
            if (r["loops"], r["scc_count"]) != fp0:
                return False, {"fail": "non_repro_struct", "fp0": fp0, "fp": (r["loops"], r["scc_count"])}

        # Parent comparison (CFG distance)
        cfg_ok = False
        cfg_sig = canonical_cfg_signature(suite[0]["edges"])
        cfg_h = cfg_hash(cfg_sig)
        p_cfg_sig = None
        dist = None
        if parent is not None:
            pst = vm.execute(parent, inputs_seeds[0][0], seed=inputs_seeds[0][1])
            p_cfg_sig = canonical_cfg_signature(pst.edge_trace)
            dist = cfg_edit_distance(cfg_sig, p_cfg_sig)
            cfg_ok = dist >= self.cfg.K_cfg

        # Global distinctness gate
        if prog.id in self.global_success_ids:
            return False, {"fail": "replay_genome_id"}
        if cfg_h in self.global_success_hashes:
            return False, {"fail": "replay_struct_hash"}

        # Subsequence transition: must have at least one rare active subseq (from run0)
        subs = extract_active_subsequences(suite[0]["op_trace"], self.cfg.min_sub_len)
        rare = [s for s in subs if self._is_rare(s)]
        subseq_ok = len(rare) > 0

        # Require both or either
        passed = (cfg_ok and subseq_ok) if self.cfg.require_both else (cfg_ok or subseq_ok)
        if not passed:
            # still update archive to learn rarity landscape? we update only on pass to reduce drift.
            return False, {
                "fail": "gate_failed",
                "cfg_ok": cfg_ok, "subseq_ok": subseq_ok,
                "cfg_dist": dist,
                "cfg_hash": cfg_h,
                "rare_count": len(rare),
            }

        # Register success
        self.global_success_ids.add(prog.id)
        self.global_success_hashes.add(cfg_h)
        # Update archive (on success only)
        self._update_archive(subs)

        # Evidence bundle
        ev = {
            "id": prog.id,
            "parent": parent.id if parent else None,
            "cfg_hash": cfg_h,
            "cfg_sig": cfg_sig,
            "parent_cfg_sig": p_cfg_sig,
            "cfg_dist": dist,
            "subseq_min_len": self.cfg.min_sub_len,
            "example_rare_subseq": list(rare[0]) if rare else None,
            "coverage": suite[0]["coverage"],
            "loops": suite[0]["loops"],
            "scc_count": suite[0]["scc_count"],
            "writes": suite[0]["writes"],
            "steps": suite[0]["steps"],
            "repro_N": len(suite),
            "trace_fps": [r["trace_fp"] for r in suite],
        }
        return True, ev


# -----------------------------
# Harness (orchestration + streaming evidence)
# -----------------------------

@dataclass
class RunConfig:
    generations: int = 200
    seed: int = 0
    evidence_target: int = 6
    evidence_path: Path = Path("evidence_v13.jsonl")
    status_every: int = 10

class Engine:
    def __init__(self, gen_cfg: GenConfig, det_cfg: DetectConfig, run_cfg: RunConfig):
        self.gen_cfg = gen_cfg
        self.det_cfg = det_cfg
        self.run_cfg = run_cfg

        self.generator = Generator(gen_cfg)
        self.vm = TinyVM(max_steps=det_cfg.max_steps)
        self.judge = StructuralJudge(det_cfg)

        self.population: List[Program] = []
        self.id_to_prog: Dict[str,Program] = {}
        self.rnd = random.Random(run_cfg.seed)

        # inputs for reproducibility suite
        self.inputs_seeds = self._make_repro_suite(det_cfg.repro_N)

    def _make_repro_suite(self, N: int) -> List[Tuple[List[int],int]]:
        suite = []
        base = [1,2,3,4,5,6,7,8]
        for i in range(N):
            inp = [(v + i) % 9 for v in base]
            suite.append((inp, i))
        return suite

    def init_population(self):
        self.population = []
        self.id_to_prog = {}
        for i in range(self.gen_cfg.pop_size):
            pid = f"g0_{self.rnd.randint(10000,99999)}_{i}"
            p = self.generator.random_program(pid)
            self.population.append(p)
            self.id_to_prog[p.id] = p

    def _select_elite(self) -> List[Program]:
        # score = coverage + loops bonus - errors; computed on single run (cheap)
        scored = []
        for p in self.population:
            st = self.vm.execute(p, [1,2,3,4,5,6,7,8], seed=0)
            score = 0.0
            if st.error is None and st.halted_cleanly:
                score = st.coverage(len(p.instrs)) + 0.15*st.loops_count - 0.02*st.writes
            scored.append((score, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _,p in scored[:self.gen_cfg.elite_keep]]

    def evolve_one_gen(self, gen: int, evidence_fh) -> int:
        elite = self._select_elite()
        next_pop: List[Program] = []
        # keep elites (cloned to ensure new IDs)
        for e in elite:
            cid = f"{e.id}_k{gen}"
            c = e.clone(new_id=cid)
            c.parents = [e.id]
            next_pop.append(c)

        # generate rest
        while len(next_pop) < self.gen_cfg.pop_size:
            parent = self.rnd.choice(elite)
            nid = f"g{gen}_{self.rnd.randint(10000,99999)}"
            child = self.generator.mutate(parent, nid)
            next_pop.append(child)

        # Replace
        self.population = next_pop
        self.id_to_prog = {p.id: p for p in self.population}

        # Evaluate for evidence: scan population; write ALL successes (distinctness is enforced in judge)
        successes = 0
        for p in self.population:
            parent = self.id_to_prog.get(p.parents[0]) if p.parents else None
            ok, ev = self.judge.evaluate(self.vm, p, parent, self.inputs_seeds)
            if ok:
                ev2 = {"gen": gen, "ts": now_ts(), **ev}
                evidence_fh.write(json.dumps(ev2, ensure_ascii=False) + "\n")
                evidence_fh.flush()
                os.fsync(evidence_fh.fileno())
                successes += 1
        return successes

    def run_evidence(self):
        self.init_population()
        target = self.run_cfg.evidence_target
        outpath = Path(self.run_cfg.evidence_path)
        outpath.parent.mkdir(parents=True, exist_ok=True)

        # open in append mode to preserve partial progress across runs
        found_before = 0
        if outpath.exists() and outpath.stat().st_size > 0:
            # count lines (best-effort)
            try:
                with outpath.open("r", encoding="utf-8") as fh:
                    for _ in fh:
                        found_before += 1
            except Exception:
                found_before = 0

        with outpath.open("a", encoding="utf-8") as fh:
            total_found = found_before
            for gen in range(1, self.run_cfg.generations+1):
                s = self.evolve_one_gen(gen, fh)
                total_found += s
                if gen % self.run_cfg.status_every == 0 or s > 0:
                    print(f"[gen {gen}] successes_this_gen={s} total_evidence_lines={total_found}", flush=True)
                if total_found >= target:
                    print(f"[DONE] target reached: {total_found} >= {target}", flush=True)
                    break

        return outpath

    def selftest(self):
        self.init_population()
        # quick: ensure we can run 3 gens and write evidence without 0-byte
        tmp = Path("/tmp/omega_v13_selftest.jsonl")
        if tmp.exists():
            tmp.unlink()
        self.run_cfg.evidence_path = tmp
        self.run_cfg.evidence_target = 1
        self.run_cfg.generations = 10
        p = self.run_evidence()
        if not p.exists() or p.stat().st_size == 0:
            raise SystemExit("SELFTEST_FAIL: evidence file missing/empty")
        print("SELFTEST_OK", p, "bytes=", p.stat().st_size, flush=True)


# -----------------------------
# CLI
# -----------------------------

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("selftest")
    p2 = sub.add_parser("evidence_run")
    p2.add_argument("--target", type=int, default=6)
    p2.add_argument("--max_generations", type=int, default=500)
    p2.add_argument("--seed", type=int, default=0)
    p2.add_argument("--out", type=str, default="evidence_v13.jsonl")

    p3 = sub.add_parser("run")
    p3.add_argument("--generations", type=int, default=200)
    p3.add_argument("--seed", type=int, default=0)

    return ap

def main(argv: List[str]) -> int:
    ap = build_argparser()
    args = ap.parse_args(argv)

    gen_cfg = GenConfig(seed=getattr(args, "seed", 0))
    det_cfg = DetectConfig()
    run_cfg = RunConfig(seed=getattr(args, "seed", 0))

    eng = Engine(gen_cfg, det_cfg, run_cfg)

    if args.cmd == "selftest":
        eng.selftest()
        return 0

    if args.cmd == "evidence_run":
        eng.run_cfg.evidence_target = int(args.target)
        eng.run_cfg.generations = int(args.max_generations)
        eng.run_cfg.seed = int(args.seed)
        eng.run_cfg.evidence_path = Path(args.out)
        p = eng.run_evidence()
        print(f"Evidence written to: {p} (bytes={p.stat().st_size})", flush=True)
        return 0

    if args.cmd == "run":
        # simple run without evidence; prints periodic status
        eng.init_population()
        for gen in range(1, int(args.generations)+1):
            # no evidence file; use os.devnull
            with open(os.devnull, "w") as fh:
                s = eng.evolve_one_gen(gen, fh)
            if gen % 10 == 0:
                print(f"[gen {gen}] successes={s} global_success={len(eng.judge.global_success_ids)}", flush=True)
        return 0

    return 2

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
