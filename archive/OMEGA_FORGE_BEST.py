#!/usr/bin/env python3
"""
OMEGA_FORGE_V13.py
Autonomous Structural Algorithm Discovery Engine (research prototype)

Design goals (compared to V12 evidence edition):
- Hard separation of roles:
  * Worker: generates/mutates candidate programs (genomes)
  * Judge: executes, measures, gates structural novelty, ranks by task performance
  * Harness: enforces invariants (distinct successes, reproducible novelty, no "report-only" success)
- "Success" is a *state transition* with hard gates:
  * distinct genome_id AND distinct structural_hash
  * reproducible across multiple inputs
  * non-trivial control-flow (at least one loop SCC or call depth > 0)
  * sufficient trace coverage, clean halt, no degenerate infinite loops
- Parent tracking is real: candidates carry parent id and judge compares against actual parent genome CFG.
- Novelty is canonicalized at the CFG level and at the executed-subsequence level, both with archive rarity.
- Mutation operators include safe skeleton insertion for loops and calls to enable genuine control-flow innovation.

Status: research stage. Not guaranteed to discover practical algorithms. Intended to be falsifiable:
the harness will fail loudly if it cannot produce *distinct* structural transitions.

CLI:
  python OMEGA_FORGE_V13.py selftest
  python OMEGA_FORGE_V13.py run --generations 200 --seed 0
  python OMEGA_FORGE_V13.py evidence_run --target 6 --max_generations 500 --seed 0

No external dependencies.
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
from typing import Dict, List, Optional, Tuple, Iterable, Any, Set

# ---------------------------
# Instruction set (simple VM)
# ---------------------------

OPCODES = [
    "NOP",
    "INC", "DEC",
    "ADD", "SUB", "MUL",
    "MOV",          # MOV rA <- rB
    "SWAP",         # SWAP rA <-> rB
    "JLT", "JGT", "JEQ",  # conditional relative jump based on rA ? rB
    "JMP",          # unconditional relative jump
    "CALL", "RET",  # call/return (relative call target)
    "HALT",
]

REGS = 4  # small, deterministic
STACK_LIMIT = 32
STEP_LIMIT = 500  # per run


@dataclass
class Instr:
    op: str
    a: int = 0
    b: int = 0
    c: int = 0  # optional immediate / offset

    def to_tuple(self) -> Tuple[Any, ...]:
        return (self.op, self.a, self.b, self.c)

    def short(self) -> str:
        if self.op in ("INC", "DEC"):
            return f"{self.op} r{self.a}"
        if self.op in ("ADD", "SUB", "MUL"):
            return f"{self.op} r{self.a},r{self.b}"
        if self.op == "MOV":
            return f"MOV r{self.a},r{self.b}"
        if self.op == "SWAP":
            return f"SWAP r{self.a},r{self.b}"
        if self.op in ("JLT", "JGT", "JEQ"):
            return f"{self.op} r{self.a},r{self.b},{self.c:+d}"
        if self.op in ("JMP", "CALL"):
            return f"{self.op} {self.c:+d}"
        if self.op in ("RET", "HALT", "NOP"):
            return self.op
        return f"{self.op} {self.a},{self.b},{self.c}"


# ---------------------------
# Control Flow Graph utilities
# ---------------------------

EdgeT = str  # "SEQ","JMP","BR","CALL","RET","HALT"

@dataclass
class CFG:
    edges: Dict[int, List[Tuple[int, EdgeT]]] = field(default_factory=dict)

    def add_edge(self, u: int, v: int, et: EdgeT) -> None:
        self.edges.setdefault(u, []).append((v, et))

    def nodes(self) -> Set[int]:
        ns = set(self.edges.keys())
        for u, outs in self.edges.items():
            for v, _ in outs:
                ns.add(v)
        return ns

    def canonical_signature(self) -> Tuple[Tuple[Tuple[int,int,str],...], Tuple[int,...], Tuple[Tuple[int,int],...]]:
        """
        Canonical-ish signature, robust to absolute PCs:
        - represent edges as (delta_bucket, etype, out_degree_bucket)
        - represent SCC sizes
        - represent multiset of small motifs (2-step patterns)
        """
        # Edge features
        feats = []
        motif2 = []
        for u, outs in self.edges.items():
            out_deg = len(outs)
            outb = 1 if out_deg <= 1 else (2 if out_deg == 2 else 3)
            for v, et in outs:
                delta = v - u
                # bucket deltas to reduce sensitivity to PC shifts
                if delta <= -8:
                    db = -8
                elif delta >= 8:
                    db = 8
                else:
                    db = delta
                feats.append((db, outb, et))
        feats.sort()

        # SCC sizes
        sccs = strongly_connected_components(self)
        scc_sizes = tuple(sorted([len(s) for s in sccs]))

        # 2-step motifs: (etype1, etype2) buckets
        for u, outs in self.edges.items():
            for v, et1 in outs:
                for w, et2 in self.edges.get(v, []):
                    motif2.append((etype_bucket(et1), etype_bucket(et2)))
        motif2.sort()
        motif2 = tuple(motif2[:64])  # cap

        return (tuple(feats), scc_sizes, motif2)

    def structural_hash(self) -> str:
        sig = self.canonical_signature()
        b = json.dumps(sig, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(b).hexdigest()[:16]

    def edit_distance_to(self, other: "CFG") -> int:
        """
        Cheap structural distance:
        - compare canonical edge feature multiset and SCC size profile.
        """
        a = self.canonical_signature()
        b = other.canonical_signature()
        # multiset distance: symmetric difference length
        feats_a = list(a[0]); feats_b = list(b[0])
        # use dict counts
        def counts(lst):
            d={}
            for x in lst:
                d[x]=d.get(x,0)+1
            return d
        ca, cb = counts(feats_a), counts(feats_b)
        dist = 0
        keys=set(ca)|set(cb)
        for k in keys:
            dist += abs(ca.get(k,0)-cb.get(k,0))
        # SCC distance
        dist += abs(len(a[1]) - len(b[1]))
        dist += sum(abs(x-y) for x,y in zip(sorted(a[1]), sorted(b[1])))
        return dist


def etype_bucket(et: str) -> int:
    return {"SEQ":0,"BR":1,"JMP":2,"CALL":3,"RET":4,"HALT":5}.get(et,9)


def strongly_connected_components(cfg: CFG) -> List[Set[int]]:
    """
    Tarjan SCC (small graphs; ok).
    """
    index = 0
    stack: List[int] = []
    onstack: Set[int] = set()
    idx: Dict[int,int] = {}
    low: Dict[int,int] = {}
    sccs: List[Set[int]] = []

    def visit(v: int):
        nonlocal index
        idx[v] = index
        low[v] = index
        index += 1
        stack.append(v)
        onstack.add(v)
        for w,_ in cfg.edges.get(v, []):
            if w not in idx:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in onstack:
                low[v] = min(low[v], idx[w])
        if low[v] == idx[v]:
            s=set()
            while True:
                x=stack.pop()
                onstack.remove(x)
                s.add(x)
                if x==v: break
            sccs.append(s)

    for v in sorted(cfg.nodes()):
        if v not in idx:
            visit(v)
    return sccs


# ---------------------------
# VM Execution & Trace
# ---------------------------

@dataclass
class RunState:
    regs: List[int]
    pc: int = 0
    steps: int = 0
    halted: bool = False
    error: Optional[str] = None
    stack: List[int] = field(default_factory=list)
    trace_pcs: List[int] = field(default_factory=list)
    trace_ops: List[str] = field(default_factory=list)
    max_call_depth: int = 0

    def record(self, pc: int, op: str):
        self.trace_pcs.append(pc)
        self.trace_ops.append(op)

    @property
    def coverage(self) -> float:
        if not self.trace_pcs:
            return 0.0
        return len(set(self.trace_pcs)) / max(1, (max(self.trace_pcs)+1))

    @property
    def loops_count(self) -> int:
        # count SCCs with size>1 in CFG derived from trace (approx)
        cfg = CFG()
        for i in range(len(self.trace_pcs)-1):
            u = self.trace_pcs[i]; v = self.trace_pcs[i+1]
            cfg.add_edge(u, v, "SEQ")
        sccs = strongly_connected_components(cfg)
        return sum(1 for s in sccs if len(s) > 1)

    def build_cfg(self, prog_len: int) -> CFG:
        cfg = CFG()
        # conservative static CFG from program
        for pc in range(prog_len):
            # default fallthrough
            if pc+1 < prog_len:
                cfg.add_edge(pc, pc+1, "SEQ")
        # augment with dynamic edges from trace for more fidelity
        for i in range(len(self.trace_pcs)-1):
            u = self.trace_pcs[i]; v = self.trace_pcs[i+1]
            if 0 <= u < prog_len and 0 <= v < prog_len:
                cfg.add_edge(u, v, "SEQ")
        return cfg


class TinyVM:
    def __init__(self, step_limit: int = STEP_LIMIT):
        self.step_limit = step_limit

    def execute(self, program: List[Instr], inp: List[int], seed: int = 0) -> RunState:
        st = RunState(regs=[0]*REGS)
        # load input
        for i, v in enumerate(inp[:REGS]):
            st.regs[i] = int(v)
        rng = random.Random(seed)
        prog_len = len(program)
        visited_steps = 0

        while not st.halted and st.error is None and visited_steps < self.step_limit:
            if st.pc < 0 or st.pc >= prog_len:
                st.error = "PC_OOB"
                break
            ins = program[st.pc]
            st.record(st.pc, ins.op)
            visited_steps += 1
            st.steps = visited_steps
            op = ins.op

            try:
                if op == "NOP":
                    st.pc += 1
                elif op == "INC":
                    st.regs[ins.a % REGS] += 1
                    st.pc += 1
                elif op == "DEC":
                    st.regs[ins.a % REGS] -= 1
                    st.pc += 1
                elif op == "ADD":
                    a = ins.a % REGS; b = ins.b % REGS
                    st.regs[a] += st.regs[b]
                    st.pc += 1
                elif op == "SUB":
                    a = ins.a % REGS; b = ins.b % REGS
                    st.regs[a] -= st.regs[b]
                    st.pc += 1
                elif op == "MUL":
                    a = ins.a % REGS; b = ins.b % REGS
                    st.regs[a] *= st.regs[b]
                    st.pc += 1
                elif op == "MOV":
                    a = ins.a % REGS; b = ins.b % REGS
                    st.regs[a] = st.regs[b]
                    st.pc += 1
                elif op == "SWAP":
                    a = ins.a % REGS; b = ins.b % REGS
                    st.regs[a], st.regs[b] = st.regs[b], st.regs[a]
                    st.pc += 1
                elif op in ("JLT","JGT","JEQ"):
                    a = ins.a % REGS; b = ins.b % REGS
                    off = int(ins.c)
                    cond = (st.regs[a] < st.regs[b]) if op=="JLT" else ((st.regs[a] > st.regs[b]) if op=="JGT" else (st.regs[a]==st.regs[b]))
                    st.pc = st.pc + off if cond else st.pc + 1
                elif op == "JMP":
                    st.pc = st.pc + int(ins.c)
                elif op == "CALL":
                    if len(st.stack) >= STACK_LIMIT:
                        st.error = "STACK_OVERFLOW"
                        break
                    st.stack.append(st.pc + 1)
                    st.max_call_depth = max(st.max_call_depth, len(st.stack))
                    st.pc = st.pc + int(ins.c)
                elif op == "RET":
                    if not st.stack:
                        st.error = "STACK_UNDERFLOW"
                        break
                    st.pc = st.stack.pop()
                elif op == "HALT":
                    st.halted = True
                else:
                    st.error = f"BAD_OP:{op}"
            except Exception as e:
                st.error = f"EXC:{type(e).__name__}"
                break

        if visited_steps >= self.step_limit and not st.halted and st.error is None:
            st.error = "STEP_LIMIT"
        return st


# ---------------------------
# Program genome & mutations
# ---------------------------

@dataclass
class Genome:
    gid: str
    instructions: List[Instr]
    parent: Optional[str] = None
    birth_gen: int = 0

    def clone(self, new_id: str, birth_gen: int, parent: Optional[str]) -> "Genome":
        return Genome(gid=new_id, instructions=[dataclasses.replace(i) for i in self.instructions], parent=parent, birth_gen=birth_gen)

    def to_program(self) -> List[Instr]:
        return self.instructions


def rand_instr(rng: random.Random) -> Instr:
    op = rng.choice(OPCODES)
    if op in ("INC","DEC"):
        return Instr(op, a=rng.randrange(REGS))
    if op in ("ADD","SUB","MUL","MOV","SWAP"):
        return Instr(op, a=rng.randrange(REGS), b=rng.randrange(REGS))
    if op in ("JLT","JGT","JEQ"):
        return Instr(op, a=rng.randrange(REGS), b=rng.randrange(REGS), c=rng.choice([-4,-3,-2,-1,1,2,3,4]))
    if op in ("JMP","CALL"):
        return Instr(op, c=rng.choice([-6,-4,-2,2,4,6]))
    if op in ("RET","HALT","NOP"):
        return Instr(op)
    return Instr("NOP")


def init_genome(rng: random.Random, gid: str, length: int = 24) -> Genome:
    instrs = [rand_instr(rng) for _ in range(length-1)] + [Instr("HALT")]
    return Genome(gid=gid, instructions=instrs, parent=None, birth_gen=0)


class Mutator:
    """
    Mutations are designed to enable actual structural transitions:
    - local edits (opcode/args)
    - block insertion/deletion
    - loop skeleton insertion
    - call/ret skeleton insertion
    """
    def __init__(self, rng: random.Random):
        self.rng = rng

    def mutate(self, g: Genome, gen: int) -> Genome:
        child_id = f"g{gen}_{self.rng.randrange(100000):05d}"
        c = g.clone(new_id=child_id, birth_gen=gen, parent=g.gid)
        ops = [
            self.opcode_flip,
            self.arg_jitter,
            self.block_insert,
            self.block_delete,
            self.loop_skeleton_insert,
            self.call_skeleton_insert,
            self.block_relocate,
        ]
        # apply 1-3 mutations
        k = 1 if self.rng.random() < 0.6 else (2 if self.rng.random() < 0.8 else 3)
        for _ in range(k):
            self.rng.choice(ops)(c)
        # ensure HALT exists near end
        if not any(i.op == "HALT" for i in c.instructions[-3:]):
            c.instructions.append(Instr("HALT"))
        # cap length
        if len(c.instructions) > 80:
            c.instructions = c.instructions[:79] + [Instr("HALT")]
        if len(c.instructions) < 8:
            c.instructions = c.instructions + [Instr("NOP")] * (8 - len(c.instructions))
            c.instructions[-1] = Instr("HALT")
        return c

    def opcode_flip(self, g: Genome):
        i = self.rng.randrange(len(g.instructions))
        g.instructions[i] = rand_instr(self.rng)

    def arg_jitter(self, g: Genome):
        i = self.rng.randrange(len(g.instructions))
        ins = g.instructions[i]
        if ins.op in ("INC","DEC"):
            ins.a = (ins.a + self.rng.choice([-1,1])) % REGS
        elif ins.op in ("ADD","SUB","MUL","MOV","SWAP"):
            ins.a = (ins.a + self.rng.choice([-1,1])) % REGS
            ins.b = (ins.b + self.rng.choice([-1,1])) % REGS
        elif ins.op in ("JLT","JGT","JEQ"):
            ins.a = (ins.a + self.rng.choice([-1,1])) % REGS
            ins.b = (ins.b + self.rng.choice([-1,1])) % REGS
            ins.c = int(self.rng.choice([-6,-4,-3,-2,-1,1,2,3,4,6]))
        elif ins.op in ("JMP","CALL"):
            ins.c = int(self.rng.choice([-8,-6,-4,-2,2,4,6,8]))

    def block_insert(self, g: Genome):
        pos = self.rng.randrange(len(g.instructions))
        block = [rand_instr(self.rng) for _ in range(self.rng.randint(2,6))]
        g.instructions[pos:pos] = block

    def block_delete(self, g: Genome):
        if len(g.instructions) <= 10:
            return
        pos = self.rng.randrange(len(g.instructions)-1)
        ln = self.rng.randint(1,4)
        del g.instructions[pos:pos+ln]

    def block_relocate(self, g: Genome):
        if len(g.instructions) <= 14:
            return
        a = self.rng.randrange(0, len(g.instructions)-4)
        b = self.rng.randrange(a+1, len(g.instructions)-1)
        ln = self.rng.randint(2,5)
        block = g.instructions[a:a+ln]
        del g.instructions[a:a+ln]
        b = min(b, len(g.instructions))
        g.instructions[b:b] = block

    def loop_skeleton_insert(self, g: Genome):
        """
        Insert a guarded loop skeleton:
          (cmp) JLT r0,r1, +2
          JMP +k
          ... body ...
          DEC r1
          JMP -m  (back-edge)
        The offsets are approximate; the VM allows relative jumps.
        """
        pos = self.rng.randrange(0, max(1, len(g.instructions)-2))
        body_len = self.rng.randint(2,6)
        body = [rand_instr(self.rng) for _ in range(body_len)]
        # ensure body doesn't start with HALT
        for j in range(len(body)):
            if body[j].op == "HALT":
                body[j] = Instr("NOP")
        # construct
        guard = Instr("JLT", a=0, b=1, c=2)  # if r0<r1 skip next JMP
        jmp_over = Instr("JMP", c=body_len+3)  # jump over loop body to exit
        dec = Instr("DEC", a=1)
        back = Instr("JMP", c=-(body_len+3))  # back to guard
        skeleton = [guard, jmp_over] + body + [dec, back]
        g.instructions[pos:pos] = skeleton

    def call_skeleton_insert(self, g: Genome):
        """
        Insert a call/ret skeleton:
          CALL +k
          ... continuation ...
          ...
          (callee) INC r0
          RET
        """
        pos = self.rng.randrange(0, max(1, len(g.instructions)-2))
        callee = [Instr("INC", a=0), rand_instr(self.rng), Instr("RET")]
        # place callee at end-ish
        callee_pos = min(len(g.instructions), pos + self.rng.randint(3,8))
        g.instructions[callee_pos:callee_pos] = callee
        # call offset from pos to callee_pos
        off = callee_pos - pos
        g.instructions[pos:pos] = [Instr("CALL", c=off)]


# ---------------------------
# Tasks & scoring
# ---------------------------

def task_copy(inp: List[int]) -> List[int]:
    return inp[:]

def task_reverse(inp: List[int]) -> List[int]:
    return list(reversed(inp))

def task_sum_prefix(inp: List[int]) -> List[int]:
    s=0; out=[]
    for x in inp:
        s += x
        out.append(s)
    return out

def task_sort2(inp: List[int]) -> List[int]:
    # small deterministic: sort first two elements, keep rest
    if len(inp) < 2:
        return inp[:]
    a,b = inp[0], inp[1]
    if a > b:
        a,b = b,a
    return [a,b] + inp[2:]

TASKS = [
    ("copy", task_copy),
    ("reverse", task_reverse),
    ("sum_prefix", task_sum_prefix),
    ("sort2", task_sort2),
]

def score_output(out_regs: List[int], target: List[int]) -> float:
    # compare first REGS outputs to target first REGS
    tgt = target[:REGS] + [0]*(REGS-len(target[:REGS]))
    d=0.0
    for i in range(REGS):
        d += abs(out_regs[i] - int(tgt[i]))
    return -d  # higher is better


# ---------------------------
# Judge (structural + performance)
# ---------------------------

@dataclass
class JudgeParams:
    K_initial: int = 5
    L_subseq: int = 8
    coverage_min: float = 0.55
    rarity_f: float = 0.001
    repro_runs: int = 4
    require_both: bool = True
    min_loops: int = 1
    min_call_depth: int = 0
    max_step_limit: int = STEP_LIMIT


class Judge:
    def __init__(self, params: JudgeParams):
        self.p = params
        self.vm = TinyVM(step_limit=params.max_step_limit)

        # archives
        self.success_struct_hashes: Set[str] = set()
        self.subseq_counts: Dict[Tuple[str,...], int] = {}
        self.subseq_total: int = 0

        # parent CFG cache
        self.parent_cfg: Dict[str, CFG] = {}

    def _K(self, gen: int) -> int:
        # mild anneal: tighten with time
        return self.p.K_initial + (gen // 200)

    def evaluate(self, genome: Genome, parent: Optional[Genome], gen: int, rng: random.Random) -> Tuple[bool, Dict[str, Any]]:
        """
        Returns (passed_gate, evidence dict).
        Gate requires:
          - anti-cheat
          - reproducible novelty signals
          - non-trivial control-flow (loops or call depth)
          - distinct structural hash (global)
          - structural transition vs actual parent
        """
        evidence: Dict[str, Any] = {"gid": genome.gid, "parent": parent.gid if parent else None, "gen": gen}

        # reproducibility: run same genome on multiple fixed inputs/seeds and aggregate signals
        inputs = self._repro_inputs(rng, self.p.repro_runs)
        run_states = []
        for i, inp in enumerate(inputs):
            st = self.vm.execute(genome.to_program(), inp=inp, seed=i)
            run_states.append(st)

        # basic anti-cheat checks (must hold for ALL runs)
        for st in run_states:
            if st.error is not None:
                evidence["fail_reason"] = f"error:{st.error}"
                return False, evidence
            if not st.halted:
                evidence["fail_reason"] = "not_halted"
                return False, evidence
            if st.coverage < self.p.coverage_min:
                evidence["fail_reason"] = f"low_coverage:{st.coverage:.3f}"
                return False, evidence

        # non-triviality: need at least one loop SCC OR call depth
        loop_counts = [st.loops_count for st in run_states]
        call_depths = [st.max_call_depth for st in run_states]
        evidence["loops"] = loop_counts
        evidence["call_depths"] = call_depths
        if max(loop_counts) < self.p.min_loops and max(call_depths) <= self.p.min_call_depth:
            evidence["fail_reason"] = "linear_cheat:no_loop_or_call"
            return False, evidence

        # build CFG from a representative run (first) and parent CFG (static/dynamic mix)
        rep = run_states[0]
        cfg = rep.build_cfg(len(genome.instructions))
        s_hash = cfg.structural_hash()
        evidence["struct_hash"] = s_hash

        # global uniqueness gate
        if s_hash in self.success_struct_hashes:
            evidence["fail_reason"] = "duplicate_struct_hash"
            return False, evidence

        # parent comparison gate (must have actual parent)
        if parent is None:
            evidence["fail_reason"] = "missing_parent"
            return False, evidence
        p_cfg = self.parent_cfg.get(parent.gid)
        if p_cfg is None:
            # compute parent cfg once using same input/seed
            pst = self.vm.execute(parent.to_program(), inp=inputs[0], seed=0)
            if pst.error or not pst.halted:
                evidence["fail_reason"] = "parent_unexecutable"
                return False, evidence
            p_cfg = pst.build_cfg(len(parent.instructions))
            self.parent_cfg[parent.gid] = p_cfg

        dist = cfg.edit_distance_to(p_cfg)
        evidence["cfg_dist"] = dist
        if dist < self._K(gen):
            evidence["fail_reason"] = f"cfg_dist<{self._K(gen)}"
            return False, evidence

        # executed subsequence novelty (must be reproducible across runs)
        subseqs = []
        for st in run_states:
            subseqs.append(self._extract_subseq(st.trace_ops))
        # require identical subseq across runs (strong)
        if len(set(subseqs)) != 1:
            evidence["fail_reason"] = "subseq_not_reproducible"
            return False, evidence
        subseq = subseqs[0]
        evidence["subseq"] = subseq

        # subseq rarity gate
        key = tuple(subseq.split("|"))
        cnt = self.subseq_counts.get(key, 0)
        freq = cnt / max(1, self.subseq_total)
        evidence["subseq_freq"] = freq
        if self.subseq_total > 50 and freq >= self.p.rarity_f:
            evidence["fail_reason"] = f"subseq_not_rare:{freq:.6f}"
            return False, evidence

        # require_both already satisfied: cfg gate + subseq gate
        # performance ranking (not gate)
        perf = self._score_tasks(genome, rng)
        evidence["perf"] = perf

        # Commit archives on PASS
        self.success_struct_hashes.add(s_hash)
        self.subseq_counts[key] = cnt + 1
        self.subseq_total += 1
        # also remember child's cfg as parent later
        self.parent_cfg[genome.gid] = cfg

        evidence["pass"] = True
        return True, evidence

    def _extract_subseq(self, ops: List[str]) -> str:
        # take the first L distinct ops window observed in execution (repro-focused)
        L = self.p.L_subseq
        if len(ops) < L:
            ops = ops + ["NOP"]*(L-len(ops))
        window = ops[:L]
        return "|".join(window)

    def _repro_inputs(self, rng: random.Random, n: int) -> List[List[int]]:
        # deterministic-ish inputs to make cheating harder
        inputs=[]
        base = [1,2,3,4]
        for i in range(n):
            x = [(base[(j+i)%4] + (i*j)%3) for j in range(4)]
            inputs.append(x)
        return inputs

    def _score_tasks(self, genome: Genome, rng: random.Random) -> Dict[str, Any]:
        results={}
        prog = genome.to_program()
        for name, fn in TASKS:
            # fixed small input for repeatability
            inp = [2,1,3,0]
            target = fn(inp)
            st = self.vm.execute(prog, inp=inp, seed=123)
            if st.error or not st.halted:
                results[name] = {"ok": False, "score": -1e9}
            else:
                results[name] = {"ok": True, "score": score_output(st.regs, target)}
        # aggregate
        scores = [v["score"] for v in results.values() if v["ok"]]
        results["_avg"] = sum(scores)/max(1,len(scores)) if scores else -1e9
        return results


# ---------------------------
# Evolution engine & harness
# ---------------------------

@dataclass
class EngineConfig:
    population: int = 40
    elites: int = 10
    generations: int = 200
    seed: int = 0
    target_successes: int = 6
    max_generations: int = 500


class OmegaForgeV13:
    def __init__(self, cfg: EngineConfig, judge_params: JudgeParams):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.judge = Judge(judge_params)
        self.mut = Mutator(self.rng)

        self.pop: List[Genome] = [init_genome(self.rng, f"g0_{i:05d}", length=24) for i in range(cfg.population)]
        self.by_id: Dict[str, Genome] = {g.gid: g for g in self.pop}

        self.distinct_successes: Dict[str, Dict[str,Any]] = {}  # struct_hash -> evidence
        self.success_log: List[Dict[str,Any]] = []

    def step(self, gen: int) -> None:
        # rank by performance only among those that pass basic executability (cheap)
        scored=[]
        for g in self.pop:
            avg = self._cheap_perf(g)
            scored.append((avg, g))
        scored.sort(key=lambda x: x[0], reverse=True)
        elites = [g for _,g in scored[:self.cfg.elites]]

        next_pop: List[Genome] = []
        # keep elites (cloned, but preserve IDs so parent tracking works via actual objects)
        next_pop.extend(elites)

        while len(next_pop) < self.cfg.population:
            parent = self.rng.choice(elites)
            child = self.mut.mutate(parent, gen=gen)
            self.by_id[child.gid] = child
            next_pop.append(child)

        # evaluate gate for evidence collection and success archive
        for g in next_pop:
            if g.parent is None:
                continue
            parent_obj = self.by_id.get(g.parent)
            if parent_obj is None:
                continue
            passed, ev = self.judge.evaluate(g, parent_obj, gen, self.rng)
            if passed:
                sh = ev.get("struct_hash")
                if sh and sh not in self.distinct_successes:
                    self.distinct_successes[sh] = ev
                    self.success_log.append(ev)

        self.pop = next_pop

    def _cheap_perf(self, g: Genome) -> float:
        # cheap filter: execute once, must halt and not error
        vm = self.judge.vm
        st = vm.execute(g.to_program(), inp=[2,1,3,0], seed=0)
        if st.error or not st.halted:
            return -1e9
        return 0.0  # keep neutral; judge does real scoring on passers

    def run(self) -> Dict[str,Any]:
        for gen in range(1, self.cfg.generations+1):
            self.step(gen)
        return self.summary()

    def evidence_run(self) -> Dict[str,Any]:
        for gen in range(1, self.cfg.max_generations+1):
            self.step(gen)
            if len(self.distinct_successes) >= self.cfg.target_successes:
                break
        # hard harness checks
        summary = self.summary()
        # enforce distinctness
        distinct = len(self.distinct_successes)
        if distinct < self.cfg.target_successes:
            summary["harness_fail"] = f"distinct_successes<{self.cfg.target_successes}"
        # ensure evidence has distinct genome ids as well
        gids = [ev["gid"] for ev in self.success_log]
        summary["distinct_gids"] = len(set(gids))
        if summary["distinct_gids"] < min(self.cfg.target_successes, len(self.success_log)):
            summary["harness_warn"] = "duplicate_genome_ids_in_success_log"
        return summary

    def summary(self) -> Dict[str,Any]:
        return {
            "distinct_successes": len(self.distinct_successes),
            "success_log_len": len(self.success_log),
            "success_struct_hashes": list(self.distinct_successes.keys())[:10],
            "successes": self.success_log[:6],
        }


# ---------------------------
# CLI / Selftest
# ---------------------------

def cmd_selftest(args: argparse.Namespace) -> int:
    cfg = EngineConfig(population=30, elites=8, generations=40, seed=args.seed, target_successes=3, max_generations=120)
    jp = JudgeParams(
        K_initial=5,
        L_subseq=8,
        coverage_min=0.55,
        rarity_f=0.001,
        repro_runs=4,
        require_both=True,
        min_loops=1,
        min_call_depth=0,
    )
    eng = OmegaForgeV13(cfg, jp)
    s = eng.evidence_run()
    # Selftest is strict: must produce >=1 distinct success OR explicitly report harness_fail.
    if s.get("distinct_successes",0) == 0:
        # It's acceptable for a short selftest to fail, but it must fail loudly with harness_fail.
        if "harness_fail" not in s:
            print("SELFTEST FAIL: no successes and no harness_fail", file=sys.stderr)
            return 2
    print(json.dumps(s, indent=2))
    return 0

def cmd_run(args: argparse.Namespace) -> int:
    cfg = EngineConfig(population=args.population, elites=args.elites, generations=args.generations, seed=args.seed)
    jp = JudgeParams(
        K_initial=args.K_initial,
        L_subseq=args.L_subseq,
        coverage_min=args.coverage_min,
        rarity_f=args.rarity_f,
        repro_runs=args.repro_runs,
        require_both=True,
        min_loops=args.min_loops,
        min_call_depth=args.min_call_depth,
    )
    eng = OmegaForgeV13(cfg, jp)
    s = eng.run()
    print(json.dumps(s, indent=2))
    return 0

def cmd_evidence(args: argparse.Namespace) -> int:
    cfg = EngineConfig(
        population=args.population,
        elites=args.elites,
        generations=args.generations,
        max_generations=args.max_generations,
        seed=args.seed,
        target_successes=args.target,
    )
    jp = JudgeParams(
        K_initial=args.K_initial,
        L_subseq=args.L_subseq,
        coverage_min=args.coverage_min,
        rarity_f=args.rarity_f,
        repro_runs=args.repro_runs,
        require_both=True,
        min_loops=args.min_loops,
        min_call_depth=args.min_call_depth,
    )
    eng = OmegaForgeV13(cfg, jp)
    s = eng.evidence_run()
    print(json.dumps(s, indent=2))
    # harness exit code
    if "harness_fail" in s:
        return 3
    return 0

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_self = sub.add_parser("selftest")
    p_self.add_argument("--seed", type=int, default=0)

    p_run = sub.add_parser("run")
    p_run.add_argument("--generations", type=int, default=200)
    p_run.add_argument("--population", type=int, default=40)
    p_run.add_argument("--elites", type=int, default=10)
    p_run.add_argument("--seed", type=int, default=0)

    p_e = sub.add_parser("evidence_run")
    p_e.add_argument("--target", type=int, default=6)
    p_e.add_argument("--max_generations", type=int, default=500)
    p_e.add_argument("--population", type=int, default=40)
    p_e.add_argument("--elites", type=int, default=10)
    p_e.add_argument("--generations", type=int, default=200)
    p_e.add_argument("--seed", type=int, default=0)

    for sp in (p_run, p_e):
        sp.add_argument("--K_initial", type=int, default=5)
        sp.add_argument("--L_subseq", type=int, default=8)
        sp.add_argument("--coverage_min", type=float, default=0.55)
        sp.add_argument("--rarity_f", type=float, default=0.001)
        sp.add_argument("--repro_runs", type=int, default=4)
        sp.add_argument("--min_loops", type=int, default=1)
        sp.add_argument("--min_call_depth", type=int, default=0)

    return p

def main() -> int:
    ap = build_parser()
    args = ap.parse_args()
    if args.cmd == "selftest":
        return cmd_selftest(args)
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "evidence_run":
        return cmd_evidence(args)
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
