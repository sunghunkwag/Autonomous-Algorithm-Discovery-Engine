#!/usr/bin/env python3
"""
OMEGA_FORGE_V13_CLEAN.py
========================
Streaming structural-transition discovery engine (CLEAN edition).

Design goals
------------
1) Crash-safe evidence logging: write JSONL incrementally with flush + fsync.
2) No "fake pass": enforce global uniqueness + real parent tracking.
3) Separate concerns: Engine (search) vs Detector (gate) vs EvidenceWriter (persistence).
4) Selftest is not a benchmark: it validates execution + logging, and does NOT fail just because
   zero successes occurred in a short horizon.

Usage
-----
  python OMEGA_FORGE_V13_CLEAN.py selftest
  python OMEGA_FORGE_V13_CLEAN.py evidence_run --target 6 --max_generations 2000 --out evidence_v13.jsonl
  python OMEGA_FORGE_V13_CLEAN.py run --generations 5000 --log v13_run.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple
from collections import defaultdict, Counter

# ==============================================================================
# 1) Instruction set
# ==============================================================================

OPS = [
    "MOV", "SET", "SWAP",
    "ADD", "SUB", "MUL", "DIV", "INC", "DEC",
    "LOAD", "STORE", "LDI", "STI",
    "JMP", "JZ", "JNZ", "JGT", "JLT",
    "CALL", "RET", "HALT"
]
CONTROL_OPS = {"JMP", "JZ", "JNZ", "JGT", "JLT", "CALL", "RET"}
MEMORY_OPS = {"LOAD", "STORE", "LDI", "STI"}

@dataclass
class Instruction:
    op: str
    a: int = 0
    b: int = 0
    c: int = 0

    def clone(self) -> "Instruction":
        return Instruction(self.op, self.a, self.b, self.c)

    def to_tuple(self) -> Tuple[Any, ...]:
        return (self.op, int(self.a), int(self.b), int(self.c))

# ==============================================================================
# 2) Program genome
# ==============================================================================

@dataclass
class ProgramGenome:
    gid: str
    instructions: List[Instruction]
    parents: List[str] = field(default_factory=list)
    generation: int = 0
    last_score: float = 0.0
    last_cfg_hash: str = ""

    def clone(self) -> "ProgramGenome":
        return ProgramGenome(
            gid=self.gid,
            instructions=[i.clone() for i in self.instructions],
            parents=list(self.parents),
            generation=self.generation,
        )

    def code_hash(self) -> str:
        h = hashlib.sha256()
        for inst in self.instructions:
            h.update(repr(inst.to_tuple()).encode("utf-8"))
        return h.hexdigest()[:16]

    def op_sequence(self) -> List[str]:
        return [i.op for i in self.instructions]

# ==============================================================================
# 3) Execution state + CFG
# ==============================================================================

@dataclass
class ExecutionState:
    regs: List[float]
    memory: Dict[int, float]
    pc: int = 0
    stack: List[int] = field(default_factory=list)
    steps: int = 0
    halted: bool = False
    halted_cleanly: bool = False
    error: Optional[str] = None

    trace: List[int] = field(default_factory=list)
    visited_pcs: Set[int] = field(default_factory=set)

    loops_count: int = 0
    conditional_branches: int = 0
    max_call_depth: int = 0
    memory_reads: int = 0
    memory_writes: int = 0

    def coverage(self, code_len: int) -> float:
        if code_len <= 0:
            return 0.0
        return len(self.visited_pcs) / float(code_len)

    def fingerprint(self) -> Tuple[int, int, int, int, int]:
        return (
            min(self.loops_count, 20),
            min(self.conditional_branches, 20),
            min(self.memory_writes, 50),
            min(self.memory_reads, 50),
            min(self.max_call_depth, 10),
        )

class ControlFlowGraph:
    def __init__(self) -> None:
        self.edges: Set[Tuple[int, int, str]] = set()
        self.nodes: Set[int] = set()

    def add_edge(self, f: int, t: int, ty: str) -> None:
        self.edges.add((int(f), int(t), str(ty)))
        self.nodes.add(int(f))
        self.nodes.add(int(t))

    @staticmethod
    def from_trace(trace: List[int], code_len: int) -> "ControlFlowGraph":
        cfg = ControlFlowGraph()
        if not trace:
            return cfg
        for i in range(len(trace) - 1):
            a = trace[i]
            b = trace[i + 1]
            ty = "SEQ"
            if b <= a:
                ty = "BACK"
            cfg.add_edge(a, b, ty)
        # Add terminal edge for out-of-range halt
        last = trace[-1]
        cfg.nodes.add(last)
        cfg.nodes.add(max(0, min(code_len, last + 1)))
        return cfg

    def canonical_hash(self) -> str:
        # canonical: sorted edges + SCC size multiset
        h = hashlib.sha256()
        for f, t, ty in sorted(self.edges):
            h.update(f"{f}->{t}:{ty};".encode("utf-8"))
        scc_sizes = sorted([len(s) for s in self.sccs()])
        h.update(("SCC:" + ",".join(map(str, scc_sizes))).encode("utf-8"))
        return h.hexdigest()[:16]

    def sccs(self) -> List[FrozenSet[int]]:
        # Kosaraju
        if not self.nodes:
            return []
        adj = defaultdict(list)
        radj = defaultdict(list)
        for f, t, _ in self.edges:
            adj[f].append(t)
            radj[t].append(f)

        visited: Set[int] = set()
        order: List[int] = []

        def dfs1(u: int) -> None:
            if u in visited:
                return
            visited.add(u)
            for v in adj[u]:
                dfs1(v)
            order.append(u)

        for n in list(self.nodes):
            dfs1(n)

        visited.clear()
        comps: List[FrozenSet[int]] = []

        def dfs2(u: int, comp: Set[int]) -> None:
            if u in visited:
                return
            visited.add(u)
            comp.add(u)
            for v in radj[u]:
                dfs2(v, comp)

        for u in reversed(order):
            if u not in visited:
                comp: Set[int] = set()
                dfs2(u, comp)
                # SCC is meaningful if size>1 or has a self-loop
                if len(comp) > 1:
                    comps.append(frozenset(comp))
                else:
                    x = next(iter(comp)) if comp else None
                    if x is not None and any((x, x, ty) in self.edges for ty in ("SEQ", "BACK")):
                        comps.append(frozenset(comp))
        return comps

    def edit_distance_to(self, other: "ControlFlowGraph") -> int:
        # symmetric difference on typed edges
        return len(self.edges ^ other.edges)

# ==============================================================================
# 4) Virtual machine
# ==============================================================================

class VirtualMachine:
    def __init__(self, max_steps: int = 400, memory_size: int = 64, stack_limit: int = 16) -> None:
        self.max_steps = max_steps
        self.memory_size = memory_size
        self.stack_limit = stack_limit

    def reset(self, inputs: List[float]) -> ExecutionState:
        regs = [0.0] * 8
        mem: Dict[int, float] = {}
        for i, v in enumerate(inputs):
            if i < self.memory_size:
                mem[i] = float(v)
        regs[1] = float(len(inputs))
        return ExecutionState(regs=regs, memory=mem)

    def execute(self, genome: ProgramGenome, inputs: List[float]) -> ExecutionState:
        st = self.reset(inputs)
        code = genome.instructions
        L = len(code)

        recent_hashes: List[int] = []
        while not st.halted and st.steps < self.max_steps:
            if st.pc < 0 or st.pc >= L:
                st.halted = True
                st.halted_cleanly = True
                break

            st.visited_pcs.add(st.pc)
            st.trace.append(st.pc)
            prev_pc = st.pc
            inst = code[st.pc]
            st.steps += 1

            # Degenerate loop detection: if state hashes collapse, stop with error
            state_sig = hash((st.pc, tuple(int(x) for x in st.regs[:4]), len(st.stack)))
            recent_hashes.append(state_sig)
            if len(recent_hashes) > 25:
                recent_hashes.pop(0)
                if len(set(recent_hashes)) < 3:
                    st.error = "DEGENERATE_LOOP"
                    st.halted = True
                    break

            try:
                self._step(st, inst)
            except Exception as e:
                st.error = f"VM_ERR:{e.__class__.__name__}"
                st.halted = True
                break

            # Loop + branch stats
            if st.pc <= prev_pc and not st.halted:
                st.loops_count += 1
            if inst.op in {"JZ", "JNZ", "JGT", "JLT"}:
                st.conditional_branches += 1
            st.max_call_depth = max(st.max_call_depth, len(st.stack))

        return st

    def _step(self, st: ExecutionState, inst: Instruction) -> None:
        op, a, b, c = inst.op, inst.a, inst.b, inst.c
        r = st.regs

        def clamp(x: float) -> float:
            if not isinstance(x, (int, float)) or math.isnan(x) or math.isinf(x):
                return 0.0
            return float(max(-1e9, min(1e9, x)))

        def addr(x: float) -> int:
            return int(max(0, min(self.memory_size - 1, int(x))))

        jump = False

        if op == "HALT":
            st.halted = True
            st.halted_cleanly = True
            return

        if op == "SET":
            r[c % 8] = float(a)
        elif op == "MOV":
            r[c % 8] = float(r[a % 8])
        elif op == "SWAP":
            ra, rb = a % 8, b % 8
            r[ra], r[rb] = r[rb], r[ra]
        elif op == "ADD":
            r[c % 8] = clamp(r[a % 8] + r[b % 8])
        elif op == "SUB":
            r[c % 8] = clamp(r[a % 8] - r[b % 8])
        elif op == "MUL":
            r[c % 8] = clamp(r[a % 8] * r[b % 8])
        elif op == "DIV":
            den = r[b % 8]
            r[c % 8] = clamp(r[a % 8] / den) if abs(den) > 1e-9 else 0.0
        elif op == "INC":
            r[c % 8] = clamp(r[c % 8] + 1.0)
        elif op == "DEC":
            r[c % 8] = clamp(r[c % 8] - 1.0)
        elif op == "LOAD":
            idx = addr(r[a % 8])
            st.memory_reads += 1
            r[c % 8] = float(st.memory.get(idx, 0.0))
        elif op == "STORE":
            idx = addr(r[a % 8])
            st.memory_writes += 1
            st.memory[idx] = clamp(r[c % 8])
        elif op == "LDI":
            base = addr(r[a % 8])
            off = addr(r[b % 8])
            st.memory_reads += 1
            r[c % 8] = float(st.memory.get(addr(base + off), 0.0))
        elif op == "STI":
            base = addr(r[a % 8])
            off = addr(r[b % 8])
            st.memory_writes += 1
            st.memory[addr(base + off)] = clamp(r[c % 8])
        elif op == "JMP":
            st.pc += int(a)
            jump = True
        elif op == "JZ":
            if abs(r[a % 8]) < 1e-9:
                st.pc += int(b)
                jump = True
        elif op == "JNZ":
            if abs(r[a % 8]) >= 1e-9:
                st.pc += int(b)
                jump = True
        elif op == "JGT":
            if r[a % 8] > r[b % 8]:
                st.pc += int(c)
                jump = True
        elif op == "JLT":
            if r[a % 8] < r[b % 8]:
                st.pc += int(c)
                jump = True
        elif op == "CALL":
            if len(st.stack) >= self.stack_limit:
                st.error = "STACK_OVERFLOW"
                st.halted = True
                return
            st.stack.append(st.pc + 1)
            st.pc += int(a)
            jump = True
        elif op == "RET":
            if not st.stack:
                st.halted = True
                st.halted_cleanly = True
                jump = True
            else:
                st.pc = st.stack.pop()
                jump = True
        else:
            # Unknown op => halt
            st.error = "UNKNOWN_OP"
            st.halted = True
            return

        if not jump:
            st.pc += 1

# ==============================================================================
# 5) Mutation operators (include structural builders)
# ==============================================================================

class MacroLibrary:
    @staticmethod
    def loop_skeleton(idx_reg: int = 2, limit_reg: int = 1) -> List[Instruction]:
        # i=0 ; if i<limit: body ; i++ ; jump back ; halt path outside
        return [
            Instruction("SET", 0, 0, idx_reg),
            Instruction("JLT", idx_reg, limit_reg, 4),   # jump into body if i < limit
            Instruction("JMP", 6, 0, 0),                # skip body (exit)
            Instruction("INC", 0, 0, idx_reg),          # body: i++
            Instruction("JMP", -3, 0, 0),               # loop back to JLT
        ]

    @staticmethod
    def call_skeleton() -> List[Instruction]:
        # CALL forward to a mini-routine and RET
        return [
            Instruction("CALL", 2, 0, 0),
            Instruction("JMP", 3, 0, 0),
            Instruction("INC", 0, 0, 0),
            Instruction("RET", 0, 0, 0),
        ]

def rand_inst() -> Instruction:
    op = random.choice(OPS)
    return Instruction(op, random.randint(-8, 31), random.randint(0, 7), random.randint(0, 7))

# ==============================================================================
# 6) Detector + evidence writer
# ==============================================================================

@dataclass
class DetectorParams:
    # Target: 0.5–5% successes. Use a curriculum so the search has a gradient early,
    # then harden constraints to avoid "linear cheats".
    K_initial: int = 6               # strict CFG edit distance (post-warmup)
    L_initial: int = 10              # strict active subseq length (post-warmup)
    C_coverage: float = 0.55         # min coverage (post-warmup)
    f_rarity: float = 0.001          # rarity threshold (post-warmup)
    N_repro: int = 4                 # reproducibility trials

    require_both: bool = True        # strict mode requires CFG + subseq
    min_loops: int = 2               # STRICT: Require at least 2 loops (Multi-Stage)
    min_scc: int = 2                 # STRICT: Require at least 2 SCCs (Complex Topology)

    allow_cfg_variants: int = 2      # reproducibility CFG variants
    max_cov_span: float = 0.30       # reproducibility coverage stability
    max_loop_span: int = 5           # reproducibility loop stability

    # Warmup curriculum (first warmup_gens generations)
    warmup_gens: int = 100
    warmup_K: int = 3
    warmup_L: int = 8
    warmup_cov: float = 0.45
    warmup_require_both: bool = True # Strict warmup
    warmup_min_loops: int = 1        # Ban linear code even in warmup
    warmup_min_scc: int = 1          # Ban acyclic graphs even in warmup

class StrictStructuralDetector:
    def __init__(self, params: Optional[DetectorParams] = None) -> None:
        self.p = params or DetectorParams()
        self.parent_cfgs: Dict[str, ControlFlowGraph] = {}
        self.subseq_counts: Counter = Counter()
        self.subseq_total: int = 0
        self.seen_success_hashes: Set[str] = set()

    def _in_warmup(self, gen: int) -> bool:
        return gen <= self.p.warmup_gens

    def _K(self, gen: int) -> int:
        # Curriculum: easier early, strict later.
        if self._in_warmup(gen):
            return max(1, int(self.p.warmup_K))
        return max(3, int(self.p.K_initial))

    def _L(self, gen: int) -> int:
        if self._in_warmup(gen):
            return max(4, int(self.p.warmup_L))
        return max(6, int(self.p.L_initial))

    def _anti_cheat(self, st: ExecutionState, code_len: int, gen: int) -> Tuple[bool, str]:
        if st.error:
            return False, f"ERR:{st.error}"
        if not st.halted_cleanly:
            return False, "DIRTY_HALT"
        cov = st.coverage(code_len)
        if cov < self.p.C_coverage:
            return False, f"LOW_COVERAGE:{cov:.3f}"
        min_loops = self.p.warmup_min_loops if self._in_warmup(gen) else self.p.min_loops
        if st.loops_count < min_loops:
            return False, "NO_LOOPS"
        return True, f"ANTI_OK cov={cov:.3f} loops={st.loops_count}"

    def _repro(self, genome: ProgramGenome, vm: VirtualMachine) -> Tuple[bool, str]:
        cfgs: List[str] = []
        covs: List[float] = []
        loops: List[int] = []
        fixed_inputs = [
            [0.0]*8,
            [1.0]*8,
            [2.0]*8,
            [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0],
        ]
        for i in range(self.p.N_repro):
            inputs = fixed_inputs[i % len(fixed_inputs)]
            st = vm.execute(genome, inputs)
            cfgs.append(ControlFlowGraph.from_trace(st.trace, len(genome.instructions)).canonical_hash())
            covs.append(st.coverage(len(genome.instructions)))
            loops.append(st.loops_count)

        if len(set(cfgs)) > self.p.allow_cfg_variants:
            return False, "CFG_UNSTABLE"
        if max(covs) - min(covs) > self.p.max_cov_span:
            return False, "COV_UNSTABLE"
        if max(loops) - min(loops) > self.p.max_loop_span:
            return False, "LOOP_UNSTABLE"
        return True, f"REPRO_OK N={self.p.N_repro}"

    def evaluate(
        self,
        genome: ProgramGenome,
        parent: Optional[ProgramGenome],
        st: ExecutionState,
        vm: VirtualMachine,
        generation: int,
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        reasons: List[str] = []
        diag: Dict[str, Any] = {}

        ok, msg = self._anti_cheat(st, len(genome.instructions), generation)
        if not ok:
            return False, [f"ANTI_FAIL:{msg}"], diag
        reasons.append(msg)

        cfg = ControlFlowGraph.from_trace(st.trace, len(genome.instructions))
        diag["cfg_hash"] = cfg.canonical_hash()
        # Track CFG for every genome so children can be compared against their parents (prevents deadlock)
        self.parent_cfgs[genome.gid] = cfg
        p_cfg = self.parent_cfgs.get(parent.gid) if parent else None
        if p_cfg is None and parent is not None:
            # Fallback: compute parent's CFG directly (robust even if parent was never a success)
            pst = vm.execute(parent, [1.0] * 8)
            p_cfg = ControlFlowGraph.from_trace(pst.trace, len(parent.instructions))

        cfg_ok = False
        cfg_msg = "CFG_NO_PARENT"
        if p_cfg is not None:
            dist = cfg.edit_distance_to(p_cfg)
            K = self._K(generation)
            cfg_ok = dist >= K
            cfg_msg = f"CFG dist={dist} K={K}"
            diag["cfg_dist"] = dist
        else:
            diag["cfg_dist"] = None

        scc_n = len(cfg.sccs())
        diag["scc_n"] = scc_n
        min_scc = self.p.warmup_min_scc if self._in_warmup(generation) else self.p.min_scc
        if scc_n < min_scc:
            cfg_ok = False
            cfg_msg = "CFG_NO_SCC"

        # subsequence novelty (only executed pcs, contiguous window in instruction index-space)
        L = self._L(generation)
        ops = genome.op_sequence()
        active: List[Tuple[str, ...]] = []
        visited = st.visited_pcs
        for i in range(0, max(0, len(ops) - L + 1)):
            window_pcs = set(range(i, i + L))
            if window_pcs.issubset(visited):
                active.append(tuple(ops[i : i + L]))

        subseq_ok = False
        subseq_msg = "SUBSEQ_NONE"
        if active:
            # rarity by empirical frequency in archive
            for seq in active:
                freq = (self.subseq_counts.get(seq, 0) / max(1, self.subseq_total))
                if freq < self.p.f_rarity:
                    subseq_ok = True
                    subseq_msg = f"SUBSEQ rarity={freq:.6f} L={L}"
                    # Defer archive updates until AFTER full success (CFG+SUBSEQ+REPRO+UNIQUENESS),
                    # otherwise near-misses rapidly poison rarity and can suppress discovery.
                    diag["_candidate_subseq"] = list(seq)
                    diag["subseq"] = list(seq)
                    diag["subseq_freq"] = freq
                    break
        diag["active_subseq_windows"] = len(active)

        # require both or at least one
        require_both = self.p.warmup_require_both if self._in_warmup(generation) else self.p.require_both
        if require_both:
            if not (cfg_ok and subseq_ok and parent is not None):
                return False, [f"REQUIRE_BOTH_FAIL cfg={cfg_ok}({cfg_msg}) subseq={subseq_ok}({subseq_msg})"], diag
        else:
            if not (cfg_ok or subseq_ok):
                return False, [f"NO_STRUCT_CHANGE {cfg_msg}; {subseq_msg}"], diag

        reasons.append(cfg_msg if cfg_ok else cfg_msg)
        reasons.append(subseq_msg if subseq_ok else subseq_msg)

        # reproducibility
        r_ok, r_msg = self._repro(genome, vm)
        if not r_ok:
            return False, [f"REPRO_FAIL:{r_msg}"], diag
        reasons.append(r_msg)

        # global uniqueness on successes (prevents repeated printing of same "success")
        succ_hash = cfg.canonical_hash() + "|" + genome.code_hash()
        if succ_hash in self.seen_success_hashes:
            return False, ["DUP_SUCCESS_HASH"], diag

        self.seen_success_hashes.add(succ_hash)
        diag["success_hash"] = succ_hash

        # Commit subsequence rarity archive ONLY on confirmed success
        if "subseq" in diag:
            key = tuple(diag["subseq"])
            self.subseq_counts[key] = self.subseq_counts.get(key, 0) + 1
            self.subseq_total += 1
        elif "_candidate_subseq" in diag:
            key = tuple(diag["_candidate_subseq"])
            self.subseq_counts[key] = self.subseq_counts.get(key, 0) + 1
            self.subseq_total += 1

        # store cfg for parent tracking
        self.parent_cfgs[genome.gid] = cfg
        return True, reasons, diag

class EvidenceWriter:
    def __init__(self, out_path: str) -> None:
        self.out_path = out_path
        # Always write a header marker so "empty file" is never ambiguous
        self.f = open(out_path, "a", encoding="utf-8", buffering=1)
        self.write({"type": "header", "version": "V13_CLEAN", "note": "jsonl; each line is crash-safe"})
        self.flush_fsync()

    def write(self, obj: Dict[str, Any]) -> None:
        self.f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def flush_fsync(self) -> None:
        self.f.flush()
        try:
            os.fsync(self.f.fileno())
        except Exception:
            pass

    def close(self) -> None:
        try:
            self.flush_fsync()
        finally:
            self.f.close()

# ==============================================================================
# 7) Engine
# ==============================================================================

@dataclass
class EngineConfig:
    pop_size: int = 30
    init_len_min: int = 18
    init_len_max: int = 28
    elite_keep: int = 12
    children_per_elite: int = 2
    max_code_len: int = 80

class OmegaForgeV13:
    def __init__(
        self,
        seed: int = 42,
        detector: Optional[StrictStructuralDetector] = None,
        vm: Optional[VirtualMachine] = None,
        config: Optional[EngineConfig] = None,
    ) -> None:
        random.seed(seed)
        self.seed = seed
        self.vm = vm or VirtualMachine()
        self.detector = detector or StrictStructuralDetector()
        self.cfg = config or EngineConfig()

        self.population: List[ProgramGenome] = []
        self.generation: int = 0
        self.parents_index: Dict[str, ProgramGenome] = {}

    def init_population(self) -> None:
        self.population = []
        for i in range(self.cfg.pop_size):
            L = random.randint(self.cfg.init_len_min, self.cfg.init_len_max)
            insts = [rand_inst() for _ in range(L)]
            g = ProgramGenome(gid=f"init_{i}", instructions=insts, parents=[], generation=0)
            self.population.append(g)
        self._reindex()

    def _reindex(self) -> None:
        self.parents_index = {g.gid: g for g in self.population}

    def _get_parent_obj(self, g: ProgramGenome) -> Optional[ProgramGenome]:
        if not g.parents:
            return None
        pid = g.parents[0]
        return self.parents_index.get(pid)

    def mutate(self, parent: ProgramGenome) -> ProgramGenome:
        child = parent.clone()
        child.generation = self.generation
        child.parents = [parent.gid]
        child.gid = f"g{self.generation}_{random.randint(0, 999999)}"

        # structural mutation mixture
        # 1) splice macro sometimes
        roll = random.random()
        if roll < 0.20 and len(child.instructions) + 5 < self.cfg.max_code_len:
            macro = MacroLibrary.loop_skeleton() if random.random() < 0.7 else MacroLibrary.call_skeleton()
            pos = random.randint(0, len(child.instructions))
            child.instructions[pos:pos] = [m.clone() for m in macro]
        elif roll < 0.45 and child.instructions:
            # replace a random instruction with a control op to encourage CFG change
            pos = random.randint(0, len(child.instructions) - 1)
            op = random.choice(list(CONTROL_OPS))
            child.instructions[pos] = Instruction(op, random.randint(-8, 8), random.randint(0, 7), random.randint(0, 7))
        elif roll < 0.75 and len(child.instructions) < self.cfg.max_code_len:
            # insert random instruction
            pos = random.randint(0, len(child.instructions))
            child.instructions.insert(pos, rand_inst())
        else:
            # delete
            if len(child.instructions) > 6:
                pos = random.randint(0, len(child.instructions) - 1)
                child.instructions.pop(pos)

        return child

    def step(self, writer: Optional[EvidenceWriter] = None) -> Tuple[int, int]:
        self.generation += 1
        successes_this_gen = 0

        # Evaluate all genomes
        for g in self.population:
            parent = self._get_parent_obj(g)
            st = self.vm.execute(g, [1.0] * 8)
            passed, reasons, diag = self.detector.evaluate(g, parent, st, self.vm, self.generation)
            if passed:
                successes_this_gen += 1
                if writer is not None:
                    ev = {
                        "type": "evidence",
                        "gen": self.generation,
                        "gid": g.gid,
                        "parent": parent.gid if parent else None,
                        "code_hash": g.code_hash(),
                        "reasons": reasons,
                        "diag": diag,
                        "metrics": {
                            "steps": st.steps,
                            "coverage": st.coverage(len(g.instructions)),
                            "loops": st.loops_count,
                            "branches": st.conditional_branches,
                            "scc_n": diag.get("scc_n", 0),
                        },
                    }
                    writer.write(ev)
                    writer.flush_fsync()


        # Reproduce: score-based elite selection + CFG-diversity (prevents random drift / stagnation)
        # Score rewards "structural potential" even when not yet passing strict detector gates.
        for g in self.population:
            parent = self._get_parent_obj(g)
            # we already executed VM above in this step's loop, but we don't store st; re-execute cheaply on fixed input
            st2 = self.vm.execute(g, [1.0] * 8)
            cfg2 = ControlFlowGraph.from_trace(st2.trace, len(g.instructions))
            cov = st2.coverage(len(g.instructions))
            scc_n = len(cfg2.sccs())
            # Potential score: coverage + (loops/branches/calls) + SCC bonus; penalize errors / dirty halts
            score = cov + 0.02 * min(st2.loops_count, 50) + 0.01 * min(st2.conditional_branches, 50) + 0.03 * min(st2.max_call_depth, 10) + 0.08 * min(scc_n, 6)
            if st2.error or (not st2.halted_cleanly):
                score -= 0.5
            g.last_score = float(score)
            g.last_cfg_hash = cfg2.canonical_hash()

        # Sort by score
        ranked = sorted(self.population, key=lambda x: x.last_score, reverse=True)

        # Diversity filter: prefer unique CFG hashes among the top band
        elites: List[ProgramGenome] = []
        seen_cfg: Set[str] = set()
        band = ranked[: max(self.cfg.elite_keep * 3, self.cfg.elite_keep)]
        for g in band:
            if len(elites) >= self.cfg.elite_keep:
                break
            if g.last_cfg_hash not in seen_cfg:
                elites.append(g)
                seen_cfg.add(g.last_cfg_hash)

        # If not enough elites due to diversity constraint, fill from ranked
        if len(elites) < self.cfg.elite_keep:
            for g in ranked:
                if len(elites) >= self.cfg.elite_keep:
                    break
                elites.append(g)

        next_pop: List[ProgramGenome] = []
        for e in elites:
            kept = e.clone()
            next_pop.append(kept)
            for _ in range(self.cfg.children_per_elite):
                next_pop.append(self.mutate(e))

        # trim to pop size
        self.population = next_pop[: self.cfg.pop_size]
        self._reindex()

        return successes_this_gen, len(getattr(self.detector, "seen_success_hashes", set()))
# ==============================================================================
# 8) CLI flows
# ==============================================================================

def cmd_selftest(args: argparse.Namespace) -> int:
    """
    Selftest validates:
      - engine executes for N generations without crashing
      - evidence file is created and non-empty (at least header line)
    It does NOT require successes in a short horizon.
    """
    out = args.out or "v13_selftest.jsonl"
    if os.path.exists(out):
        try:
            os.remove(out)
        except Exception:
            pass

    # For selftest, relax params a bit so it is more likely to see at least one success,
    # but still keep anti-cheat and logging correctness.
    p = DetectorParams(
        K_initial=4,
        L_initial=7,
        C_coverage=0.45,
        f_rarity=0.01,
        N_repro=3,
        require_both=True,
        min_loops=1,
        min_scc=1,
    )
    det = StrictStructuralDetector(p)
    eng = OmegaForgeV13(seed=args.seed, detector=det)
    eng.init_population()
    w = EvidenceWriter(out)

    gens = int(args.generations or 200)
    total_success_lines = 0
    try:
        for _ in range(gens):
            succ, _ = eng.step(writer=w)
            # progress
            # Count evidence lines roughly by success count (header already present)
            total_success_lines += succ
            if eng.generation % 10 == 0:
                # "total_evidence_lines" includes only evidence, not header
                print(f"[gen {eng.generation}] successes_this_gen={succ} total_evidence_lines={total_success_lines}", flush=True)
    finally:
        w.close()

    # Validate file exists and has at least 1 line (header)
    if not os.path.exists(out):
        print("SELFTEST_FAIL: evidence file missing", flush=True)
        return 1

    try:
        sz = os.path.getsize(out)
    except Exception:
        sz = 0

    if sz <= 0:
        print("SELFTEST_FAIL: evidence file empty (should contain header)", flush=True)
        return 1

    # Pass criteria: file non-empty + ran gens
    print(f"SELFTEST_OK: ran_gens={gens} evidence_file_bytes={sz} evidence_successes={total_success_lines}", flush=True)
    return 0

def cmd_evidence_run(args: argparse.Namespace) -> int:
    out = args.out or "evidence_v13.jsonl"
    target = int(args.target or 6)
    max_g = int(args.max_generations or 2000)

    # Use default strict params unless user overrides via flags later
    eng = OmegaForgeV13(seed=args.seed)
    eng.init_population()
    w = EvidenceWriter(out)

    found = 0
    try:
        while eng.generation < max_g and found < target:
            succ, _ = eng.step(writer=w)
            found += succ
            if eng.generation % max(1, int(args.report_every or 10)) == 0:
                print(f"[gen {eng.generation}] found={found}/{target} out={out}", flush=True)
    finally:
        w.close()

    print(f"EVIDENCE_RUN_DONE: gens={eng.generation} found={found} out={out}", flush=True)
    return 0

def cmd_run(args: argparse.Namespace) -> int:
    log = args.log or "v13_run.jsonl"
    gens = int(args.generations or 5000)
    eng = OmegaForgeV13(seed=args.seed)
    eng.init_population()
    w = EvidenceWriter(log)
    try:
        for _ in range(gens):
            succ, _ = eng.step(writer=w)
            if eng.generation % max(1, int(args.report_every or 50)) == 0:
                print(f"[gen {eng.generation}] successes_this_gen={succ} log={log}", flush=True)
    finally:
        w.close()
    print(f"RUN_DONE: gens={gens} log={log}", flush=True)
    return 0

def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="OMEGA_FORGE V13 CLEAN (streaming evidence)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("selftest", help="Run crash-safe logging selftest")
    p1.add_argument("--seed", type=int, default=42)
    p1.add_argument("--generations", type=int, default=200)
    p1.add_argument("--out", type=str, default="v13_selftest.jsonl")
    p1.set_defaults(func=cmd_selftest)

    p2 = sub.add_parser("evidence_run", help="Run until N evidence lines are found")
    p2.add_argument("--seed", type=int, default=42)
    p2.add_argument("--target", type=int, default=6)
    p2.add_argument("--max_generations", type=int, default=2000)
    p2.add_argument("--out", type=str, default="evidence_v13.jsonl")
    p2.add_argument("--report_every", type=int, default=10)
    p2.set_defaults(func=cmd_evidence_run)

    p3 = sub.add_parser("run", help="Long run (writes all evidence to log)")
    p3.add_argument("--seed", type=int, default=42)
    p3.add_argument("--generations", type=int, default=5000)
    p3.add_argument("--log", type=str, default="v13_run.jsonl")
    p3.add_argument("--report_every", type=int, default=50)
    p3.set_defaults(func=cmd_run)

    return ap

def main() -> int:
    ap = build_cli()
    args = ap.parse_args()
    return int(args.func(args))

if __name__ == "__main__":
    raise SystemExit(main())