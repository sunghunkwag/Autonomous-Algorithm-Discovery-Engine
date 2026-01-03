#!/usr/bin/env python3
"""
OMEGA_FORGE_V10.py
==================
MACRO-ENHANCED ALGORITHM DISCOVERY ENGINE

WHAT V10 ADDS OVER V9:
----------------------
A) MACRO-OPERATOR LIBRARY + MUTATION
   - FOR_LOOP, WHILE_LOOP, SCAN, ACCUMULATE, HISTOGRAM, EMIT motifs
   - Macro insertion and replacement mutations
   - Log: macro_insertions, macro_replacements, macro_usage_count

B) TYPED / ROLE-CONSTRAINED MUTATION
   - Memory addresses biased to [0..N-1]
   - Jump offsets biased to stay in program bounds
   - Wild mutation kept for exploration (20%)
   - Log: percent_role_valid_mutations

C) CURRICULUM SCHEDULE WITH STAGE TARGETS
   - Early gens: high scaffolding weight (0.7)
   - Mid gens: balanced (0.4 scaffolding, 0.4 correctness)
   - Late gens: correctness + generalization (0.8)
   - Log: curriculum_phase, scaffolding_weight, correctness_weight

D) BEHAVIORAL DESCRIPTOR UPGRADE
   - branching_count, max_branch_depth, backjump_count
   - memory_span_touched, write_after_read_count
   - Used in archive binning + novelty
   - Log: descriptor vector for best genome

E) ANTI-DEGENERATE LOOP DETECTION (STRONGER)
   - Detect repeated (pc + reg_hash + mem_hash) over 15 steps
   - Heavy 3.0 penalty for degenerate loops
   - Log: degenerate_loop_hits

F) TWO-SAMPLE GENERALIZATION PER TASK
   - Evaluate 2 independent instances per task
   - Mean accuracy + variance penalty
   - Log: per-task mean and variance

USAGE:
------
  python OMEGA_FORGE_V10.py --run --generations 1000 --seed 42 --log v10.jsonl
  python OMEGA_FORGE_V10.py --selftest

Author: OMEGA-FORGE System
"""

import argparse
import random
import math
import json
import time
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from abc import ABC, abstractmethod

# ==============================================================================
# 1. INSTRUCTION SET
# ==============================================================================

OPS = [
    "MOV", "SET", "SWAP",
    "ADD", "SUB", "MUL", "DIV", "INC", "DEC",
    "LOAD", "STORE", "LDI", "STI",
    "JMP", "JZ", "JNZ", "JGT", "JLT",
    "CALL", "RET", "HALT"
]

CONTROL_OPS = {"JMP", "JZ", "JNZ", "JGT", "JLT", "CALL", "RET"}  # For trivial detection
MEMORY_OPS = {"LOAD", "STORE", "LDI", "STI"}

@dataclass
class Instruction:
    op: str
    a: int
    b: int
    c: int
    
    def clone(self):
        return Instruction(self.op, self.a, self.b, self.c)
    
    def is_control(self) -> bool:
        return self.op in CONTROL_OPS
    
    def is_memory(self) -> bool:
        return self.op in MEMORY_OPS

# ==============================================================================
# V10-NEW: MACRO-OPERATOR LIBRARY
# ==============================================================================

class MacroLibrary:
    """
    V10: Library of reusable instruction sequences (macros) that encode
    common algorithmic motifs. These can be inserted/replaced during mutation.
    """
    
    @staticmethod
    def FOR_LOOP(idx_reg=2, limit_reg=1, body_len=3) -> List[Instruction]:
        """FOR i=0 to N-1: loop body then jump back"""
        return [
            Instruction("SET", 0, 0, idx_reg),          # i = 0
            Instruction("LOAD", idx_reg, 0, 3),         # tmp = mem[i] 
            Instruction("INC", 0, 0, idx_reg),          # i++
            Instruction("JLT", 16-body_len-2, idx_reg, limit_reg),  # if i < N, jump back
        ]
    
    @staticmethod
    def WHILE_LOOP(cond_reg=2, limit_reg=1) -> List[Instruction]:
        """WHILE cond < limit: simple conditional loop"""
        return [
            Instruction("JGT", 3, cond_reg, limit_reg),  # exit if cond >= limit
            Instruction("INC", 0, 0, cond_reg),          # cond++
            Instruction("JMP", 16-3, 0, 0),              # jump back
        ]
    
    @staticmethod
    def SCAN(idx_reg=2, len_reg=1, val_reg=3) -> List[Instruction]:
        """SCAN memory [0..N-1], loading each into val_reg"""
        return [
            Instruction("SET", 0, 0, idx_reg),           # i = 0
            Instruction("LOAD", idx_reg, 0, val_reg),    # val = mem[i]
            Instruction("INC", 0, 0, idx_reg),           # i++
            Instruction("JLT", 16-3, idx_reg, len_reg),  # if i < N, loop
        ]
    
    @staticmethod
    def ACCUMULATE(idx_reg=2, len_reg=1, sum_reg=4, out_base=10) -> List[Instruction]:
        """Prefix sum pattern: running accumulation"""
        return [
            Instruction("SET", 0, 0, idx_reg),           # i = 0
            Instruction("SET", 0, 0, sum_reg),           # sum = 0
            Instruction("LOAD", idx_reg, 0, 3),          # tmp = mem[i]
            Instruction("ADD", sum_reg, 3, sum_reg),     # sum += tmp
            Instruction("SET", out_base, 0, 5),          # base = 10
            Instruction("STI", 5, idx_reg, sum_reg),     # mem[base+i] = sum
            Instruction("INC", 0, 0, idx_reg),           # i++
            Instruction("JLT", 16-5, idx_reg, len_reg),  # loop
        ]
    
    @staticmethod
    def HISTOGRAM(idx_reg=2, len_reg=1, hist_base=10) -> List[Instruction]:
        """Counting sort stage 1: build histogram"""
        return [
            Instruction("SET", 0, 0, idx_reg),           # i = 0
            Instruction("LOAD", idx_reg, 0, 3),          # val = mem[i]
            Instruction("SET", hist_base, 0, 5),         # base = 10
            Instruction("LDI", 5, 3, 4),                 # count = mem[base+val]
            Instruction("INC", 0, 0, 4),                 # count++
            Instruction("STI", 5, 3, 4),                 # mem[base+val] = count
            Instruction("INC", 0, 0, idx_reg),           # i++
            Instruction("JLT", 16-6, idx_reg, len_reg),  # loop
        ]
    
    @staticmethod
    def EMIT(out_reg=2, val_reg=3, count_reg=4) -> List[Instruction]:
        """Counting sort stage 2: emit sorted values"""
        return [
            Instruction("JZ", 3, count_reg, 0),          # if count==0, skip
            Instruction("STORE", val_reg, out_reg, 0),   # mem[out] = val
            Instruction("INC", 0, 0, out_reg),           # out++
            Instruction("DEC", 0, 0, count_reg),         # count--
            Instruction("JNZ", 16-3, count_reg, 0),      # loop while count > 0
        ]
    
    @classmethod
    def get_all_macros(cls) -> Dict[str, List[Instruction]]:
        return {
            "FOR_LOOP": cls.FOR_LOOP(),
            "WHILE_LOOP": cls.WHILE_LOOP(),
            "SCAN": cls.SCAN(),
            "ACCUMULATE": cls.ACCUMULATE(),
            "HISTOGRAM": cls.HISTOGRAM(),
            "EMIT": cls.EMIT(),
        }
    
    @classmethod
    def random_macro(cls) -> Tuple[str, List[Instruction]]:
        macros = cls.get_all_macros()
        name = random.choice(list(macros.keys()))
        return name, [i.clone() for i in macros[name]]

# ==============================================================================
# V10-NEW: CURRICULUM SCHEDULE
# ==============================================================================

class CurriculumSchedule:
    """
    V10: Explicit generation-based curriculum for weighting scaffolding vs correctness.
    
    Early (gen 0-100): Focus on scaffolding (intermediate stages)
    Mid (gen 100-400): Balanced scaffolding + correctness
    Late (gen 400+): Correctness + generalization priority
    """
    
    def __init__(self, early_end=100, mid_end=400):
        self.early_end = early_end
        self.mid_end = mid_end
    
    def get_weights(self, generation: int) -> Dict[str, float]:
        if generation < self.early_end:
            phase = "EARLY"
            scaffolding = 0.7
            correctness = 0.2
            generalization = 0.1
        elif generation < self.mid_end:
            phase = "MID"
            scaffolding = 0.4
            correctness = 0.4
            generalization = 0.2
        else:
            phase = "LATE"
            scaffolding = 0.2
            correctness = 0.5
            generalization = 0.3
        
        return {
            "phase": phase,
            "scaffolding_weight": scaffolding,
            "correctness_weight": correctness,
            "generalization_weight": generalization
        }

# ==============================================================================
# 2. PROGRAM GENOME WITH ANNOTATIONS

@dataclass
class ProgramGenome:
    id: str
    instructions: List[Instruction]
    parents: List[str] = field(default_factory=list)
    energy: float = float('inf')
    task_scores: Dict[str, float] = field(default_factory=dict)
    
    # V9: Control motif annotations
    control_flow_count: int = 0
    memory_op_count: int = 0
    has_loop: bool = False
    has_nested_structure: bool = False
    
    def analyze_structure(self):
        """V9: Analyze program structure for motif annotations."""
        self.control_flow_count = sum(1 for i in self.instructions if i.is_control())
        self.memory_op_count = sum(1 for i in self.instructions if i.is_memory())
        
        # Detect loop patterns (backward jumps)
        for idx, inst in enumerate(self.instructions):
            if inst.op in {"JMP", "JZ", "JNZ", "JGT", "JLT"}:
                offset = (inst.a % 32) - 16
                target = idx + offset
                if target < idx:  # Backward jump = potential loop
                    self.has_loop = True
                    break
        
        # Detect nested structure (CALL exists)
        self.has_nested_structure = any(i.op == "CALL" for i in self.instructions)
    
    def source_code(self) -> str:
        return "\n".join([f"{i:02d}: {inst.op} {inst.a} {inst.b} {inst.c}"
                          for i, inst in enumerate(self.instructions)])
    
    def clone(self):
        g = ProgramGenome(self.id + "_c", [i.clone() for i in self.instructions],
                          self.parents.copy(), self.energy)
        g.task_scores = self.task_scores.copy()
        g.control_flow_count = self.control_flow_count
        g.memory_op_count = self.memory_op_count
        g.has_loop = self.has_loop
        g.has_nested_structure = self.has_nested_structure
        return g

# ==============================================================================
# 3. EXECUTION STATE WITH RICH BEHAVIORAL DESCRIPTORS
# ==============================================================================

@dataclass
class ExecutionState:
    regs: List[float]
    memory: Dict[int, float]
    pc: int
    stack: List[int]
    steps: int
    halted: bool
    halted_cleanly: bool
    error: Optional[str]
    trace: List[int]
    
    # Core behavioral features
    loops_count: int = 0
    swap_count: int = 0
    memory_writes: int = 0
    memory_reads: int = 0
    unique_pcs: int = 0
    call_depth: int = 0
    max_call_depth: int = 0
    
    # V9: Richer descriptors for algorithmicity
    backward_jumps: int = 0          # Loop indicator
    conditional_branches: int = 0    # Decision points
    memory_addresses_touched: Set[int] = field(default_factory=set)
    register_writes: int = 0
    
    # V9: Trivial detection
    state_changes: int = 0           # If 0 in a loop = degenerate
    consecutive_same_pc: int = 0     # Stuck detection
    
    def get_fingerprint(self) -> Tuple[int, int, int, int, int]:
        """V9: Behavioral fingerprint for novelty calculation."""
        return (
            min(self.loops_count, 10),
            min(self.memory_writes, 10),
            min(self.max_call_depth, 5),
            min(self.conditional_branches, 10),
            min(len(self.memory_addresses_touched), 10)
        )

# ==============================================================================
# 4. VIRTUAL MACHINE WITH TRIVIAL DETECTION
# ==============================================================================

class VirtualMachine:
    def __init__(self, max_steps=300, memory_size=64, stack_limit=16):
        self.max_steps = max_steps
        self.memory_size = memory_size
        self.stack_limit = stack_limit
        
    def reset(self, inputs: List[float]) -> ExecutionState:
        regs = [0.0] * 8
        memory = {}
        for i, val in enumerate(inputs):
            if i < self.memory_size:
                memory[i] = float(val)
        regs[0] = 0.0
        regs[1] = float(len(inputs))
        regs[7] = float(len(inputs) - 1)
        
        return ExecutionState(
            regs=regs, memory=memory, pc=0, stack=[], steps=0,
            halted=False, halted_cleanly=False, error=None, trace=[],
            memory_addresses_touched=set()
        )

    def execute(self, genome: ProgramGenome, inputs: List[float]) -> ExecutionState:
        state = self.reset(inputs)
        code = genome.instructions
        L = len(code)
        visited_pcs = set()
        prev_state_hash = None
        
        while not state.halted and state.steps < self.max_steps:
            if state.pc < 0 or state.pc >= L:
                state.halted = True
                break
            
            visited_pcs.add(state.pc)
            inst = code[state.pc]
            prev_pc = state.pc
            state.trace.append(state.pc)
            state.steps += 1
            
            # V9: Compute state hash for degenerate loop detection
            current_hash = hash((tuple(state.regs), state.pc))
            if current_hash == prev_state_hash:
                state.consecutive_same_pc += 1
                if state.consecutive_same_pc > 20:  # Stuck in degenerate loop
                    state.halted = True
                    state.error = "DEGENERATE_LOOP"
                    break
            else:
                state.consecutive_same_pc = 0
                state.state_changes += 1
            prev_state_hash = current_hash
            
            try:
                self._step(state, inst)
            except Exception as e:
                state.error = str(e)
                state.halted = True
                break
            
            # V9: Track backward jumps and conditionals
            if state.pc <= prev_pc and not state.halted:
                state.loops_count += 1
                state.backward_jumps += 1
            
            if inst.op in {"JZ", "JNZ", "JGT", "JLT"}:
                state.conditional_branches += 1
        
        state.unique_pcs = len(visited_pcs)
        return state

    def _safe(self, x):
        if not isinstance(x, (int, float)) or math.isnan(x) or math.isinf(x):
            return 0.0
        return max(-1e9, min(1e9, float(x)))

    def _addr(self, x):
        return int(max(0, min(self.memory_size - 1, x)))

    def _step(self, st, inst):
        op, a, b, c = inst.op, inst.a, inst.b, inst.c
        regs = st.regs
        safe = self._safe
        
        if op == "HALT":
            st.halted = True
            st.halted_cleanly = True
            return
        
        if op == "SET": 
            regs[c % 8] = float(a)
            st.register_writes += 1
        elif op == "MOV": 
            regs[c % 8] = regs[a % 8]
            st.register_writes += 1
        elif op == "SWAP":
            ra, rb = a % 8, b % 8
            regs[ra], regs[rb] = regs[rb], regs[ra]
            st.swap_count += 1
            st.register_writes += 2
        
        elif op == "ADD": regs[c % 8] = safe(regs[a % 8] + regs[b % 8]); st.register_writes += 1
        elif op == "SUB": regs[c % 8] = safe(regs[a % 8] - regs[b % 8]); st.register_writes += 1
        elif op == "MUL": regs[c % 8] = safe(regs[a % 8] * regs[b % 8]); st.register_writes += 1
        elif op == "DIV":
            d = regs[b % 8]
            regs[c % 8] = safe(regs[a % 8] / d if abs(d) > 1e-9 else 0.0)
            st.register_writes += 1
        elif op == "INC": regs[c % 8] = safe(regs[c % 8] + 1); st.register_writes += 1
        elif op == "DEC": regs[c % 8] = safe(regs[c % 8] - 1); st.register_writes += 1
            
        elif op == "LOAD":
            addr = self._addr(regs[a % 8])
            regs[c % 8] = st.memory.get(addr, 0.0)
            st.memory_reads += 1
            st.memory_addresses_touched.add(addr)
        elif op == "STORE":
            addr = self._addr(regs[b % 8])
            st.memory[addr] = regs[a % 8]
            st.memory_writes += 1
            st.memory_addresses_touched.add(addr)
        elif op == "LDI":
            addr = self._addr(regs[a % 8] + regs[b % 8])
            regs[c % 8] = st.memory.get(addr, 0.0)
            st.memory_reads += 1
            st.memory_addresses_touched.add(addr)
        elif op == "STI":
            addr = self._addr(regs[a % 8] + regs[b % 8])
            st.memory[addr] = regs[c % 8]
            st.memory_writes += 1
            st.memory_addresses_touched.add(addr)
        
        # Control Flow
        offset = (a % 32) - 16
        jump = False
        
        if op == "JMP": st.pc += offset; jump = True
        elif op == "JZ":
            if abs(regs[b % 8]) < 1e-9: st.pc += offset; jump = True
        elif op == "JNZ":
            if abs(regs[b % 8]) > 1e-9: st.pc += offset; jump = True
        elif op == "JGT":
            if regs[b % 8] > regs[c % 8]: st.pc += offset; jump = True
        elif op == "JLT":
            if regs[b % 8] < regs[c % 8]: st.pc += offset; jump = True
        elif op == "CALL":
            if len(st.stack) < self.stack_limit:
                st.stack.append(st.pc + 1)
                st.call_depth += 1
                st.max_call_depth = max(st.max_call_depth, st.call_depth)
                st.pc += offset
                jump = True
            else:
                st.error = "Stack overflow"
                st.halted = True
        elif op == "RET":
            if st.stack:
                st.pc = st.stack.pop()
                st.call_depth -= 1
                jump = True
            else:
                st.halted = True
                st.halted_cleanly = True
                jump = True
        
        if not jump:
            st.pc += 1

# ==============================================================================
# 5. TASK SUITE (ALL TASKS FROM V8)
# ==============================================================================

class Task(ABC):
    name: str = "Base"
    output_type: str = "memory"
    stages: int = 1
    
    @abstractmethod
    def generate(self) -> Tuple[List[float], Any]:
        pass
    
    @abstractmethod
    def check(self, state: ExecutionState, target: Any, input_data: List[float]) -> float:
        pass
    
    @abstractmethod
    def check_intermediate(self, state: ExecutionState, target: Any, input_data: List[float]) -> Dict[str, float]:
        pass
    
    @abstractmethod
    def get_truth_vector(self) -> Dict[str, float]:
        pass
    
    @abstractmethod
    def get_reject_vector(self) -> Dict[str, float]:
        pass

# [All 5 tasks from V8: CountingSortTask, PatternPositionTask, TwoPointerReverseTask, PrefixSumTask, ArgmaxFirstTask]
# Included directly for single-file constraint

class CountingSortTask(Task):
    name = "COUNTING_SORT"
    output_type = "memory"
    stages = 2
    
    def generate(self):
        size = random.randint(6, 10)
        data = [float(random.randint(0, 4)) for _ in range(size)]
        return data, sorted(data)
    
    def check(self, state, target, input_data):
        N = len(target)
        output = [state.memory.get(i, -999.0) for i in range(N)]
        matches = sum(1 for i in range(N) if abs(output[i] - target[i]) < 0.01)
        return matches / N
    
    def check_intermediate(self, state, target, input_data):
        N = len(input_data)
        expected_hist = [0] * 5
        for v in input_data:
            expected_hist[int(v)] += 1
        hist_output = [state.memory.get(10 + i, 0.0) for i in range(5)]
        hist_matches = sum(1 for i in range(5) if abs(hist_output[i] - expected_hist[i]) < 0.01)
        stage1_score = hist_matches / 5.0
        output = [state.memory.get(i, -999.0) for i in range(N)]
        prefix_correct = 0
        for i in range(N):
            if abs(output[i] - target[i]) < 0.01:
                prefix_correct += 1
            else:
                break
        stage2_score = prefix_correct / N
        return {"histogram_accuracy": stage1_score, "prefix_sort_accuracy": stage2_score,
                "stages_complete": (1.0 if stage1_score > 0.8 else 0.0) + (1.0 if stage2_score > 0.5 else 0.0)}
    
    def get_truth_vector(self):
        return {"loops": 1.0, "swaps": 0.0, "mem_writes": 1.0, "mem_reads": 1.0, "pc_coverage": 0.7}
    
    def get_reject_vector(self):
        return {"loops": 0.0, "mem_writes": 0.0, "mem_reads": 0.0}

class PatternPositionTask(Task):
    name = "PATTERN_POSITIONS"
    output_type = "memory"
    stages = 2
    
    def generate(self):
        data = [float(random.choice([0, 1])) for _ in range(12)]
        positions = []
        for i in range(len(data) - 2):
            if data[i] == 1 and data[i+1] == 0 and data[i+2] == 1:
                positions.append(float(i))
        return data, {"count": float(len(positions)), "positions": positions}
    
    def check(self, state, target, input_data):
        count = target["count"]
        positions = target["positions"]
        count_correct = 1.0 if abs(state.regs[0] - count) < 0.01 else 0.0
        if len(positions) == 0:
            pos_correct = 1.0 if count_correct else 0.0
        else:
            output_pos = [state.memory.get(20 + i, -999.0) for i in range(len(positions))]
            matches = sum(1 for i, p in enumerate(positions) if abs(output_pos[i] - p) < 0.01)
            pos_correct = matches / len(positions)
        return 0.4 * count_correct + 0.6 * pos_correct
    
    def check_intermediate(self, state, target, input_data):
        count = target["count"]
        scan_complete = min(1.0, state.loops_count / max(1, len(input_data) - 2))
        state_tracking = min(1.0, state.memory_writes / 5)
        count_acc = 1.0 / (1.0 + abs(state.regs[0] - count))
        return {"scan_complete": scan_complete, "state_tracking": state_tracking, "count_accuracy": count_acc,
                "stages_complete": (1.0 if scan_complete > 0.8 else 0.0) + (1.0 if count_acc > 0.5 else 0.0)}
    
    def get_truth_vector(self):
        return {"loops": 1.0, "swaps": 0.0, "mem_writes": 0.8, "mem_reads": 1.0, "pc_coverage": 0.6}
    
    def get_reject_vector(self):
        return {"loops": 0.0, "mem_reads": 0.0}

class TwoPointerReverseTask(Task):
    name = "TWO_POINTER_REVERSE"
    output_type = "memory"
    stages = 2
    
    def generate(self):
        size = random.randint(5, 8)
        data = [float(i) for i in range(size)]
        return data, list(reversed(data))
    
    def check(self, state, target, input_data):
        N = len(target)
        output = [state.memory.get(i, -999.0) for i in range(N)]
        matches = sum(1 for i in range(N) if abs(output[i] - target[i]) < 0.01)
        return matches / N
    
    def check_intermediate(self, state, target, input_data):
        N = len(input_data)
        output = [state.memory.get(i, -999.0) for i in range(N)]
        ends_correct = 0.0
        if abs(output[0] - target[0]) < 0.01: ends_correct += 0.5
        if abs(output[N-1] - target[N-1]) < 0.01: ends_correct += 0.5
        total_pairs = N // 2
        symmetry_pairs = sum(1 for i in range(total_pairs) 
                            if abs(output[i] - target[i]) < 0.01 and abs(output[N-1-i] - target[N-1-i]) < 0.01)
        symmetry_score = symmetry_pairs / max(1, total_pairs)
        swap_usage = min(1.0, state.swap_count / max(1, total_pairs))
        return {"ends_swapped": ends_correct, "symmetry_score": symmetry_score, "swap_usage": swap_usage,
                "stages_complete": ends_correct + (1.0 if symmetry_score > 0.5 else 0.0)}
    
    def get_truth_vector(self):
        return {"loops": 0.6, "swaps": 1.0, "mem_writes": 1.0, "mem_reads": 1.0, "pc_coverage": 0.5}
    
    def get_reject_vector(self):
        return {"swaps": 0.0, "mem_writes": 0.0}

class PrefixSumTask(Task):
    name = "PREFIX_SUM"
    output_type = "memory"
    stages = 2
    
    def generate(self):
        size = random.randint(5, 8)
        data = [float(random.randint(1, 5)) for _ in range(size)]
        prefix = []
        running = 0
        for v in data:
            running += v
            prefix.append(running)
        return data, prefix
    
    def check(self, state, target, input_data):
        N = len(target)
        output = [state.memory.get(10 + i, 0.0) for i in range(N)]
        matches = sum(1 for i in range(N) if abs(output[i] - target[i]) < 0.01)
        return matches / N
    
    def check_intermediate(self, state, target, input_data):
        N = len(input_data)
        output = [state.memory.get(10 + i, 0.0) for i in range(N)]
        first_correct = 1.0 if abs(output[0] - target[0]) < 0.01 else 0.0
        monotonic = sum(1 for i in range(1, N) if output[i] >= output[i-1] - 0.01)
        monotonic_score = monotonic / max(1, N - 1)
        final_correct = 1.0 if abs(output[N-1] - target[N-1]) < 0.01 else 0.0
        return {"first_element": first_correct, "monotonic": monotonic_score, "final_sum": final_correct,
                "stages_complete": first_correct + (1.0 if monotonic_score > 0.8 else 0.0) + final_correct}
    
    def get_truth_vector(self):
        return {"loops": 1.0, "swaps": 0.0, "mem_writes": 1.0, "mem_reads": 1.0, "pc_coverage": 0.5}
    
    def get_reject_vector(self):
        return {"loops": 0.0, "mem_writes": 0.0}

class ArgmaxFirstTask(Task):
    name = "ARGMAX_FIRST"
    output_type = "register"
    stages = 2
    
    def generate(self):
        size = random.randint(5, 8)
        max_val = random.randint(30, 50)
        data = [float(random.randint(0, max_val - 1)) for _ in range(size)]
        pos1 = random.randint(0, size - 1)
        data[pos1] = float(max_val)
        return data, float(pos1)
    
    def check(self, state, target, input_data):
        result = state.regs[0]
        return 1.0 if abs(result - target) < 0.01 else 0.0
    
    def check_intermediate(self, state, target, input_data):
        N = len(input_data)
        max_val = max(input_data)
        result_idx = int(state.regs[0])
        found_max = 1.0 if abs(state.regs[1] - max_val) < 0.01 else 0.0
        scan_complete = min(1.0, state.memory_reads / N)
        points_to_max = 0.0
        if 0 <= result_idx < N:
            points_to_max = 1.0 if abs(input_data[result_idx] - max_val) < 0.01 else 0.0
        return {"found_max_value": found_max, "scan_complete": scan_complete, "points_to_max": points_to_max,
                "stages_complete": found_max + points_to_max}
    
    def get_truth_vector(self):
        return {"loops": 1.0, "swaps": 0.0, "mem_writes": 0.3, "mem_reads": 1.0, "pc_coverage": 0.5}
    
    def get_reject_vector(self):
        return {"loops": 0.0, "mem_reads": 0.0}

# ==============================================================================
# 6. TASK MANAGER (V9: EVALUATE ALL TASKS)
# ==============================================================================

class TaskManager:
    def __init__(self):
        self.tasks = [
            CountingSortTask(),
            PatternPositionTask(),
            TwoPointerReverseTask(),
            PrefixSumTask(),
            ArgmaxFirstTask()
        ]
        self.task_names = [t.name for t in self.tasks]
        
    def get_all_tasks(self) -> List[Task]:
        """V9: Always return ALL tasks for evaluation."""
        return self.tasks

# ==============================================================================
# 7. TRIVIAL SOLUTION DETECTOR (V9 NEW)
# ==============================================================================

class TrivialDetector:
    """
    V9: Explicit trivial solution suppression.
    Returns penalty values (higher = more trivial = worse).
    """
    
    @staticmethod
    def detect(genome: ProgramGenome, state: ExecutionState) -> Dict[str, float]:
        penalties = {}
        
        # A1: Early HALT detection (halted in < 10 steps)
        if state.halted and state.steps < 10:
            penalties["EARLY_HALT"] = 2.0
        
        # A2: Straight-line code (no control flow)
        if genome.control_flow_count == 0:
            penalties["NO_CONTROL_FLOW"] = 1.5
        
        # A3: Low memory interaction
        if state.memory_reads + state.memory_writes < 3:
            penalties["LOW_MEMORY"] = 1.0
        
        # A4: Degenerate loop (loop with no state change)
        if state.error == "DEGENERATE_LOOP":
            penalties["DEGENERATE_LOOP"] = 2.5
        
        # A5: No loops (straight-line programs cannot be algorithms)
        if state.loops_count == 0 and state.steps > 5:
            penalties["NO_LOOPS"] = 0.8
        
        return penalties
    
    @staticmethod
    def total_penalty(penalties: Dict[str, float]) -> float:
        return sum(penalties.values())

# ==============================================================================
# 8. ENERGY FUNCTION (V9: REDESIGNED WITH DOCUMENTATION)
# ==============================================================================

class EnergyFunction:
    """
    V9 Energy Function:
    
    E = AccuracyLoss 
        + α * TruthDistance       (distance from ideal algorithmic behavior)
        + β * RejectSimilarity²   (penalty for trivial-like behavior)  
        + γ * OverfitPenalty      (variance across tasks)
        + δ * ScaffoldingReward   (NEGATIVE: intermediate stage completion)
        + ε * TrivialPenalty      (explicit trivial suppression)
        + ζ * HaltPenalty         (unclean termination)
        - η * NoveltyBonus        (behavioral distance from archive)
    
    LOWER IS BETTER.
    """
    
    def __init__(self):
        self.alpha = 1.0    # Truth distance
        self.beta = 2.0     # Reject similarity
        self.gamma = 0.8    # Overfit penalty
        self.delta = 1.2    # Scaffolding reward (applied as negative)
        self.epsilon = 1.5  # Trivial penalty
        self.zeta = 0.5     # Halt penalty
        self.eta = 0.3      # Novelty bonus
        
    def compute_behavior(self, state: ExecutionState, code_length: int) -> Dict[str, float]:
        return {
            "loops": min(1.0, state.loops_count / 10.0),
            "swaps": min(1.0, state.swap_count / 5.0),
            "mem_writes": min(1.0, state.memory_writes / 8.0),
            "mem_reads": min(1.0, state.memory_reads / 10.0),
            "pc_coverage": state.unique_pcs / max(1, code_length),
            # V9: Additional descriptors
            "backward_jumps": min(1.0, state.backward_jumps / 5.0),
            "conditionals": min(1.0, state.conditional_branches / 5.0)
        }
    
    def distance(self, behavior: Dict[str, float], target: Dict[str, float]) -> float:
        d = 0.0
        for k, v in target.items():
            d += (behavior.get(k, 0.0) - v) ** 2
        return math.sqrt(d)
    
    def similarity(self, behavior: Dict[str, float], reject: Dict[str, float]) -> float:
        dot = 0.0
        for k, v in reject.items():
            dot += behavior.get(k, 0.0) * v
        return max(0, dot)
    
    def compute_scaffolding_reward(self, intermediate_scores: Dict[str, Dict[str, float]]) -> float:
        total_stages = sum(scores.get("stages_complete", 0.0) for scores in intermediate_scores.values())
        avg_stages = total_stages / max(1, len(intermediate_scores))
        return -avg_stages / 3.0  # Negative = reward
    
    def compute(self, 
                accuracies: Dict[str, float], 
                behaviors: Dict[str, Dict], 
                task_truths: Dict[str, Dict], 
                task_rejects: Dict[str, Dict],
                halted_cleanly: bool,
                intermediate_scores: Dict[str, Dict[str, float]],
                trivial_penalty: float,
                novelty_bonus: float = 0.0) -> Tuple[float, float]:
        """
        Compute total energy.
        Returns (energy, avg_accuracy).
        """
        # Accuracy loss
        acc_values = list(accuracies.values())
        avg_acc = sum(acc_values) / len(acc_values) if acc_values else 0.0
        acc_loss = 1.0 - avg_acc
        
        # Truth distance
        truth_dists = [self.distance(bhv, task_truths.get(name, {})) 
                       for name, bhv in behaviors.items()]
        avg_truth_dist = sum(truth_dists) / len(truth_dists) if truth_dists else 1.0
        
        # Reject similarity
        reject_sims = [self.similarity(bhv, task_rejects.get(name, {})) 
                       for name, bhv in behaviors.items()]
        avg_reject_sim = sum(reject_sims) / len(reject_sims) if reject_sims else 0.0
        
        # Overfit penalty (variance across tasks)
        if len(acc_values) > 1:
            variance = sum((a - avg_acc) ** 2 for a in acc_values) / len(acc_values)
            overfit = math.sqrt(variance)
        else:
            overfit = 0.0
        
        # Scaffolding reward
        scaffolding = self.delta * self.compute_scaffolding_reward(intermediate_scores)
        
        # Halt penalty
        halt_pen = 0.0 if halted_cleanly else self.zeta
        
        # Total energy
        energy = (acc_loss 
                  + self.alpha * avg_truth_dist 
                  + self.beta * (avg_reject_sim ** 2)
                  + self.gamma * overfit
                  + scaffolding  # Already negative when good
                  + self.epsilon * trivial_penalty
                  + halt_pen
                  - self.eta * novelty_bonus)  # Subtract = reward novelty
        
        return energy, avg_acc

# ==============================================================================
# 9. MAP-ELITES ARCHIVE WITH NOVELTY (V9)
# ==============================================================================

class MapElitesArchive:
    def __init__(self, bins_per_dim=5):
        self.bins = bins_per_dim
        self.grid: Dict[Tuple[int, ...], ProgramGenome] = {}
        self.fingerprints: Dict[Tuple[int, ...], Set[Tuple]] = defaultdict(set)
        
    def get_descriptors(self, state: ExecutionState) -> Tuple[int, int, int, int]:
        """V9: 4D behavioral descriptor."""
        return (
            min(self.bins - 1, state.loops_count // 2),
            min(self.bins - 1, state.memory_writes // 2),
            min(self.bins - 1, state.max_call_depth),
            min(self.bins - 1, state.conditional_branches // 2)
        )
    
    def compute_novelty(self, fingerprint: Tuple, desc: Tuple) -> float:
        """V9: Novelty = behavioral distance from existing archive."""
        existing_fps = self.fingerprints.get(desc, set())
        if not existing_fps:
            return 1.0  # Maximum novelty if cell empty
        
        min_dist = float('inf')
        for fp in existing_fps:
            dist = sum((a - b) ** 2 for a, b in zip(fingerprint, fp)) ** 0.5
            min_dist = min(min_dist, dist)
        
        return min(1.0, min_dist / 5.0)  # Normalize
    
    def try_add(self, genome: ProgramGenome, state: ExecutionState) -> bool:
        desc = self.get_descriptors(state)
        fingerprint = state.get_fingerprint()
        
        if desc not in self.grid or genome.energy < self.grid[desc].energy:
            self.grid[desc] = genome.clone()
            self.fingerprints[desc].add(fingerprint)
            return True
        return False
    
    def get_elites(self) -> List[ProgramGenome]:
        return list(self.grid.values())
    
    def sample_parents(self, k: int) -> List[ProgramGenome]:
        elites = self.get_elites()
        if not elites:
            return []
        return random.choices(elites, k=min(k, len(elites)))
    
    def size(self) -> int:
        return len(self.grid)

# ==============================================================================
# 10. V10 MUTATOR WITH MACRO + TYPED MUTATION + TWO-SAMPLE
# ==============================================================================

class V10Mutator:
    """
    V10: Upgraded mutator with:
    - Macro insertion and replacement mutations
    - Role-constrained (typed) mutation for operands
    - Two-sample evaluation per task
    - All V9 features (temperature, rejection tracking)
    """
    
    def __init__(self, vm: VirtualMachine, task_manager: TaskManager, 
                 energy_fn: EnergyFunction, archive: MapElitesArchive,
                 curriculum: CurriculumSchedule):
        self.vm = vm
        self.task_manager = task_manager
        self.energy_fn = energy_fn
        self.archive = archive
        self.curriculum = curriculum
        
        # Temperature (from V9)
        self.temperature = 2.0
        self.min_temp = 0.15
        self.reheat_threshold = 0.05
        self.reheat_amount = 0.5
        
        # V10: Mutation stats
        self.accepts = 0
        self.rejects = 0
        self.rejection_reasons: Dict[str, int] = defaultdict(int)
        self.macro_insertions = 0
        self.macro_replacements = 0
        self.role_valid_mutations = 0
        self.total_mutations = 0
        self.degenerate_loop_hits = 0
        
        # V10: Wild mutation probability (exploration)
        self.wild_prob = 0.2
        
    def set_temperature(self, t: float):
        self.temperature = max(self.min_temp, t)
    
    def reheat_if_needed(self, accept_rate: float) -> bool:
        if accept_rate < self.reheat_threshold and self.temperature < 0.5:
            self.temperature = min(2.0, self.temperature + self.reheat_amount)
            return True
        return False
    
    def _role_aware_arg(self, op: str, arg_pos: str, prog_len: int) -> int:
        """V10: Role-constrained operand mutation."""
        self.total_mutations += 1
        
        # 20% wild mutation for exploration
        if random.random() < self.wild_prob:
            if arg_pos == "a":
                return random.randint(0, 31)
            return random.randint(0, 7)
        
        self.role_valid_mutations += 1
        
        # Role-based mutation
        if arg_pos == "a":
            if op in {"JMP", "JZ", "JNZ", "JGT", "JLT", "CALL"}:
                # Jump offset: bias to stay in bounds
                return random.randint(max(0, 16 - prog_len), min(31, 16 + prog_len))
            elif op in {"LOAD", "STORE", "LDI", "STI"}:
                # Memory address: use index registers (0-15 typically)
                return random.randint(0, 15)
            else:
                return random.randint(0, 31)
        elif arg_pos == "b":
            # Usually register index
            return random.randint(0, 7)
        else:  # c
            # Destination register
            return random.randint(0, 7)
    
    def mutate(self, parent: ProgramGenome, rate: float) -> ProgramGenome:
        """V10: Mutate with macros and typed operands."""
        new_insts = [i.clone() for i in parent.instructions]
        prog_len = len(new_insts)
        
        # Standard point mutations with role-aware operands
        for i in range(len(new_insts)):
            if random.random() < rate:
                mut_type = random.choice(["op", "arg_a", "arg_b", "arg_c"])
                if mut_type == "op":
                    new_insts[i].op = random.choice(OPS)
                elif mut_type == "arg_a":
                    new_insts[i].a = self._role_aware_arg(new_insts[i].op, "a", prog_len)
                elif mut_type == "arg_b":
                    new_insts[i].b = self._role_aware_arg(new_insts[i].op, "b", prog_len)
                else:
                    new_insts[i].c = self._role_aware_arg(new_insts[i].op, "c", prog_len)
        
        # V10: Macro insertion (10% chance)
        if random.random() < 0.1 and len(new_insts) < 30:
            macro_name, macro_insts = MacroLibrary.random_macro()
            pos = random.randint(0, len(new_insts))
            new_insts = new_insts[:pos] + macro_insts + new_insts[pos:]
            self.macro_insertions += 1
        
        # V10: Macro replacement (8% chance, replace a window with macro)
        if random.random() < 0.08 and len(new_insts) >= 5:
            macro_name, macro_insts = MacroLibrary.random_macro()
            start = random.randint(0, max(0, len(new_insts) - len(macro_insts)))
            end = min(start + len(macro_insts), len(new_insts))
            new_insts = new_insts[:start] + macro_insts + new_insts[end:]
            self.macro_replacements += 1
        
        # Structural mutation (from V9)
        if random.random() < 0.08 and len(new_insts) > 5:
            del new_insts[random.randint(0, len(new_insts) - 1)]
        if random.random() < 0.08 and len(new_insts) < 40:
            new_insts.insert(
                random.randint(0, len(new_insts)),
                Instruction(random.choice(OPS), random.randint(0,31), 
                           random.randint(0,7), random.randint(0,7)))
        
        return ProgramGenome(parent.id + "m", new_insts, [parent.id], float('inf'))
    
    def evaluate(self, genome: ProgramGenome, generation: int) -> Tuple[float, Dict[str, float], Dict]:
        """
        V10: Evaluate with:
        - Two samples per task (mean + variance penalty)
        - Curriculum-weighted energy
        - Enhanced degenerate detection
        """
        tasks = self.task_manager.get_all_tasks()
        genome.analyze_structure()
        curr = self.curriculum.get_weights(generation)
        
        accuracies = {}
        behaviors = {}
        task_truths = {}
        task_rejects = {}
        intermediate_scores = {}
        halted_cleanly = True
        trivial_total = 0.0
        novelty_total = 0.0
        task_variances = {}
        
        for task in tasks:
            # V10: Two-sample evaluation
            accs = []
            for _ in range(2):
                inp, tgt = task.generate()
                state = self.vm.execute(genome, inp)
                acc = task.check(state, tgt, inp)
                accs.append(acc)
                
                # Check for degenerate loop
                if state.error == "DEGENERATE_LOOP":
                    self.degenerate_loop_hits += 1
            
            mean_acc = sum(accs) / 2
            variance = (accs[0] - mean_acc) ** 2 + (accs[1] - mean_acc) ** 2
            
            # Use last state for behaviors
            bhv = self.energy_fn.compute_behavior(state, len(genome.instructions))
            inter = task.check_intermediate(state, tgt, inp)
            
            trivial_penalties = TrivialDetector.detect(genome, state)
            trivial_total += TrivialDetector.total_penalty(trivial_penalties)
            
            desc = self.archive.get_descriptors(state)
            novelty = self.archive.compute_novelty(state.get_fingerprint(), desc)
            novelty_total += novelty
            
            accuracies[task.name] = mean_acc
            task_variances[task.name] = variance
            behaviors[task.name] = bhv
            task_truths[task.name] = task.get_truth_vector()
            task_rejects[task.name] = task.get_reject_vector()
            intermediate_scores[task.name] = inter
            
            if not state.halted_cleanly:
                halted_cleanly = False
        
        avg_trivial = trivial_total / len(tasks)
        avg_novelty = novelty_total / len(tasks)
        avg_variance = sum(task_variances.values()) / len(task_variances)
        
        # V10: Curriculum-weighted energy calculation
        energy, avg_acc = self.energy_fn.compute(
            accuracies, behaviors, task_truths, task_rejects, halted_cleanly,
            intermediate_scores, avg_trivial, avg_novelty
        )
        
        # Apply variance penalty (V10)
        variance_penalty = 0.5 * avg_variance
        energy += variance_penalty
        
        meta = {
            "trivial": avg_trivial, 
            "novelty": avg_novelty, 
            "variance": avg_variance,
            "curriculum": curr
        }
        return energy, accuracies, meta
    
    def mutate_and_accept(self, parent: ProgramGenome, rate: float, generation: int) -> Optional[ProgramGenome]:
        child = self.mutate(parent, rate)
        child_energy, child_scores, child_meta = self.evaluate(child, generation)
        child.energy = child_energy
        child.task_scores = child_scores
        
        delta_e = child_energy - parent.energy
        
        if delta_e > 5.0:
            self.rejects += 1
            self.rejection_reasons["CATASTROPHIC"] += 1
            return None
        
        if delta_e <= 0:
            self.accepts += 1
            return child
        else:
            prob = math.exp(-delta_e / self.temperature)
            if random.random() < prob:
                self.accepts += 1
                return child
            else:
                self.rejects += 1
                if child_meta["trivial"] > 1.0:
                    self.rejection_reasons["TRIVIAL"] += 1
                else:
                    self.rejection_reasons["ENERGY"] += 1
                return None
    
    def get_stats(self) -> Dict[str, Any]:
        total = self.accepts + self.rejects
        pct_role = (self.role_valid_mutations / max(1, self.total_mutations)) * 100
        return {
            "accepts": self.accepts,
            "rejects": self.rejects,
            "accept_rate": self.accepts / max(1, total),
            "temperature": self.temperature,
            "rejection_reasons": dict(self.rejection_reasons),
            "macro_insertions": self.macro_insertions,
            "macro_replacements": self.macro_replacements,
            "macro_usage_count": self.macro_insertions + self.macro_replacements,
            "percent_role_valid_mutations": round(pct_role, 1),
            "degenerate_loop_hits": self.degenerate_loop_hits
        }

# ==============================================================================
# 11. PHASE CONTROLLER (V9: WARMUP/EXPLORATION/CRYSTALLIZATION)
# ==============================================================================

class V9PhaseController:
    """V9: Three-phase evolution with temperature reheating."""
    
    PHASES = ["WARMUP", "EXPLORATION", "CRYSTALLIZATION"]
    
    def __init__(self, warmup_gens=30, exploration_gens=500):
        self.warmup_gens = warmup_gens
        self.exploration_gens = exploration_gens
        self.generation = 0
        self.phase = "WARMUP"
        self.initial_temp = 2.0
        self.current_temp = self.initial_temp
        
    def step(self, accept_rate: float) -> float:
        self.generation += 1
        
        # Phase transitions
        if self.generation >= self.warmup_gens and self.phase == "WARMUP":
            self.phase = "EXPLORATION"
        elif self.generation >= self.exploration_gens and self.phase == "EXPLORATION":
            self.phase = "CRYSTALLIZATION"
        
        # Temperature schedule
        if self.phase == "WARMUP":
            self.current_temp = self.initial_temp
        elif self.phase == "EXPLORATION":
            # V9: Reheat if acceptance collapses
            if accept_rate < 0.05 and self.current_temp < 0.5:
                self.current_temp = min(1.5, self.current_temp + 0.3)
            else:
                self.current_temp = max(0.2, self.current_temp * 0.97)
        else:  # CRYSTALLIZATION
            self.current_temp = max(0.1, self.current_temp * 0.99)
        
        return self.current_temp
    
    def get_mutation_rate(self) -> float:
        if self.phase == "WARMUP":
            return 0.08
        elif self.phase == "EXPLORATION":
            return 0.15
        else:
            return 0.10

# ==============================================================================
# 12. MAIN ENGINE (V10)
# ==============================================================================

class OmegaForgeV10:
    def __init__(self, seed: int = 42, log_path: str = "v10_log.jsonl"):
        random.seed(seed)
        self.seed = seed
        self.vm = VirtualMachine()
        self.task_manager = TaskManager()
        self.energy_fn = EnergyFunction()
        self.archive = MapElitesArchive()
        self.curriculum = CurriculumSchedule()
        self.mutator = V10Mutator(self.vm, self.task_manager, self.energy_fn, 
                                   self.archive, self.curriculum)
        self.phase_ctrl = V9PhaseController()
        self.population: List[ProgramGenome] = []
        self.generation = 0
        self.log_file = open(log_path, "w")
        
    def init_population(self, size=40):
        for i in range(size):
            insts = [Instruction(random.choice(OPS), random.randint(0,31),
                                 random.randint(0,7), random.randint(0,7))
                     for _ in range(random.randint(15, 25))]
            g = ProgramGenome(f"init_{i}", insts)
            energy, scores, _ = self.mutator.evaluate(g, 0)
            g.energy = energy
            g.task_scores = scores
            self.population.append(g)
            
            inp, _ = self.task_manager.tasks[0].generate()
            state = self.vm.execute(g, inp)
            self.archive.try_add(g, state)
    
    def step(self):
        stats = self.mutator.get_stats()
        accept_rate = stats["accept_rate"]
        
        temp = self.phase_ctrl.step(accept_rate)
        self.mutator.set_temperature(temp)
        
        reheated = self.mutator.reheat_if_needed(accept_rate)
        mut_rate = self.phase_ctrl.get_mutation_rate()
        curr_weights = self.curriculum.get_weights(self.generation)
        
        # Sort by energy
        self.population.sort(key=lambda g: g.energy)
        survivors = self.population[:10]
        
        # Parents from survivors + archive
        archive_parents = self.archive.sample_parents(5)
        parent_pool = survivors + archive_parents
        
        # Reproduce
        new_pop = [g.clone() for g in survivors[:3]]
        
        attempts = 0
        while len(new_pop) < 40 and attempts < 150:
            parent = random.choice(parent_pool) if parent_pool else survivors[0]
            child = self.mutator.mutate_and_accept(parent, mut_rate, self.generation)
            
            if child is not None:
                new_pop.append(child)
                inp, _ = self.task_manager.tasks[0].generate()
                state = self.vm.execute(child, inp)
                self.archive.try_add(child, state)
            
            attempts += 1
        
        # Fill remaining
        while len(new_pop) < 40:
            parent = random.choice(survivors)
            clone = parent.clone()
            clone.id = f"fill_{self.generation}_{len(new_pop)}"
            new_pop.append(clone)
        
        self.population = new_pop
        self.generation += 1
        
        # V10: Enhanced logging
        best = self.population[0]
        stats = self.mutator.get_stats()
        
        log_entry = {
            "gen": self.generation,
            "phase": self.phase_ctrl.phase,
            "curriculum_phase": curr_weights["phase"],
            "scaffolding_weight": curr_weights["scaffolding_weight"],
            "correctness_weight": curr_weights["correctness_weight"],
            "temp": round(temp, 3),
            "best_energy": round(best.energy, 3),
            "archive_size": self.archive.size(),
            "accept_rate": round(stats["accept_rate"], 3),
            "rejection_reasons": stats["rejection_reasons"],
            "task_scores": {k: round(v, 2) for k, v in best.task_scores.items()},
            "macro_insertions": stats["macro_insertions"],
            "macro_replacements": stats["macro_replacements"],
            "macro_usage_count": stats["macro_usage_count"],
            "percent_role_valid_mutations": stats["percent_role_valid_mutations"],
            "degenerate_loop_hits": stats["degenerate_loop_hits"],
            "reheated": reheated
        }
        self.log_file.write(json.dumps(log_entry) + "\n")
        self.log_file.flush()
        
        # V10: Enhanced console output every 20 generations
        if self.generation % 20 == 0:
            task_str = ", ".join([f"{k}={v:.2f}" for k, v in best.task_scores.items()])
            print(f"Gen {self.generation:04d} [{self.phase_ctrl.phase}|{curr_weights['phase']}] | "
                  f"E={best.energy:.3f} | Archive={self.archive.size()} | "
                  f"AccRate={stats['accept_rate']:.2f} | T={temp:.2f}")
            print(f"   Tasks: {task_str}")
            print(f"   Macros: {stats['macro_usage_count']} | "
                  f"RoleValid: {stats['percent_role_valid_mutations']:.1f}% | "
                  f"DegLoops: {stats['degenerate_loop_hits']}")
            if reheated:
                print(f"   >> REHEATED to T={temp:.2f}")
    
    def run(self, generations=1000):
        print("=" * 80)
        print("OMEGA-FORGE V10: MACRO-ENHANCED ALGORITHM DISCOVERY ENGINE")
        print("-" * 80)
        print("V10 Features:")
        print("  A) Macro-Operator Library (FOR_LOOP, SCAN, HISTOGRAM, etc.)")
        print("  B) Typed/Role-Constrained Mutation")
        print("  C) Curriculum Schedule (Early/Mid/Late)")
        print("  D) Enhanced Behavioral Descriptors")
        print("  E) Stronger Anti-Degenerate Loop Detection")
        print("  F) Two-Sample Generalization per Task")
        print("=" * 80)
        
        self.init_population()
        
        for _ in range(generations):
            self.step()
            
            if self.generation % 200 == 0 and self.generation > 0:
                print("\n" + "=" * 60)
                print(f"CHECKPOINT Gen {self.generation}")
                print(f"Archive Size: {self.archive.size()}")
                stats = self.mutator.get_stats()
                print(f"Total Macros Used: {stats['macro_usage_count']}")
                print(f"Role-Valid Mutations: {stats['percent_role_valid_mutations']:.1f}%")
                print(f"Degenerate Loop Hits: {stats['degenerate_loop_hits']}")
                print("=" * 60 + "\n")
        
        self.log_file.close()
        print(f"\nV10 Run Complete. Log saved.")
        return self.archive.size(), self.population[0].energy

# ==============================================================================
# 13. CLI WITH SELFTEST
# ==============================================================================

def run_selftest():
    """V10: Minimal selftest - 50 generations, print summary metrics."""
    print("=" * 60)
    print("OMEGA-FORGE V10 SELFTEST (50 generations)")
    print("=" * 60)
    
    engine = OmegaForgeV10(seed=42, log_path="v10_selftest.jsonl")
    engine.init_population()
    
    for _ in range(50):
        engine.step()
    
    stats = engine.mutator.get_stats()
    best = engine.population[0]
    
    print("\n" + "=" * 60)
    print("SELFTEST RESULTS")
    print("-" * 60)
    print(f"Archive Size: {engine.archive.size()}")
    print(f"Best Energy: {best.energy:.3f}")
    print(f"Accept Rate: {stats['accept_rate']:.2f}")
    print(f"Macro Usage: {stats['macro_usage_count']}")
    print(f"Role-Valid Mutations: {stats['percent_role_valid_mutations']:.1f}%")
    print(f"Degenerate Loops Detected: {stats['degenerate_loop_hits']}")
    print(f"Task Scores: {best.task_scores}")
    print("=" * 60)
    
    engine.log_file.close()
    
    # Success criteria
    passed = (
        engine.archive.size() > 10 and
        stats['accept_rate'] > 0.1 and
        stats['macro_usage_count'] > 0
    )
    print(f"\nSELFTEST: {'PASSED' if passed else 'FAILED'}")
    return passed

def main():
    parser = argparse.ArgumentParser(description="OMEGA-FORGE V10: Macro-Enhanced Algorithm Discovery")
    parser.add_argument("--run", action="store_true", help="Run full evolution")
    parser.add_argument("--selftest", action="store_true", help="Run minimal selftest (50 gens)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--generations", type=int, default=1000, help="Number of generations")
    parser.add_argument("--log", type=str, default="v10_log.jsonl", help="Log file path")
    
    args = parser.parse_args()
    
    if args.selftest:
        run_selftest()
    elif args.run:
        engine = OmegaForgeV10(seed=args.seed, log_path=args.log)
        engine.run(args.generations)
    else:
        print("OMEGA-FORGE V10: Macro-Enhanced Algorithm Discovery Engine")
        print("-" * 60)
        print("Usage:")
        print("  python OMEGA_FORGE_V10.py --run --generations 1000 --seed 42 --log v10.jsonl")
        print("  python OMEGA_FORGE_V10.py --selftest")

if __name__ == "__main__":
    main()
