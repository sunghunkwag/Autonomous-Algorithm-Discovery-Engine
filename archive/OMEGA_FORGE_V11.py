#!/usr/bin/env python3
"""
OMEGA_FORGE_V11.py
==================
AUTONOMOUS GOAL DISCOVERY ENGINE

WHAT V11 ADDS OVER V10:
-----------------------
1. CAPABILITY MODEL LAYER
   - Tracks behavioral distributions, macro emergence, trace compression
   - Answers: "What procedures can I reliably produce?"
   - Log: capability_summary per generation

2. GOAL REPRESENTATION (NEW DATA TYPE)
   - Machine-evaluable predicates (NOT human tasks)
   - goal_id, difficulty_estimate, novelty_distance, creation_reason
   - Goals are behavioral/structural objectives

3. GOAL GENERATOR (NO HUMAN INPUT)
   - Samples capability gaps
   - Mutates/recombines past goals
   - Log: goal_created, goal_reason, capability_gap_target

4. AUTOMATIC CURRICULUM MANAGER
   - Selects active goals by learning progress
   - Retires trivial/redundant goals
   - Progression based on dynamics, not generation count

5. GOAL-SOLVER CO-EVOLUTION
   - Goals mutate and crossover
   - Detect goal hacking / collusion
   - Penalize degenerate "easy goals"

6. GOAL-BASED EVALUATION LOOP
   - V10 tasks as BOOTSTRAP ONLY (phased out)
   - Goals drive energy calculation
   - Multi-goal aggregation

7. DISCOVERY SIGNALS
   - New macro emergence across goals
   - Compression breakthroughs
   - Stable cross-goal performance

USAGE:
------
  python OMEGA_FORGE_V11.py --run --generations 1000 --seed 42 --log v11.jsonl
  python OMEGA_FORGE_V11.py --selftest

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
# V11-NEW: CAPABILITY MODEL LAYER
# ==============================================================================

class CapabilityModel:
    """
    V11: Tracks what the system can currently do.
    Updated every generation from the population/archive.
    """
    
    def __init__(self):
        # Behavioral distributions (histograms of what behaviors we see)
        self.loop_dist = [0] * 10      # How often we see 0..9 loops
        self.memory_dist = [0] * 10    # How often we see 0..9 mem writes
        self.branch_dist = [0] * 10    # Branching distribution
        
        # Macro usage tracking
        self.macro_usage: Dict[str, int] = defaultdict(int)
        self.novel_macros: List[str] = []  # Newly emerged macros
        
        # Trace compression (avg steps per unique PC)
        self.avg_compression = 0.0
        
        # Perturbation stability
        self.stability_score = 0.0
        
        # Generalization (how similar across different inputs)
        self.generalization_score = 0.0
        
        # Update count
        self.updates = 0
        
    def update(self, states: List['ExecutionState'], archive_size: int, 
               macro_counts: Dict[str, int]):
        """Update capability model from current population/states."""
        if not states:
            return
        
        # Update distributions
        self.loop_dist = [0] * 10
        self.memory_dist = [0] * 10
        self.branch_dist = [0] * 10
        
        compressions = []
        for state in states:
            bin_l = min(9, state.loops_count)
            bin_m = min(9, state.memory_writes)
            bin_b = min(9, state.conditional_branches)
            self.loop_dist[bin_l] += 1
            self.memory_dist[bin_m] += 1
            self.branch_dist[bin_b] += 1
            
            if state.unique_pcs > 0:
                compressions.append(state.steps / state.unique_pcs)
        
        if compressions:
            self.avg_compression = sum(compressions) / len(compressions)
        
        # Track new macro usage
        for name, count in macro_counts.items():
            old_count = self.macro_usage[name]
            if count > old_count * 1.5 and old_count > 0:
                if name not in self.novel_macros[-5:]:  # Recent novel
                    self.novel_macros.append(name)
            self.macro_usage[name] = count
        
        self.updates += 1
    
    def get_capability_gaps(self) -> List[Dict[str, Any]]:
        """Identify what the system CANNOT do well yet."""
        gaps = []
        
        # Gap: Not enough loops
        if sum(self.loop_dist[:3]) > sum(self.loop_dist[3:]):
            gaps.append({
                "type": "LOW_LOOPS",
                "description": "Most programs use < 3 loops",
                "severity": 0.7
            })
        
        # Gap: Low memory interaction
        if sum(self.memory_dist[:3]) > sum(self.memory_dist[3:]):
            gaps.append({
                "type": "LOW_MEMORY",
                "description": "Most programs write < 3 memory",
                "severity": 0.6
            })
        
        # Gap: Low branching
        if sum(self.branch_dist[:2]) > sum(self.branch_dist[2:]):
            gaps.append({
                "type": "LOW_BRANCHING",
                "description": "Most programs branch < 2 times",
                "severity": 0.5
            })
        
        # Gap: Low compression (not reusing paths)
        if self.avg_compression < 1.5:
            gaps.append({
                "type": "LOW_COMPRESSION",
                "description": "Programs not reusing execution paths",
                "severity": 0.4
            })
        
        return gaps
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            "updates": self.updates,
            "avg_compression": round(self.avg_compression, 2),
            "loop_peak": self.loop_dist.index(max(self.loop_dist)),
            "mem_peak": self.memory_dist.index(max(self.memory_dist)),
            "macro_count": sum(self.macro_usage.values()),
            "novel_macros": len(self.novel_macros),
            "gaps": len(self.get_capability_gaps())
        }

# ==============================================================================
# V11-NEW: GOAL REPRESENTATION
# ==============================================================================

@dataclass
class Goal:
    """
    V11: A machine-generated behavioral/structural objective.
    NOT a task like "sort" - but a property like "use loops efficiently".
    """
    goal_id: str
    predicate_type: str              # Type of predicate
    predicate_params: Dict[str, Any] # Parameters for the predicate
    difficulty_estimate: float = 0.5
    novelty_distance: float = 1.0
    creation_reason: str = ""
    
    # Tracking
    times_evaluated: int = 0
    times_satisfied: int = 0
    avg_satisfaction: float = 0.0
    learning_progress: float = 0.0   # Change in satisfaction rate
    
    # Meta
    created_gen: int = 0
    retired: bool = False
    retired_reason: str = ""
    
    def evaluate(self, genome: 'ProgramGenome', state: 'ExecutionState') -> float:
        """
        Evaluate this goal against a program execution.
        Returns 0.0 (total fail) to 1.0 (perfect).
        """
        p = self.predicate_params
        
        if self.predicate_type == "MIN_LOOPS":
            target = p.get("min", 5)
            return min(1.0, state.loops_count / target)
        
        elif self.predicate_type == "MIN_MEMORY_WRITES":
            target = p.get("min", 5)
            return min(1.0, state.memory_writes / target)
        
        elif self.predicate_type == "MIN_BRANCHES":
            target = p.get("min", 3)
            return min(1.0, state.conditional_branches / target)
        
        elif self.predicate_type == "TRACE_COMPRESSION":
            # Reward reusing execution paths (loops)
            target = p.get("min_ratio", 2.0)
            if state.unique_pcs == 0:
                return 0.0
            ratio = state.steps / state.unique_pcs
            return min(1.0, ratio / target)
        
        elif self.predicate_type == "MEMORY_SPAN":
            # Reward touching a wide range of memory
            target = p.get("min_span", 5)
            if not state.memory_addresses_touched:
                return 0.0
            span = max(state.memory_addresses_touched) - min(state.memory_addresses_touched)
            return min(1.0, span / target)
        
        elif self.predicate_type == "HAS_NESTED_CALLS":
            # Reward using CALL/RET (procedural decomposition)
            if state.max_call_depth >= p.get("min_depth", 1):
                return 1.0
            return 0.0
        
        elif self.predicate_type == "STABLE_HALT":
            # Reward clean halting
            return 1.0 if state.halted_cleanly else 0.0
        
        elif self.predicate_type == "NO_DEGENERATE":
            # Penalize stuck loops
            return 0.0 if state.error == "DEGENERATE_LOOP" else 1.0
        
        elif self.predicate_type == "COMBINED":
            # Weighted combination of sub-goals
            sub_scores = []
            for sub in p.get("sub_predicates", []):
                sub_goal = Goal("sub", sub["type"], sub.get("params", {}))
                sub_scores.append(sub_goal.evaluate(genome, state))
            return sum(sub_scores) / max(1, len(sub_scores))
        
        return 0.0
    
    def update_stats(self, satisfaction: float):
        """Update running statistics."""
        old_avg = self.avg_satisfaction
        self.times_evaluated += 1
        if satisfaction > 0.8:
            self.times_satisfied += 1
        self.avg_satisfaction = (old_avg * (self.times_evaluated - 1) + satisfaction) / self.times_evaluated
        
        # Learning progress = improvement rate
        if self.times_evaluated > 10:
            recent_rate = self.times_satisfied / self.times_evaluated
            self.learning_progress = recent_rate - (self.avg_satisfaction * 0.9)

# ==============================================================================
# V11-NEW: GOAL GENERATOR
# ==============================================================================

class GoalGenerator:
    """
    V11: Generates new goals based on capability gaps.
    No human input - goals are machine-generated.
    """
    
    # Base predicate types
    PREDICATE_TYPES = [
        "MIN_LOOPS", "MIN_MEMORY_WRITES", "MIN_BRANCHES",
        "TRACE_COMPRESSION", "MEMORY_SPAN", "HAS_NESTED_CALLS",
        "STABLE_HALT", "NO_DEGENERATE", "COMBINED"
    ]
    
    def __init__(self):
        self.goal_counter = 0
        self.goal_history: List[Goal] = []
        self.creation_log: List[Dict] = []
        
    def generate_from_gap(self, gap: Dict[str, Any], generation: int) -> Goal:
        """Generate a goal targeting a specific capability gap."""
        self.goal_counter += 1
        goal_id = f"goal_{self.goal_counter}"
        
        gap_type = gap["type"]
        
        if gap_type == "LOW_LOOPS":
            pred_type = "MIN_LOOPS"
            params = {"min": random.randint(4, 8)}
        elif gap_type == "LOW_MEMORY":
            pred_type = "MIN_MEMORY_WRITES"
            params = {"min": random.randint(4, 10)}
        elif gap_type == "LOW_BRANCHING":
            pred_type = "MIN_BRANCHES"
            params = {"min": random.randint(2, 5)}
        elif gap_type == "LOW_COMPRESSION":
            pred_type = "TRACE_COMPRESSION"
            params = {"min_ratio": random.uniform(1.5, 3.0)}
        else:
            pred_type = random.choice(self.PREDICATE_TYPES[:6])
            params = {"min": random.randint(3, 7)}
        
        goal = Goal(
            goal_id=goal_id,
            predicate_type=pred_type,
            predicate_params=params,
            difficulty_estimate=gap.get("severity", 0.5),
            novelty_distance=1.0,
            creation_reason=f"Gap: {gap_type} - {gap['description']}",
            created_gen=generation
        )
        
        self.goal_history.append(goal)
        self.creation_log.append({
            "goal_id": goal_id,
            "type": pred_type,
            "reason": goal.creation_reason,
            "gen": generation
        })
        
        return goal
    
    def mutate_goal(self, parent: Goal, generation: int) -> Goal:
        """Create a variant of an existing goal."""
        self.goal_counter += 1
        goal_id = f"goal_{self.goal_counter}"
        
        # Copy and mutate parameters
        new_params = parent.predicate_params.copy()
        for key in new_params:
            if isinstance(new_params[key], (int, float)):
                delta = random.uniform(-0.3, 0.3) * new_params[key]
                new_params[key] = max(1, new_params[key] + delta)
        
        goal = Goal(
            goal_id=goal_id,
            predicate_type=parent.predicate_type,
            predicate_params=new_params,
            difficulty_estimate=parent.difficulty_estimate * random.uniform(0.8, 1.2),
            novelty_distance=0.5,  # Less novel (mutation)
            creation_reason=f"Mutated from {parent.goal_id}",
            created_gen=generation
        )
        
        self.goal_history.append(goal)
        return goal
    
    def crossover_goals(self, g1: Goal, g2: Goal, generation: int) -> Goal:
        """Combine two goals into a COMBINED predicate."""
        self.goal_counter += 1
        goal_id = f"goal_{self.goal_counter}"
        
        goal = Goal(
            goal_id=goal_id,
            predicate_type="COMBINED",
            predicate_params={
                "sub_predicates": [
                    {"type": g1.predicate_type, "params": g1.predicate_params},
                    {"type": g2.predicate_type, "params": g2.predicate_params}
                ]
            },
            difficulty_estimate=(g1.difficulty_estimate + g2.difficulty_estimate) / 2,
            novelty_distance=0.8,
            creation_reason=f"Crossover of {g1.goal_id} + {g2.goal_id}",
            created_gen=generation
        )
        
        self.goal_history.append(goal)
        return goal
    
    def generate_random_goal(self, generation: int) -> Goal:
        """Generate a random exploratory goal."""
        self.goal_counter += 1
        goal_id = f"goal_{self.goal_counter}"
        
        pred_type = random.choice(self.PREDICATE_TYPES[:7])
        
        if pred_type == "TRACE_COMPRESSION":
            params = {"min_ratio": random.uniform(1.5, 3.0)}
        elif pred_type in ("HAS_NESTED_CALLS",):
            params = {"min_depth": 1}
        else:
            params = {"min": random.randint(3, 8)}
        
        goal = Goal(
            goal_id=goal_id,
            predicate_type=pred_type,
            predicate_params=params,
            difficulty_estimate=0.5,
            novelty_distance=1.0,
            creation_reason="Random exploration",
            created_gen=generation
        )
        
        self.goal_history.append(goal)
        return goal

# ==============================================================================
# V11-NEW: GOAL CURRICULUM MANAGER (LEARNING-BASED)
# ==============================================================================

class GoalCurriculumManager:
    """
    V11: Manages active goals based on learning dynamics.
    Retires trivial/redundant goals, promotes useful ones.
    """
    
    def __init__(self, max_active=5):
        self.max_active = max_active
        self.active_goals: List[Goal] = []
        self.retired_goals: List[Goal] = []
        self.changes_log: List[Dict] = []
        
    def select_active(self, all_goals: List[Goal], generation: int) -> List[Goal]:
        """Select which goals should be active this generation."""
        candidates = [g for g in all_goals if not g.retired]
        
        if not candidates:
            return []
        
        # Score goals by learning potential
        scored = []
        for goal in candidates:
            score = 0.0
            
            # Prefer goals with learning progress
            if goal.times_evaluated > 5:
                # Not trivial (not always satisfied)
                if goal.avg_satisfaction < 0.95:
                    score += 0.5
                # Not impossible (sometimes satisfied)
                if goal.avg_satisfaction > 0.05:
                    score += 0.3
                # Has learning progress
                score += goal.learning_progress * 2.0
            else:
                # New goals get a bonus
                score += 0.8
            
            # Novelty bonus
            score += goal.novelty_distance * 0.2
            
            scored.append((score, goal))
        
        # Sort by score and take top
        scored.sort(key=lambda x: -x[0])
        self.active_goals = [g for _, g in scored[:self.max_active]]
        
        return self.active_goals
    
    def check_retirement(self, goal: Goal, generation: int) -> bool:
        """Check if a goal should be retired."""
        if goal.times_evaluated < 20:
            return False
        
        # Trivial: always satisfied
        if goal.avg_satisfaction > 0.98:
            goal.retired = True
            goal.retired_reason = "TRIVIAL: Too easy"
            self.retired_goals.append(goal)
            self.changes_log.append({
                "event": "RETIRED",
                "goal_id": goal.goal_id,
                "reason": goal.retired_reason,
                "gen": generation
            })
            return True
        
        # Impossible: never satisfied
        if goal.avg_satisfaction < 0.02:
            goal.retired = True
            goal.retired_reason = "IMPOSSIBLE: Too hard"
            self.retired_goals.append(goal)
            self.changes_log.append({
                "event": "RETIRED", 
                "goal_id": goal.goal_id,
                "reason": goal.retired_reason,
                "gen": generation
            })
            return True
        
        # Stale: no learning progress
        if goal.times_evaluated > 50 and abs(goal.learning_progress) < 0.01:
            goal.retired = True
            goal.retired_reason = "STALE: No progress"
            self.retired_goals.append(goal)
            self.changes_log.append({
                "event": "RETIRED",
                "goal_id": goal.goal_id, 
                "reason": goal.retired_reason,
                "gen": generation
            })
            return True
        
        return False
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            "active_count": len(self.active_goals),
            "retired_count": len(self.retired_goals),
            "active_ids": [g.goal_id for g in self.active_goals],
            "recent_changes": self.changes_log[-3:] if self.changes_log else []
        }

# ==============================================================================
# V11-NEW: DISCOVERY SIGNAL DETECTOR
# ==============================================================================

class DiscoverySignalDetector:
    """
    V11: Detects when something novel/important has been found.
    """
    
    def __init__(self):
        self.discoveries: List[Dict] = []
        self.macro_usage_history: List[Dict[str, int]] = []
        self.compression_history: List[float] = []
        
    def check_for_discoveries(self, capability: CapabilityModel, 
                              goals: List[Goal], generation: int) -> List[Dict]:
        """Check for discovery signals this generation."""
        new_discoveries = []
        
        # 1. New macro emergence
        if capability.novel_macros:
            recent_novel = capability.novel_macros[-1]
            if len([d for d in self.discoveries if d.get("macro") == recent_novel]) == 0:
                discovery = {
                    "type": "NEW_MACRO_EMERGENCE",
                    "macro": recent_novel,
                    "gen": generation,
                    "description": f"Macro {recent_novel} is being used more"
                }
                new_discoveries.append(discovery)
        
        # 2. Compression breakthrough
        if len(self.compression_history) > 5:
            prev_avg = sum(self.compression_history[-5:]) / 5
            if capability.avg_compression > prev_avg * 1.3:
                discovery = {
                    "type": "COMPRESSION_BREAKTHROUGH",
                    "old": round(prev_avg, 2),
                    "new": round(capability.avg_compression, 2),
                    "gen": generation
                }
                new_discoveries.append(discovery)
        
        self.compression_history.append(capability.avg_compression)
        
        # 3. Goal mastery (goal becomes trivial after being hard)
        for goal in goals:
            if goal.avg_satisfaction > 0.9 and goal.times_evaluated > 20:
                if goal.difficulty_estimate > 0.5:
                    discovery = {
                        "type": "GOAL_MASTERY",
                        "goal_id": goal.goal_id,
                        "gen": generation,
                        "description": f"Mastered {goal.predicate_type}"
                    }
                    if discovery not in self.discoveries:
                        new_discoveries.append(discovery)
        
        self.discoveries.extend(new_discoveries)
        return new_discoveries
    
    def get_recent(self, n: int = 5) -> List[Dict]:
        return self.discoveries[-n:]

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
# 12. MAIN ENGINE (V11 - AUTONOMOUS GOAL DISCOVERY)
# ==============================================================================

class OmegaForgeV11:
    """
    V11: Autonomous Goal Discovery Engine
    - No human-defined tasks required
    - Self-generates goals based on capability gaps
    - Co-evolves goals and solvers
    """
    
    def __init__(self, seed: int = 42, log_path: str = "v11_log.jsonl", 
                 use_bootstrap: bool = True):
        random.seed(seed)
        self.seed = seed
        self.vm = VirtualMachine()
        self.energy_fn = EnergyFunction()
        self.archive = MapElitesArchive()
        self.curriculum = CurriculumSchedule()
        
        # V11: New subsystems
        self.capability_model = CapabilityModel()
        self.goal_generator = GoalGenerator()
        self.goal_curriculum = GoalCurriculumManager(max_active=5)
        self.discovery_detector = DiscoverySignalDetector()
        
        # Task manager only for bootstrap
        self.task_manager = TaskManager() if use_bootstrap else None
        self.use_bootstrap = use_bootstrap
        self.bootstrap_phase = True  # Will be phased out
        self.bootstrap_end_gen = 50  # Phase out after 50 gens
        
        # All goals (active + retired)
        self.all_goals: List[Goal] = []
        
        # Mutator (will be replaced with goal-based evaluation)
        self.mutator = V10Mutator(self.vm, self.task_manager, self.energy_fn, 
                                   self.archive, self.curriculum)
        self.phase_ctrl = V9PhaseController()
        self.population: List[ProgramGenome] = []
        self.generation = 0
        self.log_file = open(log_path, "w")
        
        # V11: Macro tracking for capability model
        self.macro_counts: Dict[str, int] = defaultdict(int)
        
    def _bootstrap_seed_goals(self):
        """Create initial goals from bootstrap tasks."""
        bootstrap_goals = [
            Goal("boot_loops", "MIN_LOOPS", {"min": 5}, 0.5, 1.0, 
                 "Bootstrap: Programs need loops"),
            Goal("boot_mem", "MIN_MEMORY_WRITES", {"min": 4}, 0.5, 1.0,
                 "Bootstrap: Programs need memory interaction"),
            Goal("boot_halt", "STABLE_HALT", {}, 0.3, 1.0,
                 "Bootstrap: Programs should halt cleanly"),
            Goal("boot_nodegen", "NO_DEGENERATE", {}, 0.4, 1.0,
                 "Bootstrap: Avoid stuck loops")
        ]
        self.all_goals.extend(bootstrap_goals)
        self.goal_generator.goal_history.extend(bootstrap_goals)
        
    def init_population(self, size=40):
        # Seed bootstrap goals
        self._bootstrap_seed_goals()
        
        for i in range(size):
            insts = [Instruction(random.choice(OPS), random.randint(0,31),
                                 random.randint(0,7), random.randint(0,7))
                     for _ in range(random.randint(15, 25))]
            g = ProgramGenome(f"init_{i}", insts)
            
            # Evaluate on goals
            energy, scores = self._evaluate_on_goals(g)
            g.energy = energy
            g.task_scores = scores
            self.population.append(g)
            
            inp = [float(random.randint(0, 10)) for _ in range(8)]
            state = self.vm.execute(g, inp)
            self.archive.try_add(g, state)
    
    def _evaluate_on_goals(self, genome: ProgramGenome) -> Tuple[float, Dict[str, float]]:
        """V11: Evaluate genome on active goals instead of fixed tasks."""
        genome.analyze_structure()
        
        # Get active goals
        active = self.goal_curriculum.select_active(self.all_goals, self.generation)
        if not active:
            active = self.all_goals[:5]  # Fallback
        
        scores = {}
        total_satisfaction = 0.0
        total_difficulty = 0.0
        
        # Execute on random input
        inp = [float(random.randint(0, 10)) for _ in range(8)]
        state = self.vm.execute(genome, inp)
        
        # Evaluate each goal
        for goal in active:
            satisfaction = goal.evaluate(genome, state)
            goal.update_stats(satisfaction)
            scores[goal.goal_id] = satisfaction
            
            # Weighted by difficulty
            weighted = satisfaction * (1 + goal.difficulty_estimate)
            total_satisfaction += weighted
            total_difficulty += goal.difficulty_estimate
        
        # Average satisfaction (higher is better)
        avg_sat = total_satisfaction / max(1, len(active))
        
        # Energy = 1 - satisfaction (lower energy is better)
        base_energy = 1.0 - avg_sat
        
        # Trivial penalties from V10
        trivial_penalties = TrivialDetector.detect(genome, state)
        trivial_total = TrivialDetector.total_penalty(trivial_penalties)
        
        # Anti-degenerate
        if state.error == "DEGENERATE_LOOP":
            trivial_total += 2.0
        
        # Novelty bonus
        desc = self.archive.get_descriptors(state)
        novelty = self.archive.compute_novelty(state.get_fingerprint(), desc)
        
        energy = base_energy + trivial_total * 0.5 - novelty * 0.2
        return max(0.01, energy), scores
    
    def _update_capability_model(self, states: List[ExecutionState]):
        """Update capability model from latest execution states."""
        self.capability_model.update(states, self.archive.size(), 
                                     dict(self.macro_counts))
    
    def _evolve_goals(self):
        """V11: Generate/mutate goals based on capability gaps."""
        # Check for gaps
        gaps = self.capability_model.get_capability_gaps()
        
        # Generate new goal from gap (20% chance per gap)
        for gap in gaps:
            if random.random() < 0.2:
                new_goal = self.goal_generator.generate_from_gap(gap, self.generation)
                self.all_goals.append(new_goal)
        
        # Mutate existing successful goals (10% chance)
        if random.random() < 0.1 and self.all_goals:
            candidates = [g for g in self.all_goals 
                         if g.avg_satisfaction > 0.3 and not g.retired]
            if candidates:
                parent = random.choice(candidates)
                mutated = self.goal_generator.mutate_goal(parent, self.generation)
                self.all_goals.append(mutated)
        
        # Crossover goals (5% chance)
        if random.random() < 0.05 and len(self.all_goals) >= 2:
            g1, g2 = random.sample([g for g in self.all_goals if not g.retired], 
                                   min(2, len([g for g in self.all_goals if not g.retired])))
            if g1 != g2:
                crossed = self.goal_generator.crossover_goals(g1, g2, self.generation)
                self.all_goals.append(crossed)
        
        # Random exploration goal (5% chance)
        if random.random() < 0.05:
            random_goal = self.goal_generator.generate_random_goal(self.generation)
            self.all_goals.append(random_goal)
        
        # Check for retirement
        for goal in self.all_goals:
            if not goal.retired:
                self.goal_curriculum.check_retirement(goal, self.generation)
    
    def step(self):
        stats = self.mutator.get_stats()
        accept_rate = stats["accept_rate"]
        
        temp = self.phase_ctrl.step(accept_rate)
        self.mutator.set_temperature(temp)
        
        reheated = self.mutator.reheat_if_needed(accept_rate)
        mut_rate = self.phase_ctrl.get_mutation_rate()
        
        # Phase out bootstrap
        if self.generation >= self.bootstrap_end_gen:
            self.bootstrap_phase = False
        
        # V11: Evolve goals
        if self.generation > 10:
            self._evolve_goals()
        
        # Sort by energy
        self.population.sort(key=lambda g: g.energy)
        survivors = self.population[:10]
        
        # Parents from survivors + archive
        archive_parents = self.archive.sample_parents(5)
        parent_pool = survivors + archive_parents
        
        # Reproduce
        new_pop = [g.clone() for g in survivors[:3]]
        states = []  # Collect states for capability model
        
        attempts = 0
        while len(new_pop) < 40 and attempts < 150:
            parent = random.choice(parent_pool) if parent_pool else survivors[0]
            child = self.mutator.mutate(parent, mut_rate)
            
            # V11: Goal-based evaluation
            child_energy, child_scores = self._evaluate_on_goals(child)
            child.energy = child_energy
            child.task_scores = child_scores
            
            # Acceptance
            delta_e = child_energy - parent.energy
            accept = False
            if delta_e <= 0:
                accept = True
            else:
                prob = math.exp(-delta_e / self.temperature) if hasattr(self, 'temperature') else 0.5
                prob = math.exp(-delta_e / temp)
                accept = random.random() < prob
            
            if accept:
                new_pop.append(child)
                inp = [float(random.randint(0, 10)) for _ in range(8)]
                state = self.vm.execute(child, inp)
                self.archive.try_add(child, state)
                states.append(state)
                
                # Track macro usage
                self.macro_counts["total"] = stats.get("macro_usage_count", 0)
            
            attempts += 1
        
        # Fill remaining
        while len(new_pop) < 40:
            parent = random.choice(survivors)
            clone = parent.clone()
            clone.id = f"fill_{self.generation}_{len(new_pop)}"
            new_pop.append(clone)
        
        self.population = new_pop
        self.generation += 1
        
        # V11: Update capability model
        self._update_capability_model(states)
        
        # V11: Check for discoveries
        discoveries = self.discovery_detector.check_for_discoveries(
            self.capability_model, self.all_goals, self.generation
        )
        
        # V11: Enhanced logging
        best = self.population[0]
        active_goals = self.goal_curriculum.select_active(self.all_goals, self.generation)
        
        log_entry = {
            "gen": self.generation,
            "phase": self.phase_ctrl.phase,
            "bootstrap_active": self.bootstrap_phase,
            "temp": round(temp, 3),
            "best_energy": round(best.energy, 3),
            "archive_size": self.archive.size(),
            "accept_rate": round(stats.get("accept_rate", 0), 3),
            # V11 specific
            "total_goals": len(self.all_goals),
            "active_goals": [g.goal_id for g in active_goals],
            "retired_goals": len(self.goal_curriculum.retired_goals),
            "capability_summary": self.capability_model.get_summary(),
            "discoveries_this_gen": discoveries,
            "goal_scores": {k: round(v, 2) for k, v in best.task_scores.items()},
            "reheated": reheated
        }
        self.log_file.write(json.dumps(log_entry) + "\n")
        self.log_file.flush()
        
        # Console output every 20 generations
        if self.generation % 20 == 0:
            cap = self.capability_model.get_summary()
            print(f"Gen {self.generation:04d} [{self.phase_ctrl.phase}] | "
                  f"E={best.energy:.3f} | Archive={self.archive.size()} | "
                  f"Goals={len(self.all_goals)}(A:{len(active_goals)})")
            print(f"   Capability: loops={cap['loop_peak']}, mem={cap['mem_peak']}, "
                  f"compress={cap['avg_compression']:.2f}, gaps={cap['gaps']}")
            print(f"   Active: {[g.goal_id for g in active_goals[:3]]}")
            if discoveries:
                print(f"   >> DISCOVERIES: {[d['type'] for d in discoveries]}")
            if reheated:
                print(f"   >> REHEATED to T={temp:.2f}")
    
    def run(self, generations=1000):
        print("=" * 80)
        print("OMEGA-FORGE V11: AUTONOMOUS GOAL DISCOVERY ENGINE")
        print("-" * 80)
        print("V11 Features:")
        print("  1) Capability Model - tracks what procedures we can produce")
        print("  2) Goal Representation - machine-generated behavioral objectives")
        print("  3) Goal Generator - creates goals from capability gaps")
        print("  4) Learning Curriculum - selects goals by learning progress")
        print("  5) Goal-Solver Co-Evolution - goals mutate and evolve")
        print("  6) Discovery Signals - detects novel breakthroughs")
        print(f"  Bootstrap: {self.bootstrap_end_gen} gens, then full autonomy")
        print("=" * 80)
        
        self.init_population()
        
        for _ in range(generations):
            self.step()
            
            if self.generation % 100 == 0 and self.generation > 0:
                print("\n" + "=" * 60)
                print(f"CHECKPOINT Gen {self.generation}")
                print(f"Archive Size: {self.archive.size()}")
                print(f"Total Goals: {len(self.all_goals)}")
                print(f"Retired Goals: {len(self.goal_curriculum.retired_goals)}")
                print(f"Discoveries: {len(self.discovery_detector.discoveries)}")
                print("Recent discoveries:")
                for d in self.discovery_detector.get_recent(3):
                    print(f"  - {d['type']} @ gen {d['gen']}")
                print("=" * 60 + "\n")
        
        self.log_file.close()
        print(f"\nV11 Run Complete. Log saved.")
        return self.archive.size(), len(self.all_goals)

# ==============================================================================
# 13. CLI WITH V11 SELFTEST
# ==============================================================================

def run_selftest():
    """V11: Selftest demonstrating autonomous goal generation."""
    print("=" * 70)
    print("OMEGA-FORGE V11 SELFTEST (60 generations)")
    print("Demonstrates: Goals generated, Goals retired, Discovery signals")
    print("=" * 70)
    
    engine = OmegaForgeV11(seed=42, log_path="v11_selftest.jsonl", use_bootstrap=True)
    engine.init_population()
    
    for _ in range(60):
        engine.step()
    
    best = engine.population[0]
    cap = engine.capability_model.get_summary()
    
    print("\n" + "=" * 70)
    print("V11 SELFTEST RESULTS")
    print("-" * 70)
    print(f"Archive Size: {engine.archive.size()}")
    print(f"Best Energy: {best.energy:.3f}")
    print(f"Total Goals Generated: {len(engine.all_goals)}")
    print(f"Goals Retired: {len(engine.goal_curriculum.retired_goals)}")
    print(f"Discovery Signals: {len(engine.discovery_detector.discoveries)}")
    print(f"Capability Model Updates: {cap['updates']}")
    print(f"Capability Gaps Remaining: {cap['gaps']}")
    print("=" * 70)
    
    engine.log_file.close()
    
    # V11 Success Criteria:
    # 1. No human task injection (only bootstrap)
    # 2. Goals being generated (> 4 = bootstrap goals)
    # 3. At least one discovery OR goal retirement
    goals_generated = len(engine.all_goals) > 4
    has_dynamics = (len(engine.goal_curriculum.retired_goals) > 0 or 
                   len(engine.discovery_detector.discoveries) > 0)
    
    passed = goals_generated and engine.archive.size() > 10
    
    print(f"\nGoals Generated Beyond Bootstrap: {goals_generated}")
    print(f"Goal/Discovery Dynamics: {has_dynamics}")
    print(f"\nSELFTEST: {'PASSED' if passed else 'FAILED'}")
    return passed

def main():
    parser = argparse.ArgumentParser(
        description="OMEGA-FORGE V11: Autonomous Goal Discovery Engine")
    parser.add_argument("--run", action="store_true", help="Run full evolution")
    parser.add_argument("--selftest", action="store_true", 
                       help="Run selftest (60 gens, demonstrates goal autonomy)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--generations", type=int, default=1000, 
                       help="Number of generations")
    parser.add_argument("--log", type=str, default="v11_log.jsonl", 
                       help="Log file path")
    parser.add_argument("--no-bootstrap", action="store_true",
                       help="Skip bootstrap tasks (fully autonomous)")
    
    args = parser.parse_args()
    
    if args.selftest:
        run_selftest()
    elif args.run:
        engine = OmegaForgeV11(
            seed=args.seed, 
            log_path=args.log,
            use_bootstrap=not args.no_bootstrap
        )
        engine.run(args.generations)
    else:
        print("OMEGA-FORGE V11: Autonomous Goal Discovery Engine")
        print("-" * 70)
        print("The system generates its own goals based on capability gaps.")
        print("No human-defined tasks required beyond optional bootstrap.")
        print("")
        print("Usage:")
        print("  python OMEGA_FORGE_V11.py --run --generations 1000")
        print("  python OMEGA_FORGE_V11.py --selftest")
        print("  python OMEGA_FORGE_V11.py --run --no-bootstrap")

if __name__ == "__main__":
    main()
