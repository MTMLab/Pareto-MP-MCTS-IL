import os
import sys
import argparse
import copy
import math
import random
import pickle
import csv
import subprocess
import zipfile
import pandas as pd
import numpy as np
import tensorflow as tf

import smiles
import sascorer
import asdi

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.QED import qed

# Pareto utilities
def dominates(a, b):
    """Return True if objective vector ``a`` Pareto-dominates ``b``."""
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

def update_front(front, r):
    """Update a Pareto front with a new objective vector."""
    new_front = []
    dominated = False
    for v in front:
        if dominates(r, v):
            continue
        if dominates(v, r):
            dominated = True
            new_front.append(v)
        else:
            new_front.append(v)
    if not dominated:
        new_front.append(r)
    return new_front

# Reward function
def reward_fn(T, constant):
    """Compute a scalar reward from a temperature-like value."""
    return constant * T

# MCTS classes
PLAYER_COUNTS = {
    "AM": 4,
    "IM": 2,
    "Pyr": 2,
    "Py": 1,
}


class Tee(object):
    """Write standard output to multiple file-like objects."""

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

class MultiPlayerNode():
    """Tree node used by multi-player Monte Carlo Tree Search."""

    def __init__(self, players=2):
        self.player_lists = [[0] for _ in range(players)]
        self.player_smiles = ["" for _ in range(players)]
        self.player_final_smiles = ["" for _ in range(players)]
        self.depth = 0
        self.current_player = 0
        self.father = None
        self.children = []
        self.is_none = True
        self.smile = ""
        self.visits = 0
        self.pareto_front = []
        self.root = False
        self.activate = False
        self.index = []

    def get_root(self):
        """Return the root node of the current search tree."""
        node = self
        while not node.root:
            node = node.father
        return node

    def children_index(self, index):
        """Return a descendant node specified by a child-index path."""
        root = self.get_root()
        node = root
        for i in index:
            node = node.children[i]
        return node

    def ucb(self, C=0.1):
        """Compute the UCB score used for child selection."""
        if self.visits == 0:
            return math.inf
        score = len(self.pareto_front) / self.visits
        return score + C * math.sqrt(math.log(self.father.visits) / self.visits)
        


def normalize_reward(h, sij, d,
                     h_range=(10.0, 30.0),
                     sij_range=(0.050, 0.090),
                     d_range=(0.50, 0.75)):
    """Scale objective values to comparable ranges for Pareto-based search."""
    h_norm = (h - h_range[0]) / (h_range[1] - h_range[0])
    sij_norm = (sij - sij_range[0]) / (sij_range[1] - sij_range[0])
    d_norm = (d - d_range[0]) / (d_range[1] - d_range[0])
    return (h_norm, sij_norm, d_norm)
    
    

class MPMCTS():
    """Multi-player Monte Carlo Tree Search for cation structure generation."""

    def __init__(self, cation_type="IM", max_depth=100, min_invalid_depth=15):
        if cation_type not in PLAYER_COUNTS:
            raise ValueError(f"Unsupported cation type: {cation_type}")
        players = PLAYER_COUNTS[cation_type]
        self.num_players = players
        self.cation_type = cation_type
        self.smiles = []
        self.max_depth = max_depth
        self.min_invalid_depth = min_invalid_depth

    def build_cation(self, node: MultiPlayerNode):
        """Build a cation SMILES string from the selected scaffold type."""
        fragments = node.player_final_smiles
        if self.cation_type == "AM":
            return fragments[0] + "[N+](" + fragments[1] + ")(" + fragments[2] + ")" + fragments[3]
        if self.cation_type == "IM":
            return fragments[0] + "(n1c[n+](" + fragments[1] + ")cc1)"
        if self.cation_type == "Pyr":
            return fragments[0] + "[N+]1(" + fragments[1] + ")CCCC1"
        if self.cation_type == "Py":
            return "C1=CC=[N+](" + fragments[0] + ")C=C1"
        raise ValueError(f"Unsupported cation type: {self.cation_type}")

    def node_is_terminal(self, node: MultiPlayerNode):
        """Return True when a terminal token appears in any player string."""
        for s in node.player_final_smiles:
            if "$" in s:
                return True
        return False

    def selection(self, root):
        """Select a leaf node by recursively following the highest UCB score."""
        node = root
        while node.children:
            ucb_list = [child.ucb(2) for child in node.children]
            node = node.children[ucb_list.index(max(ucb_list))]
        return node

    def expansion(self, node: MultiPlayerNode):
        """Expand a node by adding valid next molecular fragments."""
        if node.visits == 0 and not node.activate:
            node.activate = True
            return node

        if self.node_is_terminal(node) or node.depth >= self.max_depth:
            return node

        next_player = (node.current_player + 1) % self.num_players
        next_player_next_lists = smiles.next_lists(node.player_lists[next_player])
        if not node.children:
            for i, next_list in enumerate(next_player_next_lists):
                new_node = copy.deepcopy(node)
                new_node.current_player = next_player
                new_node.father = node
                new_node.depth = node.depth + 1
                new_node.player_lists[next_player] = next_list
                new_node.player_smiles[next_player] = smiles.list2string(new_node.player_lists[next_player])
                new_node.player_final_smiles[next_player] = smiles.list2string(smiles.final_list(next_list))
                new_node.index = node.index + [i]
                node.children.append(new_node)

        unvisited_children = [child for child in node.children if child.visits == 0]
        chosen = random.choice(unvisited_children) if unvisited_children else max(node.children, key=lambda c: c.ucb(2))
        chosen.activate = True

        cation = self.build_cation(chosen)
        mol = Chem.MolFromSmiles(cation)
        chosen.smile = cation if mol is None else Chem.MolToSmiles(mol)
        chosen.is_none = (mol is None)
        return chosen

    def simulation(self, node, ty="n2", sac=False):
        """Evaluate a generated cation and assign its normalized reward vector."""
        cation = node.smile
        first_built = cation not in self.smiles
        if first_built:
            self.smiles.append(cation)
            mol = Chem.MolFromSmiles(cation)
            if mol is None:
                node.reward_vector = (-100.0, -100.0, -100.0)
                return

            print(cation, ty)
            value = asdi.ASDI(cation, ty)
            print(value)
            if value is not None:
                h, sij, d = value[1], value[2], value[3]
                node.reward_vector = normalize_reward(h, sij, d)
            else:
                node.reward_vector = (0.0, 0.0, 0.0)
        else:
            node.reward_vector = (0.0, 0.0, 0.0)

    def backpropagation(self, node: MultiPlayerNode):
        """Propagate the reward vector through ancestor Pareto fronts."""
        r = node.reward_vector
        node.pareto_front = update_front(node.pareto_front, r)
        node.visits += 1
        node.activate = False
        parent = node.father
        while not parent.root:
            parent.pareto_front = update_front(parent.pareto_front, r)
            parent.visits += 1
            parent = parent.father

    def restart_tree(self):
        """Create and return a fresh root node."""
        new_root = MultiPlayerNode(self.num_players)
        new_root.father = new_root
        new_root.current_player = -1
        new_root.root = True
        return new_root

    def run(self, n):
        """Run the MCTS loop for ``n`` iterations."""
        root = MultiPlayerNode(self.num_players)
        root.father = root
        root.current_player = -1
        root.root = True

        for i in range(n):
            node = self.selection(root)
            node = self.expansion(node)
            self.simulation(node)
            self.backpropagation(node)
            if node.depth >= self.max_depth or (all(v < -50 for v in node.reward_vector) and node.depth >= self.min_invalid_depth):
                root = self.restart_tree()
        print("MCTS run complete.")

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MPMCTS cation generation.")
    parser.add_argument(
        "cation_type",
        nargs="?",
        default="IM",
        choices=PLAYER_COUNTS.keys(),
        help="Cation scaffold type to generate.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=50,
        help="Maximum search depth.",
    )
    parser.add_argument(
        "--min-depth",
        type=int,
        default=30,
        help="Minimum depth before restarting after invalid rewards.",
    )
    parser.add_argument(
        "-n",
        "--iterations",
        type=int,
        default=1000,
        help="Number of MCTS iterations.",
    )
    args = parser.parse_args()

    mcts = MPMCTS(
        cation_type=args.cation_type,
        max_depth=args.max_depth,
        min_invalid_depth=args.min_invalid_depth,
    )
    with open("mcts_output.txt", "w", encoding="utf-8") as outfile:
        original_stdout = sys.stdout
        sys.stdout = Tee(original_stdout, outfile)
        mcts.run(n=args.iterations)
        sys.stdout = original_stdout
