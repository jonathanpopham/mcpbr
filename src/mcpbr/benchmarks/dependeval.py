"""DependEval benchmark implementation for repository dependency understanding.

DependEval evaluates LLMs on three hierarchical tasks:
1. Dependency Recognition (DR) - Identify file dependencies
2. Repository Construction (RC) - Generate project structure from description
3. Multi-file Editing (ME) - Modify code across files to implement features

See: https://github.com/ink7-sudo/DependEval
Paper: https://arxiv.org/abs/2503.06689
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ..docker_env import DockerEnvironmentManager, TaskEnvironment
from .base import BenchmarkTask

# DependEval GitHub repository
DEPENDEVAL_REPO = "https://github.com/ink7-sudo/DependEval.git"

# Supported languages (directory names in the repo)
SUPPORTED_LANGUAGES = ["python", "java", "javascript", "typescript", "c", "c++", "c#", "php"]

# Task mapping: our task type -> (DependEval task name, file suffix)
TASK_FILE_PATTERNS = {
    "dr": ("task2", "_final"),    # Dependency Recognition: task2_{lang}_final.json
    "rc": ("task4", "_new"),      # Repository Construction: task4_{lang}_new.json
    "me": ("task1", ""),          # Multi-file Editing: task1_{lang}.json
}


class DependEvalBenchmark:
    """DependEval benchmark for repository dependency understanding.

    Tasks evaluate LLMs on understanding code dependencies and structure:
    - DR: Given code, identify which files depend on which
    - RC: Generate project structure from requirements
    - ME: Modify code across multiple files to add functionality
    """

    name = "dependeval"

    def __init__(
        self,
        dataset: str | None = None,
        task_type: str = "dr",
        languages: list[str] | None = None,
        cache_dir: Path | None = None,
    ):
        """Initialize DependEval benchmark.

        Args:
            dataset: Path to local DependEval data directory (optional, clones if not provided).
            task_type: Task type - 'dr' (Dependency Recognition), 'rc' (Repository Construction),
                       or 'me' (Multi-file Editing). Default: 'dr'.
            languages: List of languages to include (default: all supported).
            cache_dir: Directory to cache cloned repository.
        """
        self.dataset_path = dataset
        self.task_type = task_type.lower()
        if self.task_type not in TASK_FILE_PATTERNS:
            raise ValueError(
                f"Invalid task_type: {task_type}. Must be one of: {list(TASK_FILE_PATTERNS.keys())}"
            )
        self.languages = languages or SUPPORTED_LANGUAGES
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "mcpbr_dependeval"
        self._data_dir: Path | None = None

    def _ensure_data(self) -> Path:
        """Ensure DependEval data is available, cloning if necessary."""
        if self._data_dir is not None:
            return self._data_dir

        if self.dataset_path:
            self._data_dir = Path(self.dataset_path)
            if not self._data_dir.exists():
                raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")
            return self._data_dir

        repo_dir = self.cache_dir / "DependEval"
        data_dir = repo_dir / "data"

        if data_dir.exists():
            self._data_dir = data_dir
            return self._data_dir

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", DEPENDEVAL_REPO, str(repo_dir)],
            check=True,
            capture_output=True,
        )

        if not data_dir.exists():
            raise RuntimeError(f"Data directory not found after cloning: {data_dir}")

        self._data_dir = data_dir
        return self._data_dir

    def _get_task_file(self, data_dir: Path, lang: str) -> Path | None:
        """Get the task data file path for a given language."""
        task_name, suffix = TASK_FILE_PATTERNS[self.task_type]
        # Try exact filename pattern: task{N}_{lang}{suffix}.json
        filename = f"{task_name}_{lang}{suffix}.json"
        path = data_dir / lang / filename
        if path.exists():
            return path
        return None

    def load_tasks(
        self,
        sample_size: int | None = None,
        task_ids: list[str] | None = None,
        level: int | None = None,
    ) -> list[dict[str, Any]]:
        """Load tasks from DependEval dataset.

        Args:
            sample_size: Maximum number of tasks to load (None for all).
            task_ids: Specific task IDs to load (format: "lang/task_type/idx").
            level: Unused for DependEval.

        Returns:
            List of DependEval task dictionaries.
        """
        data_dir = self._ensure_data()
        tasks = []

        for lang in self.languages:
            data_file = self._get_task_file(data_dir, lang)
            if data_file is None:
                continue

            with open(data_file) as f:
                lang_data = json.load(f)

            if not isinstance(lang_data, list):
                continue

            for idx, item in enumerate(lang_data):
                task_id = f"{lang}/{self.task_type}/{idx}"

                if task_ids and task_id not in task_ids:
                    continue

                task = {
                    "instance_id": task_id,
                    "language": lang,
                    "task_type": self.task_type,
                    "idx": idx,
                    "raw_data": item,
                    "problem_statement": self._generate_problem_statement(item, lang),
                }

                # Extract ground truth
                if self.task_type == "dr":
                    task["gt"] = item.get("gt", [])
                elif self.task_type == "rc":
                    task["gt"] = item.get("gt", [])
                elif self.task_type == "me":
                    task["gt"] = item.get("modified_complete_code", "")
                    task["feature_description"] = item.get("feature_description", "")

                tasks.append(task)

        if sample_size and len(tasks) > sample_size:
            tasks = tasks[:sample_size]

        return tasks

    def _generate_problem_statement(self, item: dict[str, Any], language: str) -> str:
        """Generate problem statement based on task type and actual data fields."""
        if self.task_type == "dr":
            return self._generate_dr_statement(item, language)
        elif self.task_type == "rc":
            return self._generate_rc_statement(item, language)
        else:
            return self._generate_me_statement(item, language)

    def _generate_dr_statement(self, item: dict[str, Any], language: str) -> str:
        """Generate Dependency Recognition problem statement."""
        content = item.get("content", "")
        files = item.get("files", [])
        files_str = ", ".join(str(f) for f in files) if files else "the provided files"

        # Truncate very long content
        if len(content) > 10000:
            content = content[:10000] + "\n... [truncated]"

        return (
            f"Analyze the following {language} code and identify the file dependency chain.\n\n"
            f"Files in this repository: {files_str}\n\n"
            f"Code:\n{content}\n\n"
            f"TASK: Determine the dependency order of these files. A file depends on another "
            f"if it imports, includes, or calls functions/classes defined in that file.\n\n"
            f"Return the dependency chain as a Python list, from dependent to dependency.\n"
            f"Example: ['dependent_file.py', 'imported_file.py', 'base_file.py']\n\n"
            f"IMPORTANT: Write ONLY the dependency list to 'answer.txt':\n"
            f"echo \"['file_a.py', 'file_b.py']\" > answer.txt"
        )

    def _generate_rc_statement(self, item: dict[str, Any], language: str) -> str:
        """Generate Repository Construction problem statement."""
        description = item.get("description", "")
        function = item.get("function", "")
        files_info = item.get("files", [])

        files_desc = ""
        if files_info:
            for f in files_info[:15]:
                if isinstance(f, dict):
                    files_desc += f"- {f.get('file', 'unknown')}: {f.get('function', 'no description')}\n"
                else:
                    files_desc += f"- {f}\n"

        return (
            f"Generate the dependency structure for a {language} repository.\n\n"
            f"Project: {item.get('repo', 'unknown')}\n"
            f"Description: {description}\n"
            f"Function: {function}\n\n"
            f"Files in the repository:\n{files_desc}\n"
            f"TASK: Determine the call-dependency structure between files. Which files call "
            f"functions/classes from which other files?\n\n"
            f"Return the structure as nested lists of call chains:\n"
            f"[['caller.py', 'callee.py'], ['another_caller.py', 'another_callee.py']]\n"
            f"Each inner list represents a call relationship from caller to callee.\n\n"
            f"IMPORTANT: Write ONLY the dependency structure to 'answer.txt':\n"
            f"echo \"[['a.py', 'b.py']]\" > answer.txt"
        )

    def _generate_me_statement(self, item: dict[str, Any], language: str) -> str:
        """Generate Multi-file Editing problem statement."""
        content = item.get("content", "")
        feature_desc = item.get("feature_description", "")
        detailed_desc = item.get("detailed_feature_description", "")
        called_segment = item.get("called_code_segment", "")
        invoking_segment = item.get("invoking_code_segment", "")

        # Truncate long content
        if len(content) > 8000:
            content = content[:8000] + "\n... [truncated]"

        prompt = (
            f"Modify the following {language} code to implement a new feature.\n\n"
            f"Feature: {feature_desc}\n"
        )
        if detailed_desc:
            prompt += f"Details: {detailed_desc}\n"
        prompt += f"\nCurrent code:\n{content}\n\n"

        if called_segment:
            prompt += f"Called code segment:\n{called_segment[:2000]}\n\n"
        if invoking_segment:
            prompt += f"Invoking code segment:\n{invoking_segment[:2000]}\n\n"

        prompt += (
            f"TASK: Modify the code across files to implement the feature. "
            f"Ensure correct function calls between files and proper dependencies.\n\n"
            f"Write your complete modified code to 'answer.txt' as a JSON object mapping "
            f"file names to their contents:\n"
            f"echo '{{\"#file 1\": \"modified code...\"}}' > answer.txt"
        )
        return prompt

    def normalize_task(self, task: dict[str, Any]) -> BenchmarkTask:
        """Convert DependEval task to normalized format."""
        return BenchmarkTask(
            task_id=task["instance_id"],
            problem_statement=task["problem_statement"],
            repo=f"dependeval/{task['language']}/{task['task_type']}",
            commit="",
            metadata={
                "language": task["language"],
                "task_type": task["task_type"],
                "idx": task["idx"],
            },
        )

    async def create_environment(
        self,
        task: dict[str, Any],
        docker_manager: DockerEnvironmentManager,
    ) -> TaskEnvironment:
        """Create environment for DependEval task (no Docker needed)."""
        workdir = Path(tempfile.mkdtemp(prefix="dependeval_"))

        raw_data = task.get("raw_data", {})

        # Write the code content if available
        content = raw_data.get("content", "")
        if content:
            code_file = workdir / "code_context.txt"
            code_file.write_text(content)

        # Write task description
        task_file = workdir / "task.txt"
        task_file.write_text(task["problem_statement"])

        # Write files list if available
        files = raw_data.get("files", [])
        if files:
            files_file = workdir / "files.json"
            files_file.write_text(json.dumps(files, indent=2))

        return _DependEvalEnvironment(workdir, task)

    async def evaluate(
        self,
        env: TaskEnvironment,
        task: dict[str, Any],
        solution: str,
    ) -> dict[str, Any]:
        """Evaluate a solution for DependEval task."""
        gt = task.get("gt", [])

        if self.task_type == "dr":
            return self._evaluate_dr(solution, gt)
        elif self.task_type == "rc":
            return self._evaluate_rc(solution, gt)
        else:
            return self._evaluate_me(solution, gt, task)

    def _evaluate_dr(self, solution: str, gt: list) -> dict[str, Any]:
        """Evaluate Dependency Recognition using exact match."""
        predicted = self._extract_list(solution)

        if not gt:
            return {"resolved": False, "error": "No ground truth available"}

        # Normalize both for comparison (strip quotes, normalize paths)
        predicted_norm = [self._normalize_path(p) for p in predicted]
        gt_norm = [self._normalize_path(g) for g in gt]

        exact_match = predicted_norm == gt_norm

        # Partial match (element overlap)
        if predicted_norm and gt_norm:
            pred_set = set(predicted_norm)
            gt_set = set(gt_norm)
            overlap = len(pred_set & gt_set)
            partial_score = overlap / max(len(pred_set), len(gt_set))
        else:
            partial_score = 0.0

        return {
            "resolved": exact_match,
            "exact_match": exact_match,
            "partial_score": partial_score,
            "predicted": predicted_norm,
            "expected": gt_norm,
        }

    def _evaluate_rc(self, solution: str, gt: list) -> dict[str, Any]:
        """Evaluate Repository Construction using graph F1."""
        predicted = self._extract_nested_list(solution)

        if not gt:
            return {"resolved": False, "error": "No ground truth available"}

        # Build graphs
        pred_nodes, pred_edges = self._build_graph(predicted)
        gt_nodes, gt_edges = self._build_graph(gt)

        # Node F1
        if pred_nodes and gt_nodes:
            node_tp = len(pred_nodes & gt_nodes)
            node_precision = node_tp / len(pred_nodes)
            node_recall = node_tp / len(gt_nodes)
            node_f1 = (2 * node_precision * node_recall / (node_precision + node_recall)
                       if (node_precision + node_recall) > 0 else 0)
        else:
            node_f1 = 0.0

        # Edge F1
        if pred_edges and gt_edges:
            edge_tp = len(pred_edges & gt_edges)
            edge_precision = edge_tp / len(pred_edges)
            edge_recall = edge_tp / len(gt_edges)
            edge_f1 = (2 * edge_precision * edge_recall / (edge_precision + edge_recall)
                       if (edge_precision + edge_recall) > 0 else 0)
        else:
            edge_f1 = 0.0

        # Combined F1 (DependEval weighting: 0.15 * node + 0.85 * edge)
        combined_f1 = 0.15 * node_f1 + 0.85 * edge_f1
        resolved = combined_f1 >= 0.5

        return {
            "resolved": resolved,
            "combined_f1": combined_f1,
            "node_f1": node_f1,
            "edge_f1": edge_f1,
        }

    def _evaluate_me(self, solution: str, gt: Any, task: dict[str, Any]) -> dict[str, Any]:
        """Evaluate Multi-file Editing using heuristics.

        Full evaluation would use LLM-based scoring (as in original DependEval).
        We use keyword/structure heuristics as a practical approximation.
        """
        if not gt:
            return {"resolved": False, "error": "No ground truth available"}

        feature_desc = task.get("feature_description", "")

        # Extract keywords from feature description
        stop_words = {'the', 'a', 'an', 'to', 'of', 'and', 'or', 'in', 'for', 'is',
                      'be', 'that', 'this', 'it', 'with', 'as', 'on', 'by', 'from'}
        keywords = set(re.findall(r'\b\w{3,}\b', feature_desc.lower())) - stop_words

        solution_lower = solution.lower()
        matched_keywords = sum(1 for kw in keywords if kw in solution_lower)
        keyword_coverage = matched_keywords / len(keywords) if keywords else 0

        # Check for code structure
        has_functions = bool(re.search(r'def \w+|function \w+|func \w+|fn \w+', solution))
        has_imports = bool(re.search(r'import |from .+ import|require\(|#include|use ', solution))

        # Simple scoring
        score = 0.0
        if len(solution) > 100:
            score += 0.2
        if has_functions:
            score += 0.2
        if has_imports:
            score += 0.1
        score += keyword_coverage * 0.5

        return {
            "resolved": score >= 0.5,
            "score": score,
            "keyword_coverage": keyword_coverage,
            "has_functions": has_functions,
            "has_imports": has_imports,
        }

    def _normalize_path(self, path: str) -> str:
        """Normalize a file path for comparison."""
        # Remove surrounding quotes
        p = str(path).strip().strip("'\"")
        return p

    def _extract_list(self, text: str) -> list:
        """Extract a Python list from text."""
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            try:
                result = eval(match.group())
                if isinstance(result, list):
                    return result
            except Exception:
                pass

        try:
            parsed = json.loads(text.strip())
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

        return []

    def _extract_nested_list(self, text: str) -> list:
        """Extract nested lists from text."""
        match = re.search(r'\[\s*\[.*?\]\s*\]', text, re.DOTALL)
        if match:
            try:
                result = eval(match.group())
                if isinstance(result, list):
                    return result
            except Exception:
                pass
        return self._extract_list(text)

    def _build_graph(self, chains: list) -> tuple[set, set]:
        """Build graph nodes and edges from call chains."""
        nodes: set[str] = set()
        edges: set[tuple[str, str]] = set()

        for chain in chains:
            if not isinstance(chain, list):
                continue
            for i, node in enumerate(chain):
                node_str = self._normalize_path(node)
                nodes.add(node_str)
                if i > 0:
                    prev = self._normalize_path(chain[i - 1])
                    edges.add((prev, node_str))

        return nodes, edges

    def get_prebuilt_image(self, task: dict[str, Any]) -> str | None:
        """DependEval doesn't use Docker images."""
        return None

    def get_prompt_template(self) -> str:
        """Get DependEval prompt template."""
        if self.task_type == "dr":
            return (
                "You are a code analysis assistant specializing in dependency analysis. "
                "Your task is to identify file dependencies in a codebase.\n\n"
                "{problem_statement}\n\n"
                "Use your code analysis tools to understand the dependency relationships. "
                "Write ONLY the dependency list to 'answer.txt'."
            )
        elif self.task_type == "rc":
            return (
                "You are a software architect assistant. Your task is to determine the dependency "
                "structure for a code repository.\n\n"
                "{problem_statement}\n\n"
                "Use your code analysis tools to understand how files relate to each other. "
                "Write ONLY the dependency structure to 'answer.txt'."
            )
        else:
            return (
                "You are a code modification assistant. Your task is to modify code across "
                "multiple files to implement a new feature while maintaining correct dependencies.\n\n"
                "{problem_statement}\n\n"
                "Use your code analysis tools to understand the codebase structure, then "
                "implement the feature. Write your modified code to 'answer.txt'."
            )


class _DependEvalEnvironment(TaskEnvironment):
    """Minimal environment for DependEval (no Docker needed)."""

    def __init__(self, workdir: Path, task: dict):
        self._workdir = workdir
        self._task = task
        self.uses_prebuilt = False
        self.claude_cli_installed = False  # Force local execution

    @property
    def workdir(self) -> str:
        return str(self._workdir)

    @property
    def host_workdir(self) -> str:
        return str(self._workdir)

    async def exec_command(
        self,
        command: str | list,
        timeout: int = 60,
        workdir: str | None = None,
        **kwargs,
    ) -> tuple[int, str, str]:
        """Execute command in the environment."""
        import asyncio

        cwd = workdir or str(self._workdir)
        cmd = command if isinstance(command, str) else " ".join(command)

        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return proc.returncode or 0, stdout.decode(), stderr.decode()
        except asyncio.TimeoutError:
            return -1, "", "Command timed out"
        except Exception as e:
            return -1, "", str(e)

    async def cleanup(self) -> None:
        """Clean up the environment."""
        import shutil

        if self._workdir.exists():
            shutil.rmtree(self._workdir, ignore_errors=True)
