"""
health_metrics.py
=================

Métricas de salud para código generado por grafos writer/reviewer de IA.
Diseñado para ejecutarse en CI o dentro del nodo 'reviewer' del grafo.

Detecta degradación silenciosa que los tests unitarios no capturan:
  - Deriva del vocabulario de dominio (misma idea, muchos nombres).
  - Acoplamiento entre clusters semánticos (imports cruzados).
  - Código 'dormido': módulos que nunca pasaron por revisión humana.
  - Literales mágicos repetidos (señal de tipos/enums faltantes).

Uso:
    python health_metrics.py src/ --vocab design.md --baseline baseline.json
    python health_metrics.py src/ --vocab design.md --baseline baseline.json --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# 1. Vocabulario de dominio extraído del design.md
# ---------------------------------------------------------------------------

# Patrón Deliberadamente conservador: captura términos en negrita, code spans
# y headings de un Markdown de diseño. Evita prosa suelta.
_VOCAB_PATTERN = re.compile(
    r"(?:`([^`]+)`|\*\*([^*]+)\*\*|^#{1,6}\s+(.+?)\s*$)",
    re.MULTILINE,
)

# Tokens que casi nunca son nombres de dominio válidos.
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into",
    "when", "then", "else", "true", "false", "none", "null", "return",
    "import", "from", "class", "def", "self", "args", "kwargs",
}


def extract_vocabulary(design_path: Path) -> set[str]:
    """Devuelve el conjunto canónico de términos de dominio del design.md."""
    if not design_path.is_file():
        return set()
    text = design_path.read_text(encoding="utf-8")
    terms: set[str] = set()
    for match in _VOCAB_PATTERN.finditer(text):
        raw = next(group for group in match.groups() if group)
        for token in re.split(r"[\s,/]+", raw.lower()):
            token = token.strip("-_")
            if len(token) >= 3 and token not in _STOPWORDS and token.isalpha():
                terms.add(token)
    return terms


# ---------------------------------------------------------------------------
# 2. Recorrido del código y aliasing del vocabulario
# ---------------------------------------------------------------------------

# Detecta identificadores estilo snake_case y camelCase.
_IDENT_PATTERN = re.compile(r"\b([a-z_][a-z0-9_]*|[a-z][a-zA-Z0-9]+)\b")
# Detecta literales string repetidos (>=3 chars, sin f-strings triviales).
_STRING_PATTERN = re.compile(r"""(['"])([A-Za-z0-9_:/.\-]{3,})\1""")
# Bloques "humanos": comentarios, marcadores TODO, líneas firmadas.
_HUMAN_MARKERS = re.compile(r"(?i)(#\s*(?:todo|fixme|reviewed-by|hack)|co-authored-by)")


def iter_python_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.rglob("*.py"))


@dataclass
class FileReport:
    path: str
    vocab_coverage: float          # 0..1, fracción de términos del design.md usados
    aliasing_ratio: float          # alias por término canónico (1.0 = perfecto)
    magic_string_duplicates: int   # literales repetidos >=2 veces en el archivo
    human_touch_ratio: float       # 0..1, líneas con marcador humano / total


@dataclass
class HealthReport:
    files: list[FileReport] = field(default_factory=list)
    cross_cluster_imports: int = 0
    overall_score: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "overall_score": round(self.overall_score, 3),
            "cross_cluster_imports": self.cross_cluster_imports,
            "warnings": self.warnings,
            "files": [asdict(f) for f in self.files],
        }


def _term_aliases(file_path: Path, vocab: set[str]) -> Counter:
    """Cuenta cuántos alias distintos usa el archivo para cada término del vocabulario."""
    text = file_path.read_text(encoding="utf-8", errors="ignore").lower()
    identifiers = _IDENT_PATTERN.findall(text)
    # Mapeo término -> conjunto de identificadores que lo contienen.
    alias_map: dict[str, set[str]] = defaultdict(set)
    for ident in identifiers:
        for term in vocab:
            if term in ident and ident != term:
                alias_map[term].add(ident)
    # Para el aliasing ratio: si no hay identificadores derivados, ratio = 1.0
    return Counter({t: len(aliases) for t, aliases in alias_map.items()})


def _magic_strings(file_path: Path) -> int:
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    counts = Counter(m.group(2) for m in _STRING_PATTERN.finditer(text))
    # Penalizamos literales que aparecen 2+ veces fuera de docstrings/tests.
    return sum(c - 1 for c in counts.values() if c >= 2)


def _human_touch(file_path: Path) -> float:
    lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        return 0.0
    touched = sum(1 for ln in lines if _HUMAN_MARKERS.search(ln))
    return min(1.0, touched / max(1, len(lines) // 20))  # normalizado


# ---------------------------------------------------------------------------
# 3. Acoplamiento entre clusters (heurística de paquetes top-level)
# ---------------------------------------------------------------------------

def _detect_clusters(root: Path) -> dict[str, set[str]]:
    """Cada subdirectorio inmediato bajo root es un cluster."""
    clusters: dict[str, set[str]] = defaultdict(set)
    for py in iter_python_files(root):
        rel = py.relative_to(root)
        cluster = rel.parts[0] if len(rel.parts) > 1 else "_root"
        clusters[cluster].add(py.stem)
    return dict(clusters)


def _cross_cluster_imports(root: Path, clusters: dict[str, set[str]]) -> int:
    """Cuenta imports que cruzan la frontera entre clusters."""
    imports_re = re.compile(r"^(?:from\s+([\w.]+)|import\s+([\w.]+))", re.MULTILINE)
    violations = 0
    for py in iter_python_files(root):
        rel = py.relative_to(root)
        own_cluster = rel.parts[0] if len(rel.parts) > 1 else "_root"
        text = py.read_text(encoding="utf-8", errors="ignore")
        for m in imports_re.finditer(text):
            mod = (m.group(1) or m.group(2) or "").split(".")[0]
            if not mod or mod not in clusters:
                continue
            if mod != own_cluster:
                violations += 1
    return violations


# ---------------------------------------------------------------------------
# 4. Orquestador
# ---------------------------------------------------------------------------

def analyze(root: Path, vocab: set[str]) -> HealthReport:
    report = HealthReport()
    if not vocab:
        report.warnings.append(
            "Vocabulario vacío: ¿existe design.md y contiene términos en `code` o **bold**?"
        )

    for py in iter_python_files(root):
        text = py.read_text(encoding="utf-8", errors="ignore").lower()
        used_terms = {t for t in vocab if re.search(rf"\b{re.escape(t)}\b", text)}
        coverage = len(used_terms) / max(1, len(vocab))

        aliases = _term_aliases(py, vocab)
        # Ratio promedio: 1.0 cuando cada término tiene <=1 alias.
        aliasing = (sum(aliases.values()) / max(1, len(aliases))) if aliases else 1.0
        # Invertimos: queremos MENOS alias, así que medimos "pureza" = 1/(1+aliasing).
        alias_purity = 1.0 / (1.0 + aliasing)

        report.files.append(FileReport(
            path=str(py.relative_to(root)),
            vocab_coverage=round(coverage, 3),
            aliasing_ratio=round(alias_purity, 3),
            magic_string_duplicates=_magic_strings(py),
            human_touch_ratio=round(_human_touch(py), 3),
        ))

    clusters = _detect_clusters(root)
    report.cross_cluster_imports = _cross_cluster_imports(root, clusters)

    # Score compuesto: penaliza aliasing, magic strings y deuda dormida;
    # recompensa cobertura de vocabulario y toque humano.
    if report.files:
        avg = lambda key: sum(getattr(f, key) for f in report.files) / len(report.files)
        score = (
            0.30 * avg("vocab_coverage")
            + 0.25 * avg("aliasing_ratio")
            + 0.20 * avg("human_touch_ratio")
            + 0.15 * (1.0 if report.cross_cluster_imports == 0
                      else max(0.0, 1.0 - report.cross_cluster_imports / 20))
            + 0.10 * max(0.0, 1.0 - avg("magic_string_duplicates") / 10)
        )
        report.overall_score = score

    # Reglas de alerta que disparan al reviewer del grafo.
    for f in report.files:
        if f.vocab_coverage < 0.20 and f.path != "__init__.py":
            report.warnings.append(f"{f.path}: cobertura de vocabulario muy baja ({f.vocab_coverage})")
        if f.aliasing_ratio < 0.5:
            report.warnings.append(f"{f.path}: alta fragmentación de aliases")
        if f.human_touch_ratio == 0.0:
            report.warnings.append(f"{f.path}: sin marcadores de revisión humana")

    return report


# ---------------------------------------------------------------------------
# 5. Diff contra baseline (detección de regresión silenciosa)
# ---------------------------------------------------------------------------

def diff_against_baseline(report: HealthReport, baseline_path: Path | None) -> list[str]:
    if not baseline_path or not baseline_path.is_file():
        return ["Sin baseline: se creará uno al terminar el análisis."]
    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [f"Baseline corrupto en {baseline_path}; se ignorará."]

    drifts: list[str] = []
    prev_score = baseline.get("overall_score", 0.0)
    if report.overall_score < prev_score - 0.05:
        drifts.append(
            f"Score global cayó: {prev_score:.3f} -> {report.overall_score:.3f}"
        )
    if report.cross_cluster_imports > baseline.get("cross_cluster_imports", 0):
        drifts.append(
            f"Acoplamiento cross-cluster subió: "
            f"{baseline.get('cross_cluster_imports', 0)} -> {report.cross_cluster_imports}"
        )
    return drifts


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------

def _print_human(report: HealthReport, drifts: list[str]) -> None:
    print(f"\n=== Health Score: {report.overall_score:.3f} ===")
    print(f"Cross-cluster imports: {report.cross_cluster_imports}")
    if report.warnings:
        print("\nWarnings:")
        for w in report.warnings:
            print(f"  - {w}")
    if drifts:
        print("\nDrift vs baseline:")
        for d in drifts:
            print(f"  - {d}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Métricas de salud para código IA.")
    parser.add_argument("root", type=Path, help="Directorio con código Python")
    parser.add_argument("--vocab", type=Path, default=Path("design.md"),
                        help="Markdown con el vocabulario de dominio")
    parser.add_argument("--baseline", type=Path, help="JSON con métricas anteriores")
    parser.add_argument("--json", action="store_true", help="Salida en JSON")
    args = parser.parse_args(argv)

    if not args.root.is_dir():
        print(f"Error: {args.root} no es un directorio.", file=sys.stderr)
        return 2

    vocab = extract_vocabulary(args.vocab)
    report = analyze(args.root, vocab)
    drifts = diff_against_baseline(report, args.baseline)

    if args.json:
        payload = report.to_dict()
        payload["drift"] = drifts
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        _print_human(report, drifts)

    # Persistir baseline nuevo si se proporcionó ruta.
    if args.baseline:
        args.baseline.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # Exit code: 1 si hay drift o score bajo (útil para gating en CI).
    return 1 if drifts or report.overall_score < 0.5 else 0


if __name__ == "__main__":
    raise SystemExit(main())