"""Register metadata only. Never create an environment or draw a scenario."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import sys
import tarfile
import zipfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from securelink.config import load_config, config_digest
from securelink.seeding import SeedDescriptor, _label_u32


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def seed_hits(handle) -> tuple[int, list[int]]:
    csv.field_size_limit(10_000_000)
    reader = csv.DictReader(handle)
    if not reader.fieldnames or "scenario_seed" not in reader.fieldnames:
        return 0, []
    seen = set()
    count = 0
    for row in reader:
        value = row.get("scenario_seed", "")
        if value:
            seen.add(int(Decimal(value)))
            count += 1
    return count, sorted(seen)


def main() -> None:
    if (OUT / "lock_manifest.json").exists():
        raise RuntimeError("Already locked; create a new version instead of overwriting")
    proposed = set(range(31001, 31031)) | set(range(32001, 32101)) | set(range(50001, 50101))
    audit = []
    for path in sorted((ROOT / "results").rglob("*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            count, seen = seed_hits(handle)
        if count:
            audit.append({"path": str(path.relative_to(ROOT)), "sha256": sha(path),
                          "record_rows": count, "scenario_seeds": seen,
                          "proposed_block_hits": sorted(proposed.intersection(seen))})
    for path in sorted((ROOT / "archive").rglob("*.tar.gz")):
        with tarfile.open(path, "r:gz") as archive:
            for member in archive.getmembers():
                if member.isfile() and member.name.endswith(".csv") and "episode" in member.name:
                    stream = archive.extractfile(member)
                    assert stream is not None
                    with io.TextIOWrapper(stream, encoding="utf-8-sig", newline="") as handle:
                        count, seen = seed_hits(handle)
                    if count:
                        audit.append({"path": str(path.relative_to(ROOT)) + "::" + member.name,
                                      "archive_sha256": sha(path), "record_rows": count,
                                      "scenario_seeds": seen,
                                      "proposed_block_hits": sorted(proposed.intersection(seen))})
    if any(row["proposed_block_hits"] for row in audit):
        raise RuntimeError("A proposed block appears in existing local records")
    dump(OUT / "scenario_usage_audit.json", {
        "scope": "All local results CSVs with scenario_seed and archive tar.gz episode CSVs; no remote execution asserted",
        "files_or_members_with_records": len(audit), "proposed_block_record_hits": 0,
        "remote_inventory_gate_before_execution": True, "records": audit,
    })
    records = []
    blocks = [("selection", "p1_rule_select_v1", 31001, 31030),
              ("confirmation", "p1_rule_confirm_v1", 32001, 32100),
              ("final", "p1_final_v1", 50001, 50100)]
    for block, purpose, start, stop in blocks:
        for scenario in range(start, stop + 1):
            descriptor = SeedDescriptor(purpose, 0, 0, 0, scenario)
            random_purpose = purpose + "_bangbang"
            records.append({"block": block, "status": "reserved_not_used", "purpose": purpose,
                "run_seed": 0, "worker_id": 0, "episode_id": 0, "scenario_id": scenario,
                "key": descriptor.key, "fingerprint": descriptor.fingerprint(),
                "root_entropy": [2, _label_u32(purpose), 0, 0, 0, 1, scenario],
                "rng": "numpy.random.Generator(PCG64)",
                "child_order": ["channel", "observation", "attack_location_error"],
                "random_switch_key": SeedDescriptor(random_purpose, 0, 0, 0, scenario).key,
                "random_switch_entropy": [2, _label_u32(random_purpose), 0, 0, 0, 1, scenario],
                "random_switch_child": 0})
    dump(OUT / "scenario_registry.json", records)
    candidates = []
    for base in (0.001, 0.002, 0.004):
        for middle in (0.01, 0.02, 0.1, 0.3):
            candidates.append({"id": f"T{len(candidates)+1:02d}", "family": "time_piecewise",
                               "outer_w": base, "middle_w": middle, "cut_slots": [33, 67]})
    number = 0
    for threshold in (0.0, 100.0, 200.0):
        for high in (0.02, 0.1, 0.3, 1.0):
            number += 1
            candidates.append({"id": f"G{number:02d}", "family": "geometry_threshold",
                               "threshold_m": threshold, "low_w": 0.002, "high_w": high})
    number = 0
    for start in (80, 90, 95):
        for burst in (0.02, 0.1, 0.3, 1.0):
            number += 1
            candidates.append({"id": f"B{number:02d}", "family": "late_burst",
                               "burst_start_slot": start, "low_w": 0.002, "burst_w": burst})
    dump(OUT / "candidate_catalog.json", candidates)
    config = load_config(ROOT / "configs" / "motion_residual_v1.yaml")
    if config_digest(config) != "e9161b54ca242e4f":
        raise RuntimeError("Nominal configuration changed")
    shutil.copyfile(ROOT / "results" / "attack_episode_rules_v1" / "resolved_config.yaml", OUT / "resolved_config.yaml")
    reference = json.loads((ROOT / "results" / "attack_episode_rules_v1" / "manifest.json").read_text(encoding="utf-8"))
    models = {}
    for seed in (11001, 11002, 11003):
        path = ROOT / "results" / "motion_residual_v1_32768" / "models" / f"B10_seed{seed}.zip"
        digest = sha(path)
        if digest != reference["model_sha256"][str(seed)]:
            raise RuntimeError("Frozen model changed")
        models[str(seed)] = {"path": path.relative_to(ROOT).as_posix(), "sha256": digest,
                            "training_transitions": 32768, "checkpoint": "budget_end_no_selection"}
    sources = sorted((ROOT / "src" / "securelink").rglob("*.py"))
    sources += [ROOT / "scripts" / "run_attack_pressure_scan.py", ROOT / "scripts" / "run_attack_privileged_fine.py",
                OUT / "statistics_v1.py", OUT / "build_protocol_lock.py"]
    source_hashes = {path.relative_to(ROOT).as_posix(): sha(path) for path in sources}
    freeze_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload = {"protocol_id": "p1-adaptive-rules-preregister-v1", "locked_utc": freeze_utc,
        "status": "protocol_locked_evaluation_not_started", "config_digest": config_digest(config),
        "resolved_config_sha256": sha(OUT / "resolved_config.yaml"), "models": models,
        "base_source_hashes": source_hashes,
        "candidate_catalog_sha256": sha(OUT / "candidate_catalog.json"),
        "scenario_registry_sha256": sha(OUT / "scenario_registry.json"),
        "usage_audit_sha256": sha(OUT / "scenario_usage_audit.json"),
        "white_list": "仅时隙序号、剩余时隙、自身预算/峰值、带 25 m σ 的 Alice 位置估计。",
        "position_contract": {"delay_slots": 0, "sigma_m": 25, "coordinates": "current_pre_move_communication_position",
                              "each_real_slot_calls_allowed_observation_once": True,
                              "random_switch_rng_separate_from_location_rng": True},
        "baselines": ["F01_low_0p002", "F02_silent", "F03_uniform_0p3", "F04_frontload", "F05_random"],
        "selection": {"models_equal_weight": True, "scenarios_equal_weight": True,
                      "family_candidates": 12, "family_count": 3, "total_new_candidates": 36,
                      "tie_tolerance_bpshz": 1e-8, "tie_break": "lexicographic_id",
                      "overall_pool": "three family winners plus five frozen baselines",
                      "freeze_output": "selection_lock.json required before confirmation"},
        "posthoc_constant_grid_w": [0, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.0075,
                                   0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3],
        "primary": {"delta_bpshz": 0.005, "q_threshold": 0.8, "bootstrap": "paired_scenarios_shared_across_fixed_three_models",
                    "replicates": 10000, "confirm_seed": 94201, "final_seed": 94202,
                    "ratio_nonpositive_draw_policy": "insufficient_evidence_never_filter_or_clip"},
        "planned_task3_upper_slots": {"selection": 369000, "confirmation_B10": 840000,
                                      "confirmation_B00_auxiliary": 280000, "total": 1489000},
        "new_training_transitions": 0, "evaluation_transitions_this_task": 0,
        "final_test_status": "reserved_not_used",
        "execution_code_gate": "Task3 driver/adapters must pass frozen contract checks and be hash-sealed before selection; no physical/defender source change allowed",
        "final_scheme_gate": "Future W1 hashes and executable launch manifest must be sealed before any final scenario is drawn; this registration does not authorize final evaluation"}
    dump(OUT / "protocol_payload.json", payload)
    payload_sha = sha(OUT / "protocol_payload.json")
    report = ROOT.parent / "feedback" / "13_P1自适应规则预注册与测试协议.md"
    content = report.read_text(encoding="utf-8")
    for old, new in (("@LOCK_UTC@", freeze_utc), ("@PAYLOAD_SHA256@", payload_sha),
                     ("@AUDIT_COUNT@", str(len(audit)))):
        content = content.replace(old, new)
    report.write_text(content, encoding="utf-8", newline="\n")
    with zipfile.ZipFile(OUT / "frozen_sources.zip", "w", zipfile.ZIP_DEFLATED) as handle:
        for path in sources:
            handle.write(path, path.relative_to(ROOT).as_posix())
    files = sorted(path for path in OUT.iterdir() if path.is_file())
    dump(OUT / "lock_manifest.json", {"protocol_id": payload["protocol_id"], "locked_utc": freeze_utc,
         "protocol_payload_sha256": payload_sha, "report_path": report.relative_to(ROOT.parent).as_posix(),
         "report_sha256": sha(report), "files": {path.name: sha(path) for path in files},
         "model_hashes_verified": True, "scenario_record_hits": 0,
         "environments_created": 0, "new_training_transitions": 0, "new_evaluation_transitions": 0})
    print(json.dumps({"locked_utc": freeze_utc, "payload_sha256": payload_sha,
                      "local_record_files_or_members": len(audit), "record_hits": 0,
                      "scenario_keys_registered": len(records), "new_candidates": len(candidates),
                      "new_evaluation_transitions": 0}, indent=2))


if __name__ == "__main__":
    main()
