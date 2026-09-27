"""Preserve the byte-identical first report seal; publish a report-only rounding erratum."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

OUT = Path(__file__).resolve().parent
REPORT = OUT.parents[2] / "feedback/15_P1_W1准入决定与冻结协议.md"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    revision = OUT / "report_revision_lock.json"
    if revision.exists():
        raise RuntimeError("Already revised; preserve this version")
    original_lock = json.loads((OUT / "lock_manifest.json").read_text(encoding="utf-8"))
    current = REPORT.read_text(encoding="utf-8")
    old = current.replace("| 0.02 | −0.11865 |", "| 0.02 | −0.11864 |")
    old = old.replace("| 0.3 | 0.65214 |", "| 0.3 | 0.65217 |")
    old_bytes = old.replace("\n", "\r\n").encode("utf-8")
    if sha(old_bytes) != original_lock["report_sha256"]:
        raise RuntimeError("Cannot reconstruct exact first report seal; stop")
    (OUT / "report_locked_original.md").write_bytes(old_bytes)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    current += (
        "\n## 10. 报告勘误与封签追溯\n\n"
        f"报告整理勘误时间 `{stamp}`：第 3 节两处逆映射近似值按冻结公式的五位小数四舍五入校正，"
        "0.02 W 的 u 由 −0.11864 改为 −0.11865，0.3 W 由 0.65217 改为 0.65214。"
        "校正发生在首份报告封签之后，故保留原件并另行封签当前报告，不覆盖原锁文件。"
        "动作公式、尺度常量、精确元数据、训练/评估预算、模型、场景、统计和准入结论均未改变；"
        "新增实验交互仍为 0。\n\n"
        "`lock_manifest.json` 的 report_sha256 对应本协议目录的 `report_locked_original.md`，"
        f"原 SHA-256 `{original_lock['report_sha256']}`；当前交付的 15 号文件哈希见 "
        "`report_revision_lock.json`，它同时绑定原锁文件、原报告及协议载荷。"
        "后续执行前核对须覆盖原协议锁与当前报告勘误锁。\n"
    )
    REPORT.write_text(current,encoding="utf-8")
    value = {
        "revision":"report-only-rounding-r1", "revised_utc":stamp,
        "reason":"Two displayed inverse-map rounding values were corrected after the first report seal; preserve first sealed bytes",
        "protocol_changed":False, "new_experiment_interactions":0,
        "protocol_payload_sha256":original_lock["protocol_payload_sha256"],
        "original_lock_sha256":sha((OUT / "lock_manifest.json").read_bytes()),
        "original_report_sha256":sha((OUT / "report_locked_original.md").read_bytes()),
        "delivered_report_sha256":sha(REPORT.read_bytes()),
        "erratum_source_sha256":sha(Path(__file__).read_bytes()),
        "corrections":[{"power_w":0.02,"old_display":-0.11864,"new_display":-0.11865},
                       {"power_w":0.3,"old_display":0.65217,"new_display":0.65214}]
    }
    revision.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(value,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
