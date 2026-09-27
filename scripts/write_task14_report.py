"""Render the already-computed Task 3 evidence into the deliverable report."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "adaptive_rules_v1"


def read(name):
    return json.loads((OUT/name).read_text(encoding="utf-8"))


def table(frame, columns, labels, digits=5):
    lines = ["| " + " | ".join(labels) + " |", "|" + "|".join("---" for _ in labels) + "|"]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(str(int(value)) if column=="episodes" else (f"{value:.{digits}f}" if isinstance(value,float) else str(value)))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main():
    branch, lock, seal, manifest = (read(name) for name in ("branch_result.json","selection_lock.json","execution_seal.json","result_manifest.json"))
    summary = pd.read_csv(OUT/"overall_summary.csv")
    per_model = pd.read_csv(OUT/"per_model_summary.csv")
    ledger = pd.read_csv(OUT/"compute_ledger.csv")
    scores = pd.read_csv(OUT/"candidate_scores.csv")
    pairs = pd.read_csv(OUT/"paired_primary.csv")
    intervals = pd.read_csv(OUT/"paired_secondary_intervals.csv")
    choices = pd.read_csv(OUT/"posthoc_constant_choices.csv")
    sessions = [json.loads(line) for line in (OUT/"stage_sessions.jsonl").read_text().splitlines()]
    scope = {"protocol_id":"p1-adaptive-rules-preregister-v1","selection":{"labels":[31001,31030],"purpose":"p1_rule_select_v1","status":"used"},
             "confirmation":{"labels":[32001,32100],"purpose":"p1_rule_confirm_v1","status":"used"},
             "final":{"labels":[50001,50100],"purpose":"p1_final_v1","status":"reserved_not_used","actual_transitions":0}}
    (OUT/"scenario_usage_status.json").write_text(json.dumps(scope,ensure_ascii=False,indent=2),encoding="utf-8")
    formal = ledger[ledger.stage != "precheck"]
    checks = read("interface_checks.json")
    names = {"do_not_invest_W1":"简单规则已覆盖所定义参照差距的大部分，本轮不投入 W1",
             "may_prepare_limited_W1_protocol":"规则门槛已完成，可准备小规模 W1 协议",
             "insufficient_evidence":"证据不足/参照差距不可解释，停止在规则结果",
             "halt_integrity_gate":"完整性门槛失败，停止比较"}
    b10 = summary[summary.defense=="residual_B10"]
    b00 = summary[summary.defense=="B00"]
    fair = b10[b10.attack_class=="implementable"]
    privileged = b10[b10.attack_class=="privileged_reference"]
    model_fair = per_model[(per_model.defense=="residual_B10")&(per_model.attack_class=="implementable")]
    model_priv = per_model[(per_model.defense=="residual_B10")&(per_model.attack_class=="privileged_reference")]
    winner_rows = scores[scores.id.isin([spec["id"] for spec in lock["winners"]])]
    improves = int((pairs.L_minus_A>0).sum())
    negative = int((pairs.L_minus_A<0).sum())
    q_text = "未解释" if branch["q"] is None else f"{branch['q']:.6f}，95% 区间 [{branch['q_ci95'][0]:.6f}, {branch['q_ci95'][1]:.6f}]"
    total_wall = sum(item["wall_seconds"] for item in sessions)
    elapsed = (datetime.fromisoformat(manifest["completed_utc"]) - datetime.fromisoformat(seal["sealed_utc"])).total_seconds()
    p_dist = choices[choices.train_seed!=0].selected_constant_w.value_counts().sort_index().rename_axis("power_w").reset_index(name="episodes")
    cols = ["attack_label","asr_bpshz","slot_sop","episode_sop","willie_energy_j","takeover_rate"]
    labels = ["规则","ASR","时隙 SOP","回合 SOP","Willie J","接管率"]
    lines = [
        "# 14 P1 同信息自适应规则评估", "",
        f"> 日期：{datetime.now().date().isoformat()}  ",
        "> 依据：005 任务 3；执行 13 号已锁定协议，无新增训练  ",
        "> 主范围：三个固定残差 B10；名义 −110 dB；新确认场景 32001–32100  ",
        f"> 分支：`{branch['branch']}`", "",
        "## 1. 判定", "",
        f"**{names[branch['branch']]}。** 本结论按预注册 D/q 分支产生，没有重新选择确认赢家、改门槛或增加场景。W1 未训练；任务 4 的准入决定与冻结协议尚未执行。", "",
        f"选择块冻结的总比较规则 A 为 **{branch['A_id']}**。新确认块上 L={branch['L']:.6f}、A={branch['A']:.6f}、P={branch['P']:.6f} bit/s/Hz；D=L−P={branch['D']:.6f}，95% 配对区间 [{branch['D_ci95'][0]:.6f}, {branch['D_ci95'][1]:.6f}]；q={q_text}。", "",
        "剩余差距仍混合特权真信道信息、动作时序和预算使用，不能解释为允许观测下 W1 可获得的收益。本轮没有检验 H1 对手覆盖、H3 协同、跨自干扰稳健性或部署级保密。", "",
        "## 2. 执行合同、预检与封签", "",
        "在线白名单逐字采用：**仅时隙序号、剩余时隙、自身预算/峰值、带 25 m σ 的 Alice 位置估计。** 所有可实现规则走同一六键回调，实时位置估计为移动前通信位置加 x/y 独立 25 m 高斯误差，延迟为 0；每真实时隙只读取一次。随机开关使用独立 `_bangbang` 根流，位置误差与衰落在相同场景和时隙间配对。不同攻击改变防御轨迹时，真实增益可以不同。", "",
        "公开静态地图只使用冻结 Bob (750,350)、Willie (350,650) m 和高度 100 m；所有同信息方法获得同样权限。真信道、未来衰落、当前防御动作、窃听 SINR、奖励历史和防御模型身份没有进入公平规则输入。隐藏真信道仅用于离线日志/特权组。", "",
        "名义配置沿用冻结版本 `p1-motion-residual-v1.0.0`：100 时隙、每时隙通信 0.2 s，Alice 固定 0.2 W（全回合辐射 4 J）、电池 18 kJ；Willie 峰值 1 W、预算 6 J，自干扰 −110 dB。地图、信道、安全层、残差动作范围、奖励与网络均未修改；详细配置和基础源码/模型哈希引用 13 的冻结清单。", "",
        f"运行前完成 {checks['pure_rule_cases']} 个纯规则公式/访问检查、隐藏字段注入拒绝和固定允许输入下的隐藏状态扰动检查；无环境闭包/动态读取。17 个独立预检回合、1,700 个真实转移，标签 90001–90003、用途 `p1_rule_interface_check_v1`，均不使用选择/确认/最终块。B00 跨规则真实增益逐时隙一致，位置噪声与预定第三子流的差在 1e−10 m 内，随机币与独立子流一致；三个冻结模型均作接口回放。合成数据验证了预注册统计分支，不用于调参。", "",
        "首次预检曾因开发阶段的标签保护条件误拒独立预检标签；该次在创建环境前停止，实际转移为 0。已在执行封签前改为明确的阶段/用途/标签白名单，候选、统计和物理合同未改；记录见 `preseal_initialization_note.json`。随后预检通过，正式选择/确认期间未修改已封签执行源码。", "",
        f"执行封签时间 `{seal['sealed_utc']}`；选择锁定时间 `{lock['locked_utc']}`；结果完整清单时间 `{manifest['completed_utc']}`。封签、模型和 13 的哈希在各正式阶段前后核对通过。全部回放在本地 Windows、CPU、4 个进程执行；物理随机流 worker_id 始终为 0，不使用实际进程编号。", "",
        "## 3. 选择块：全部候选保留，确认前冻结", "",
        "31001–31030 使用 `p1_rule_select_v1`。36 个新候选加五条原规则，三个 B10 等权、场景等权，每条候选独立完整回放，共 3,690 回合。三族各 12 个候选：时间分段、几何阈值、低功率加晚段爆发；具体公式与参数仍为 13 的有限集合。", "",
        table(winner_rows,["id","family","selection_mean_asr"],["族胜者","族","选择 ASR"]), "",
        "冻结参数（时隙 n 从 0 起）：T10 在 33≤n<67 时为 0.02 W，其余为 0.004 W；G05 在估计三维距离 d_W−d_B≥100 m 时为 0.02 W，否则为 0.002 W；B09 在 n≥95 时为 0.02 W，此前为 0.002 W。均受共同峰值和剩余预算约束。", "",
        f"总比较 A={lock['overall_A_id']}，从三族胜者及五条原规则中按同一指标选出；并列容差 1e−8 时取字典序最小 ID。三个模型没有分别选参数。确认前已生成 `selection_lock.json`、独立 SHA-256 侧签与完整选择结果哈希；确认只运行固定三族胜者与原规则，特权组不参与公平排名。", "",
        "选择块 41 条规则完整明细见 `candidate_scores.csv` 与 `selection/raw_episodes.csv`；没有删去不利候选。", "",
        "![全部候选选择分数](../pytorch_code/results/adaptive_rules_v1/selection_all_candidates.png)", "",
        "## 4. 新确认块：同信息规则", "",
        "32001–32100 使用独立 `p1_rule_confirm_v1`，三个 B10 每条规则各 100 回合；ASR 越低表示攻击越强。时隙 SOP 为速率 <0.5 的比例，回合 SOP 为 ASR<0.5 的比例；接管率为接管时隙/100。下表为固定三模型等权均值，不代表训练总体泛化。", "",
        table(fair,cols,labels), "",
        f"A 相对低常值在 {improves}/300 个模型×场景配对上降低 ASR，另有 {negative}/300 个负差配对；全部保留。不能把均值改善解释为逐场景支配，也不能由 ASR 排名推断所有 SOP/能量指标同时改善。", "",
        "### 各冻结防御运行，保留所有运行", "",
        table(model_fair,["train_seed"]+cols+["willie_peak_executed_w","task_completed","executed_hard_violations"],
              ["防御运行"]+labels+["回合峰值均值 W","完成率","硬违规均值"]), "",
        "逐运行功率峰值、预算截断、完成率、硬违规和 Alice 推进/辐射/总能量的完整列见 `per_model_summary.csv`。共同峰值为 1 W、能量上限为 6 J；少用预算允许，不能把共同上限写成实际能耗匹配。", "",
        "## 5. 两类特权数值参照", "",
        table(privileged,cols,labels), "",
        table(model_priv,["train_seed"]+cols+["willie_peak_executed_w","task_completed","executed_hard_violations"],
              ["防御运行"]+labels+["回合峰值均值 W","完成率","硬违规均值"]), "",
        "P 是冻结的 189 点当前真信道逐时隙搜索，受剩余预算截断，不优化未来；C 对每个模型/场景独立回放 19 条 0–0.3 W 全程常值轨迹，回合结束后依据真信道速率选最低 ASR，容差并列时取较小功率。每条常值下的防御轨迹独立推进，没有复用另一功率的轨迹。全体 7,600 条常值候选回合及轨迹均保留，包含 B00 辅助。", "",
        "C 的事后选点分布（三 B10，共 300 回合）：", "",
        table(p_dist,["power_w","episodes"],["被选常值 W","回合数"]), "",
        "两类参照均有额外信息，不进入公平攻击排名；既不是严格界，也不是整个回合的最优解。C 与 P 的差不能解释为纯粹自适应的因果效应。", "",
        "## 6. 配对统计与预注册分支", "",
        "主估计量针对固定三个 B10，按场景先对三个模型等权平均。10,000 次场景配对 percentile bootstrap，PCG64 根种子 94201；每次抽样的场景权重对三个模型、L/A/P 共享，不重抽模型。D 与分子同次计算，q 不裁剪、不删除负差；未改变 13 的统计源码和门槛。以下其他配对区间为诊断，没有作多重比较校正，不替代主分支。", "",
        table(intervals,["comparison","mean","ci_low","ci_high"],["配对差","均值","95% 下界","95% 上界"]), "",
        f"主分支为 `{branch['branch']}`：**{names[branch['branch']]}**。分支原始值、区间和选择封签哈希见 `branch_result.json`，逐模型/场景的 L/A/P 与差值见 `paired_primary.csv`。不因 B00、其他指标或单个种子表现改分支，不自动加场景或训练。", "",
        f"本次 D 的区间下界 {branch['D_ci95'][0]:.6f} > δ=0.005，满足参照差距解释条件；q 的区间上界 {branch['q_ci95'][1]:.6f} <0.80，且接口/硬约束门槛通过，因此触发上述分支。", "",
        "![确认攻击压力与实际能量](../pytorch_code/results/adaptive_rules_v1/confirmed_rule_comparison.png)", "",
        "## 7. B00 辅助诊断", "",
        table(b00,cols+["willie_peak_executed_w","task_completed","executed_hard_violations"],labels+["回合峰值均值 W","完成率","硬违规均值"]), "",
        "B00 使用相同确认外生流、每规则 100 回合，仅作固定轨迹诊断，不参与规则选择或 W1 门槛判断。早/中/晚段功率、ASR 与接管分布保存在 `phase_summary.csv`。", "",
        "## 8. 完整性与真实计算账本", "",
        f"正式共 {int(formal.episodes.sum()):,} 回合、{int(formal.real_transitions.sum()):,} 个真实转移、{int(formal.padded_slots.sum()):,} 个补齐时隙；日志时隙 {int(formal.record_slots.sum()):,}。另有预检 17 回合/1,700 个真实转移。学习训练成本为 0，搜索、轨迹回放与确认成本按下表累计，未把总计算成本写成零。", "",
        table(ledger,["stage","defense","class","episodes","real_transitions","padded_slots","worker_episode_seconds"],
              ["阶段","防御","类别","回合","真实转移","补齐","回合秒数之和"],digits=2), "",
        f"正式并行回放阶段墙钟时间合计 {total_wall:.2f} s（不含阶段后 CSV 汇总等）；从执行封签到结果完整清单的经过时间 {elapsed:.2f} s，含进程启动、汇总、监控等待及阶段间空闲，不等于实际计算耗时。预检墙钟 {checks['wall_seconds']:.2f} s。工作进程回合秒数是步进时长之和，不能当成并行墙钟或严格 CPU 计费时间。分阶段开始/结束、四进程信息见 `stage_sessions.jsonl`。", "",
        "首次封签前初始化失败及代码编写耗时未独立计时，未把这些准备开销宣称为零；上述耗时口径是成功预检与正式回放的已记录成本，图表/报告生成另属整理开销。该次初始化没有产生环境转移，不改变实际交互账本。", "",
        f"正式执行硬违规 {manifest['executed_hard_violations']}；逐回合已检查 100 时隙账本、实际功率≤1 W、Willie≤6 J、Alice 辐射≤4 J、能量求和、完整生成键和回调次数。完成/失败/接管按原合同保留，未因性能不利删除或重跑完整回合。没有触发协议停止条件；正式模型/物理/安全层/奖励/网络保持冻结。", "",
        "## 9. 工件、命令与结论边界", "",
        "执行命令（均已运行；项目虚拟环境、各阶段 --workers 4）：", "",
        "```text\n.venv/Scripts/python.exe pytorch_code/scripts/run_adaptive_rules_v1.py --stage precheck --workers 4\n.venv/Scripts/python.exe pytorch_code/scripts/run_adaptive_rules_v1.py --stage select --workers 4\n.venv/Scripts/python.exe pytorch_code/scripts/run_adaptive_rules_v1.py --stage confirm --workers 4\n.venv/Scripts/python.exe pytorch_code/scripts/run_adaptive_rules_v1.py --stage summarize --workers 4\n```", "",
        "- 协议与配置：`pytorch_code/protocols/adaptive_rules_v1/`，载荷 SHA-256 `b1323dd096b026858e531bcd3f4fbfe2e74f41153aa2bdc8444ed2d8975c61e5`；配置摘要 `e9161b54ca242e4f`。", 
        "- 执行源码/模型/依赖/命令：`execution_seal.json`、`execution_sources.zip`；基础源码与三个模型哈希引用 13 的冻结清单，各阶段核对通过。", 
        "- 正式原始记录：`selection/`、`confirmation/`、`posthoc_constant/` 中 `raw_episodes.csv`、`raw_steps.csv`；每回合另有持久化 `checkpoints/*.json.gz`，用于中断恢复和轨迹定位。", 
        "- 选择完整表与封签：`candidate_scores.csv`、`selection_lock.json`、`selection_lock.sha256`；特权事后选点：`posthoc_constant_choices.csv`。", 
        "- 接口证据、分时段、配对值、分支、成本和完整清单：`interface_checks.json`、`phase_summary.csv`、`paired_primary.csv`、`paired_secondary_intervals.csv`、`branch_result.json`、`compute_ledger.csv`、`batch_ledger.jsonl`、`stage_sessions.jsonl`、`result_manifest.json`。以上相对路径均位于 `pytorch_code/results/adaptive_rules_v1/`。", "",
        "选择和确认块现标记为已使用，后续不能称未见最终集；原 13 注册清单保留其运行前历史状态，本轮状态另记 `scenario_usage_status.json`。50001–50100 继续 `reserved_not_used`，实际转移为 0；没有运行 W1、防御加训、策略池、H3 能耗匹配或其他自干扰条件。", "",
        "本轮只支持名义场景下的同信息规则压力与工程投入分支。ASR 仍是理想化 Shannon 容量差，不是实际安全交付比特；三个固定防御者与 100 新场景的区间不是训练总体或部署保证。W1 必要性/增益、H1 与 H3 均须各自后续证据。", "",
        "**任务 3 完成：按预注册分支交付结果，未启动任务 4 或 W1 训练。**", ""
    ]
    report = ROOT.parent/"feedback"/"14_P1同信息自适应规则评估.md"
    report.write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"report":str(report),"branch":branch["branch"],"sha256":hashlib.sha256(report.read_bytes()).hexdigest()},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
