"""Render completed Task 006/1 evidence; never execute an environment or training."""
from pathlib import Path
import csv
import hashlib
import json
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/w1_feasibility_v1"
REPORT=ROOT.parent/"feedback/17_P1_W1执行预检与封签.md"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if REPORT.exists() or (OUT/"delivery_manifest_task17.json").exists():
        raise RuntimeError("Preserve delivered report; publish a separate revision if needed")
    checks=read(OUT/"precheck_checks.json")
    current=read(OUT/"execution_seal_current.json")
    seal=read(OUT/current["filename"])
    revision=read(OUT/"execution_revision_checks.json")
    audit=read(OUT/"preexecution_inventory_audit.json")
    pairs=read(OUT/"precheck/pair_comparisons.json")
    assert checks["passed"] and len(pairs)==15 and checks["ledger"]["sampler_calls"]==3000
    assert not (OUT/"training").exists() and not (OUT/"models").exists()
    rows=list(csv.DictReader((OUT/"precheck/raw_episodes.csv").open(encoding="utf-8")))
    assert len(rows)==30 and sum(int(row["real_slots"]) for row in rows)==3000
    errors={key:max(row["numeric_max_errors"][key] for row in pairs) for key in pairs[0]["numeric_max_errors"]}
    noise=max(row["noise_max_error_m"] for row in pairs)
    rate=max(abs(row["asr_original"]-row["asr_adapter"]) for row in pairs)
    now=datetime.now(timezone.utc).isoformat(timespec="seconds")
    body=f"""# 17 P1 W1 执行预检与封签

> 日期：2026-09-27  
> 依据：006 任务一；执行 15 的冻结观测、动作、预检与累计预算合同  
> 状态：固定预检通过，训练前执行封签已生成；任务二训练/开发尚未运行  
> 当前执行封签：`{current['filename']}`，SHA-256 `{current['sha256']}`

## 1. 交付与范围

**任务一完成。** 已新增 W1 六维观测/单维功率薄适配器、按阶段执行的驱动、持久调用账本及检查脚本，按规定完成全部三个冻结 B10 的 **15 对/30 回合**预检。真实调用 **3,000**、真实转移 **3,000**、补齐 **0**、执行硬违规 **0**。没有 W1 学习更新、隐含优化器预热、参数搜索或新增训练。

006 已作为后续执行指示纳入执行封签；本次用户要求只完成任务一，故本报告不启动任务二、开发登记或独立可行性评估。预检通过说明实现与冻结合同在规定检查范围内一致，不是 W1 有效性或学习必要性的实验结论。

## 2. 冻结工件与本地执行核对

原 15 协议载荷 `c5788b711a6d5792bcd975956d19610a2e2f242638349f5fcfc78defc58ff442`、原协议锁、原报告原件、报告勘误锁与当前 15 文件均通过哈希核对；未修改 `pytorch_code/protocols/w1_admission_v1/` 的合同、统计或名义配置。原物理/防御源码与三个 B10 文件也逐项核对通过。

名义版本 `p1-motion-residual-v1.0.0`，配置摘要 `e9161b54ca242e4f`；100 时隙、每时隙通信 0.2 s，Alice 固定 0.2 W/辐射 4 J/电池 18 kJ，Willie 峰值 1 W/预算上限 6 J，−110 dB 自干扰。地图、信道、动力学、残差动作、安全层、防御网络及原指标均保持冻结。

| 固定防御 | 预算末模型 SHA-256 |
|---|---|
| 11001 | `382f6291c9ff61652e5e03cdce41d2d99cc5fde15fcbc5401729fa297c9f4e26` |
| 11002 | `e5545922a0a8c8d81bb5be125a59ab7362e6e58547dd98a41eb1dc305fbbb4a9` |
| 11003 | `b371255cb309de8e2889c4aec06b8f4de1f41b92677b753f41480abc01885ff2` |

本次在本地 Windows、CPU 执行，n_envs=1、Torch 1 线程。Python 3.12.12、SB3 2.7.0、Torch 2.5.1、Gymnasium 1.2.1、NumPy 2.3.3 与冻结环境一致。没有改用服务器，故不声称完成服务器库存审计；若未来转移到服务器，仍需按 15/006 补其只读库存审计。

执行前重新扫描本地 results 所有含 scenario_seed 的 CSV 及 archive tar.gz 的逐回合 CSV 成员，共 **{audit['entries']}** 个记录文件/成员；拟用训练、预检、开发、可行性和最终块命中 **{audit['hits']}**。路径、文件哈希和完整历史标签见 `preexecution_inventory_audit.json`。该核对发生于首次新预检 reset 前，不仅依据整数范围判断独立。

## 3. 薄适配器与在线信息隔离

在线白名单逐字采用：**仅时隙序号、剩余时隙、自身预算/峰值、带 25 m σ 的 Alice 位置估计。**

策略输入严格为 slot/100、remaining_slots/100、willie_remaining_j/6、willie_peak_w/1、alice_x_est_m/1000、alice_y_est_m/1000，float32、形状 (6,)。固定尺度，没有历史拼接、奖励历史、动态归一化或位置裁剪；地图外估计保留。actor 和 critic 都使用该六维状态，未加入真信道、当前防御动作、模型身份、场景 ID 或日志对象。

`CachedAttackEnv.prepare_attack` 每时隙从原位置误差流生成一次六键观测并缓存；后续原环境回调只复用该缓存。`PendingPower` 只持有已形成的请求功率和调用计数，没有环境、模型、奖励或日志引用。基础环境的物理 `step`、裁剪和安全层未重写。

W1 适配器执行顺序为 **攻击动作映射并提交 → 冻结防御预测 → 原基础步进**。防御的原 18 维观测仅供防御预测；攻击只接收六维状态。终止后返回六维零向量，策略侧 info 为空，不透传基础环境隐藏信息。真实增益和逐回合汇总只留在离线记录通路。

训练奖励接口返回 `−R_s`，标注 `simulator_privileged_reward`；它是仿真器特权奖励，未加入 actor/critic 的状态或历史。无可行运动失败与尾部补齐沿用原 0 奖励合同。本次没有调用 PPO.learn、PPO.train 或优化器 step 来学习；纯概率检查仅初始化未训练 ActorCriticPolicy 并做只读前向，参数不变、优化器状态为空。

## 4. 动作、概率与合成账本检查

动作按 15 固定公式计算：u 裁剪至 [−1,1]，z=(u+1)/2，请求功率 `0.001*expm1(z*log1p(P_peak/0.001))`，执行功率再受峰值和 E_remaining/0.2 限制。没有改成仿射、网格或强制非零映射。

| 检查 | 结果 |
|---|---|
| 纯映射/逆映射/裁剪/预算组合 | {checks['pure']['mapping_cases']} 个检查，通过；覆盖 0、1 W、低功率点、越界动作、剩余预算与无预算 |
| 非有限/维数错误动作 | 拒绝 NaN、Inf、空动作和多维动作 |
| 隐藏字段注入 | {checks['pure']['extra_fields_rejected']} 类额外字段均拒绝，覆盖真增益、当前防御动作、身份、奖励历史、日志与场景 |
| 固定允许输入的隐藏状态扰动 | 六维编码保持不变，映射与回调无环境闭包或动态隐藏读取 |
| 原始采样动作概率 | 原始动作的前向 log_prob 与 evaluate_actions 重算一致 |
| 越界动作概率 | 越界采样的 raw log_prob 与裁剪动作 log_prob 不同，未把二者混用 |
| 原生 rollout 动作 | 原始动作进入 RolloutBuffer；核对 SB3 原生 collect_rollouts 的 add 动作参数仍为 actions |
| 合成计数 | 真实步/零物理步失败/补齐分开；累计上限阻止超额调用 |
| 合成终止/部分回合 | 终止零观测、隐藏 info 不透传、终止后 step 拒绝；部分回合保存前缀且不追加步进 |
| 持久账本重载 | 已提交调用重建计数；未提交预约不允许重放或下一次预约 |

驱动保持原生 SB3 采样、概率比和更新路径：`actions` 为原始高斯采样，`clipped_actions` 仅用于环境请求。训练审计回调分别记录 `ppo_raw_action`、`ppo_clipped_action`、原始动作 old_log_prob、映射请求功率与实际执行功率；执行功率不替换 rollout 中的采样动作。原始动作可超出 [−1,1]，这是高斯策略加环境裁剪的合同，不按违规删除采样。

计数合成检查在独立临时账本中运行，没有 SecureLink reset/物理转移，不混入 3,000 个真实预检步。实际调用账本在步进前预约并落盘、步进后提交；未决调用无法证明未执行时禁止重放。当前驱动不支持对训练活环境的完整快照续接，遇到已有训练或评估账本会拒绝从头重启，按 15 保留 incomplete，而不静默重采样。合成计数恢复通过不等于已支持训练无损恢复。

## 5. 固定 15 对/30 回合配对回放

严格使用 **39001–39005 ×三个 B10**，每个组合独立运行原 T10 回调与 W1 适配器的同请求功率无学习策略，共 15 对/30 回合。测试策略只从六维输入恢复时隙号，n=33…66 请求 0.02 W，其余 0.004 W，再经固定逆映射进入适配器；没有训练策略或调参。

根键为 `v2:p1_w1_interface_check_v1:run=0:worker=0:episode=0:scenario={{39001…39005}}`，不含 arm 或防御身份。相同场景使用配对外生根流。逐对都保留原回调与适配器的独立轨迹记录，不复用原轨迹计算另一分支。

| 全部 15 对的最大误差 | 数值 |
|---|---:|
| 位置/下一位置/执行位移 m | 0 |
| 三条真实增益 | 0 |
| 冻结防御的两个策略动作 | 0 |
| Alice 功率、推进/时隙能量、电池与辐射状态 | 0 |
| Willie 执行功率 W | {errors['willie_power_w']:.12g} |
| 逐时隙保密速率 bit/s/Hz | {errors['secrecy_rate_bpshz']:.12g} |
| 整回合 ASR 差绝对值 bit/s/Hz | {rate:.12g} |
| 位置噪声与注册第三子流误差 m | {noise:.12g} |

每个适配器回合位置估计生成次数=回调次数=100；配对位置估计误差≤1e−10 m，功率/能量等数值比较≤1e−8，几何比较≤1e−7。接管与补齐标志逐时隙一致，双方硬违规为 0。30 回合均正常完成，无任务失败、无部分回合、无失败补齐；所有结果保留，不因 ASR 不利删除。ASR 在这里仅作实现一致性核对，不用于攻击性能排名。

逐对最大差异及原/适配 ASR 见 `precheck/pair_comparisons.json`。固定额度已全部使用：累计 30 回合/3,000 次调用；**39006–39010 虽已注册，仍未使用，本版没有追加回放额度**。再次运行 precheck 会被拒绝，不能每次修复重领额度。

## 6. 源码封签、版本记录和训练门槛

首份 `execution_seal.json` 时间 `{read(OUT/'execution_seal.json')['sealed_utc']}`，SHA-256 `{sha(OUT/'execution_seal.json')}`；对应原预检驱动源码保存在 `execution_sources.zip`，原封签与快照未覆盖。

首份封签后补入**未来训练更新的非有限损失/权重停止检查**，造成驱动文件哈希变化。训练尚未 reset，随即保留原版并新增执行修订；没有修改物理适配器、账本、配对检查、PPO 超参数、观测/动作或冻结协议。该检查调用原生 PPO 更新后核对 loss/参数是否有限，不新增优化或改梯度。

另用 {revision['synthetic_guard_cases']} 个纯合成检查核对正常损失、四类非有限损失和非有限权重；原生 PPO.train 被 mock，没有实际优化器更新或环境调用。记录见 `execution_revision_checks.json`。这是训练数值停止保护的工程补充，不是 W1 的效果证据，也没有重新领取预检额度。

当前版本时间 `{seal['sealed_utc']}`，封签 `{current['filename']}`；`execution_seal_current.json` 指向该文件并绑定其 SHA-256 与原封签。对应当前源码归档为 `execution_sources_revision1.zip`。后续驱动先核对原锁、勘误锁、全部基础源码/模型/依赖和**当前执行封签**，不使用失配的原驱动哈希继续训练。

当前封签包含真实 precheck 命令、后续各阶段命令、006 文件哈希、配置/协议/模型/源码/依赖、CPU/单线程/单环境、场景审计、累计预算计数口径及检查记录哈希。此刻训练 reset 为 0，满足训练前封签时点。新增执行记录引用 006；15 机器配置的历史 no_training_authorization 状态不覆盖或改写。

| 当前新增执行源码 | SHA-256 |
|---|---|
"""
    for name,digest in seal["execution_sources"].items():
        body+=f"| `{name}` | `{digest}` |\n"
    body+=f"""

训练阶段只允许 21001→21002→21003，每次 64 个 512 步 rollout、32,768 调用，末次更新后保存预算末模型；防御轮换和训练标签取自 15。开发/可行性入口先要求全部三模型的完整预算账本和哈希；可行性另要求方案锁。CLI 没有最终测试选项，阶段标签检查拒绝 50001–50100。上述入口已实现，但本任务未调用，尚无 W1 模型、开发登记、可行性性能或统计分支。

## 7. 实际命令、计算账本与工件

实际已运行：

```text
.venv/Scripts/python.exe pytorch_code/scripts/run_w1_feasibility_v1.py --stage precheck
.venv/Scripts/python.exe pytorch_code/scripts/seal_w1_execution_revision.py
.venv/Scripts/python.exe pytorch_code/scripts/write_task17_report.py
```

后续 train/development/feasibility 命令仅登记于当前封签，**未运行**；本任务的训练、开发、独立可行性、最终测试命令均不适用。所有新增可执行文件与已引用冻结文件均可从清单/快照核对。

| 本次类别 | 调用 | 真实转移 | 补齐 | 优化器更新 |
|---|---:|---:|---:|---:|
| 固定物理预检 | 3,000 | 3,000 | 0 | 0 |
| 纯映射/策略概率/合成计数/数值停止检查 | 0 个物理调用 | 0 | 0 | 0 |
| W1 学习训练 | 0 | 0 | 0 | 0 |
| 开发/独立可行性/策略池/最终测试 | 0 | 0 | 0 | 0 |

配对回放已记录墙钟 **{checks['wall_seconds']:.6f} s**，包含该阶段账本落盘和比较；整段编写、纯检查、库存审计、元数据封签与报告整理未独立计时，不将这些准备开销写成 0，也不把配对阶段时长冒充任务总耗时。没有失败回合、预算末部分回合或实际中断恢复；这些分支只完成明确标注的合成检查。

工件根目录 `pytorch_code/results/w1_feasibility_v1/`：

- 原始逐回合/逐时隙：`precheck/raw_episodes.csv`、`precheck/raw_steps.csv`，30 份 `precheck/checkpoints/*.json.gz`。
- 累计调用预约/提交：`precheck/call_ledger.jsonl`；逐对比较：`precheck/pair_comparisons.json`。
- 接口/映射/概率/账本结果：`precheck_checks.json`；新阶段库存审计：`preexecution_inventory_audit.json`。
- 原执行封签/源码：`execution_seal.json`、`execution_sources.zip`，完整保留。
- 当前封签/指针/源码/合成数值保护：`execution_seal_revision1.json`、`execution_seal_current.json`、`execution_sources_revision1.zip`、`execution_revision_checks.json`。
- 本报告及全部现有结果工件的交付哈希：`delivery_manifest_task17.json`；另有 `scenario_usage_status_task17.json` 记录当前使用状态，不修改原协议注册表。

## 8. 状态与结论边界

39001–39005 已用于本次预检；39006–39010 保留但本版额度已用完。训练 33001–34000、开发 34001–34030、独立可行性 35001–35100 仍未使用；最终 50001–50100 始终 reserved_not_used，调用/转移 0。

本轮没有触发物理预检否证条件，当前版本通过协议/源/模型/依赖与执行封签核对。封签后工程补充和原版本均已明确登记，未静默覆盖或重跑。后续遇到接口、依赖、哈希、预算或非有限数值问题即按 15/006 暂停，保留消耗；技术失败不解释为 W1 无效。

**可按 006 进入任务二；本次交付止于任务一。** 没有 W1 可行性结果，不支持 H1、学习攻击必要性、对未见防御策略泛化、跨自干扰或真实部署安全；H3、W2、策略池、防御加训和最终测试均未启动。
"""
    REPORT.write_text(body,encoding="utf-8")
    status=dict(utc=now,precheck_used=list(range(39001,39006)),precheck_calls=3000,
        precheck_reserved_without_remaining_allowance=list(range(39006,39011)),
        training="reserved_not_used",development="reserved_not_used",feasibility="reserved_not_used",
        final="reserved_not_used",training_transitions=0,final_test_transitions=0)
    (OUT/"scenario_usage_status_task17.json").write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding="utf-8")
    manifest=dict(delivered_utc=now,report_sha256=sha(REPORT),report_path=REPORT.relative_to(ROOT.parent).as_posix(),
        active_execution_seal_sha256=sha(OUT/current["filename"]),
        actual=dict(precheck_episodes=30,precheck_calls=3000,real_transitions=3000,padded_slots=0,hard_violations=0,training_transitions=0,final_transitions=0),
        artifacts={path.relative_to(OUT).as_posix():sha(path) for path in sorted(OUT.rglob("*")) if path.is_file()},
        report_builder_sha256=sha(Path(__file__)))
    (OUT/"delivery_manifest_task17.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(dict(report=str(REPORT),report_sha256=manifest["report_sha256"],actual=manifest["actual"]),ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
