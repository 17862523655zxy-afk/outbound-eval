"""Evaluation report generation.

Produces a self-contained HTML report (and optional Markdown) summarizing
one run of the evaluation pipeline. The HTML inlines CSS so it can be
opened offline and forwarded to non-technical stakeholders.
"""

from __future__ import annotations

import html as html_lib
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from outbound_eval.analytics.bootstrap import BootstrapCI
from outbound_eval.dataset.task import EvaluationTask


DIMENSION_KEYS = [
    ("task_success", "Task Success", "任务完成度"),
    ("flow_adherence", "Flow Adherence", "流程执行"),
    ("state_tracking", "State Tracking", "状态记录"),
    ("compliance", "Compliance", "话术合规"),
    ("recovery", "Recovery", "异常恢复"),
    ("naturalness", "Naturalness", "自然度"),
    ("efficiency", "Efficiency", "成本/效率"),
]

RISK_THRESHOLD = 60.0
MAX_DIALOGUE_TURNS_IN_REPORT = 30
MAX_EVIDENCE_PER_CASE = 5
MAX_FAILURE_CASES = 5


class ReportBuilder:
    """Build an evaluation report from a task + its result records."""

    def __init__(
        self,
        task: EvaluationTask,
        results: list[dict],
        run_id: str = "",
        run_name: str = "",
    ):
        self.task = task
        self.results = results
        self.run_id = run_id or (results[0].get("run_id", "") if results else "unknown")
        self.run_name = run_name or (results[0].get("run_name", "") if results else self.run_id)
        self._ci = BootstrapCI(n_bootstrap=1000)

    # ---------- entry points ----------

    def build_html(self) -> str:
        return self._render_html()

    def build_markdown(self) -> str:
        return self._render_markdown()

    def save(self, output_path: Path, format: str = "html") -> Path:
        """Persist the report. ``format`` ∈ {"html", "md", "pdf"}.

        For ``"pdf"`` the HTML is rendered via :mod:`weasyprint`; raises
        :class:`outbound_eval.reporting.pdf.PdfRenderError` if the system
        libraries are missing.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fmt = (format or "html").lower()
        if fmt == "html":
            text = self.build_html()
        elif fmt in ("md", "markdown"):
            text = self.build_markdown()
        elif fmt == "pdf":
            from outbound_eval.reporting.pdf import html_to_pdf, PdfRenderError
            try:
                return html_to_pdf(self.build_html(), output_path)
            except PdfRenderError as e:
                raise
        else:
            raise ValueError(f"Unsupported format: {format!r} (use html/md/pdf)")

        if fmt == "html" and len(text.encode("utf-8")) > 5 * 1024 * 1024:
            text = text[: 5 * 1024 * 1024] + "\n<!-- truncated -->\n"
        output_path.write_text(text, encoding="utf-8")
        return output_path

    # ---------- shared stats helpers ----------

    def _summary(self) -> dict[str, Any]:
        n = len(self.results)
        if n == 0:
            return {
                "total": 0, "passed": 0, "failed": 0, "success_rate": 0.0,
                "avg_score": 0.0, "risks": [],
            }
        passed = sum(1 for r in self.results if r.get("passed", False))
        scores = [float(r.get("overall_score", 0.0)) for r in self.results]
        risks = []
        for key, en, zh in DIMENSION_KEYS:
            samples = [float(r.get(key, 0.0)) for r in self.results if r.get(key) is not None]
            if samples and (sum(samples) / len(samples)) < RISK_THRESHOLD:
                risks.append(f"{zh}（{en}）均值 {sum(samples) / len(samples):.1f} < {RISK_THRESHOLD}")
        return {
            "total": n,
            "passed": passed,
            "failed": n - passed,
            "success_rate": passed / n * 100,
            "avg_score": sum(scores) / n,
            "risks": risks,
        }

    def _dimension_stats(self) -> list[dict[str, Any]]:
        out = []
        for key, en, zh in DIMENSION_KEYS:
            samples = [float(r.get(key, 0.0)) for r in self.results if r.get(key) is not None]
            if not samples:
                continue
            mean = sum(samples) / len(samples)
            std = statistics.pstdev(samples) if len(samples) > 1 else 0.0
            ci = self._ci.calculate_ci(samples, metric="mean", confidence_level=0.95)
            out.append({
                "key": key, "en": en, "zh": zh,
                "n": len(samples), "mean": mean, "std": std,
                "ci_low": ci.ci_95_lower, "ci_high": ci.ci_95_upper,
            })
        return out

    def _worst_cases(self, k: int = MAX_FAILURE_CASES) -> list[dict[str, Any]]:
        failed = [r for r in self.results if not r.get("passed", False)]
        failed.sort(key=lambda r: float(r.get("overall_score", 100.0)))
        return failed[:k]

    def _truncate_dialogue(self, dialogue: list[dict]) -> list[dict]:
        return dialogue[:MAX_DIALOGUE_TURNS_IN_REPORT]

    def _format_failure_case(self, case: dict) -> str:
        persona = case.get("persona_type", "")
        difficulty = case.get("difficulty", "")
        score = case.get("overall_score", 0.0)
        dialogue = self._truncate_dialogue(case.get("dialogue_history", []))
        details = case.get("criterion_details", []) or []
        failed_details = [d for d in details if not d.get("satisfied", True)]
        violations = case.get("failure_violations", []) or []
        gold = case.get("gold_comparison") or {}

        # === 评分依据追溯（所有 criterion，含三态） ===
        if details:
            all_items = []
            for d in details[:MAX_EVIDENCE_PER_CASE]:
                state = d.get("satisfied")
                score_pct = float(d.get("score", 0)) * 100
                if state:
                    state_chip = "<span class='ok-chip'>✓ 满足</span>"
                elif score_pct >= 40:
                    state_chip = f"<span class='warn-chip'>⚠ 部分 {score_pct:.0f}%</span>"
                else:
                    state_chip = f"<span class='viol-chip'>✗ 未满足 {score_pct:.0f}%</span>"
                ev = d.get("evidence", []) or []
                ev_html = "".join(
                    f"<div class='evidence'>{html_lib.escape(str(e))}</div>"
                    for e in ev[:3]
                )
                method_label = d.get("method", "")
                all_items.append(
                    f"<div class='crit-item'>"
                    f"<div class='crit-head'>"
                    f"{state_chip}"
                    f"<span class='crit-name'>{html_lib.escape(str(d.get('name', '')))}</span>"
                    f"<span class='crit-meta'>{html_lib.escape(str(d.get('priority', '')))} · {html_lib.escape(str(method_label))} · 得分 {score_pct:.0f}</span>"
                    f"</div>"
                    f"<div class='crit-reason'>{html_lib.escape(str(d.get('reasoning', '')))}</div>"
                    f"{ev_html}"
                    f"</div>"
                )
            crit_html = "<div class='crit-list'>" + "".join(all_items) + "</div>"
        else:
            crit_html = "<div class='empty'>无未满足的 success criterion 详情（数据缺少 criterion_details）</div>"

        # === 扣分明细（按扣分从大到小） ===
        penalty_rows = []
        for d in failed_details:
            loss = (1 - float(d.get("score", 0))) * 100
            penalty_rows.append({
                "name": d.get("name", ""),
                "priority": d.get("priority", ""),
                "loss": loss,
                "reasoning": d.get("reasoning", ""),
            })
        penalty_rows.sort(key=lambda r: -r["loss"])
        if penalty_rows:
            penalty_html = (
                "<table class='dims' style='margin:8px 0;'>"
                "<thead><tr><th>扣分</th><th>条件</th><th>优先级</th><th>扣分依据</th></tr></thead><tbody>"
                + "".join(
                    f"<tr><td><span class='viol-chip'>-{p['loss']:.0f}分</span></td>"
                    f"<td><strong>{html_lib.escape(p['name'])}</strong></td>"
                    f"<td>{html_lib.escape(p['priority'])}</td>"
                    f"<td style='font-size:11px;color:#cbd5e1;'>{html_lib.escape(p['reasoning'])}</td></tr>"
                    for p in penalty_rows[:8]
                )
                + "</tbody></table>"
            )
        else:
            penalty_html = "<div class='empty'>无扣分项</div>"

        # Gold comparison strip
        gold_html = ""
        if gold:
            cov = gold.get("coverage", 0.0)
            seq = gold.get("sequence_alignment", 0.0)
            outcome = gold.get("outcome_match")
            outcome_chip = (
                "<span class='viol-chip'>结果不一致</span>"
                if outcome is False
                else ("<span class='ok-chip'>结果一致</span>" if outcome is True
                      else "<span class='empty'>无信号</span>")
            )
            gold_id = html_lib.escape(str(gold.get("matched_gold_id") or "-"))
            quality = html_lib.escape(str(gold.get("gold_quality") or "-"))
            notes = "".join(f"<li>{html_lib.escape(str(n))}</li>" for n in (gold.get("notes") or []))
            gold_html = f"""
                <div class='gold-card'>
                    <div class='case-section-title'>与标杆对话对比 (gold: {gold_id} · quality={quality})</div>
                    <div class='kpi-row' style='margin-bottom:6px;'>
                        <div class='kpi' style='padding:8px;'><div class='kpi-num'>{cov*100:.0f}%</div><div class='kpi-label'>内容覆盖</div></div>
                        <div class='kpi' style='padding:8px;'><div class='kpi-num'>{seq*100:.0f}%</div><div class='kpi-label'>顺序对齐</div></div>
                        <div class='kpi' style='padding:8px;'><div class='kpi-num' style='font-size:18px;'>{outcome_chip}</div><div class='kpi-label'>结果对齐</div></div>
                    </div>
                    <ul style='margin:0 0 0 16px;font-size:12px;color:#cbd5e1;'>{notes}</ul>
                </div>"""

        crit_html = ""
        if failed_details:
            items = []
            for d in failed_details[:MAX_EVIDENCE_PER_CASE]:
                ev = d.get("evidence", []) or []
                ev_html = "".join(
                    f"<div class='evidence'>{html_lib.escape(str(e))}</div>"
                    for e in ev[:3]
                )
                method_label = d.get("method", "")
                items.append(
                    f"<div class='crit-item crit-fail'>"
                    f"<div class='crit-head'>"
                    f"<span class='crit-name'>{html_lib.escape(str(d.get('name', '')))}</span>"
                    f"<span class='crit-meta'>{html_lib.escape(str(d.get('priority', '')))} · {html_lib.escape(str(method_label))}</span>"
                    f"</div>"
                    f"<div class='crit-reason'>{html_lib.escape(str(d.get('reasoning', '')))}</div>"
                    f"{ev_html}"
                    f"</div>"
                )
            crit_html = "<div class='crit-list'>" + "".join(items) + "</div>"
        else:
            crit_html = "<div class='empty'>无未满足的 success criterion 详情（数据缺少 criterion_details）</div>"

        viol_html = ""
        if violations:
            viol_html = "<div class='viol-list'>" + "".join(
                f"<span class='viol-chip'>{html_lib.escape(str(v))}</span>"
                for v in violations
            ) + "</div>"

        turns_html = "".join(
            f"<div class='turn turn-{(t.get('role') or 'unknown').lower()}'>"
            f"<span class='turn-label'>{html_lib.escape(str(t.get('role', '?')))} t{t.get('turn_number', i + 1)}</span>"
            f"<div class='turn-content'>{html_lib.escape(str(t.get('content', '')))}</div>"
            f"</div>"
            for i, t in enumerate(dialogue)
        )

        return (
            f"<div class='case-card'>"
            f"<div class='case-head'>"
            f"<div><span class='case-tag'>失败案例</span>"
            f"<span class='persona-tag'>{html_lib.escape(str(persona))}</span>"
            f"<span class='diff-tag'>{html_lib.escape(str(difficulty))}</span></div>"
            f"<div class='case-score'>综合分 <strong>{float(score):.1f}</strong></div>"
            f"</div>"
            f"<div class='case-section-title'>📉 扣分明细（按扣分从大到小）</div>"
            f"{penalty_html}"
            f"<div class='case-section-title'>📊 评分依据追溯（每条判定的方法、得分、证据）</div>"
            f"{crit_html}"
            f"<div class='case-section-title'>触发的失败红线</div>"
            f"{viol_html or '<div class=\"empty\">无</div>'}"
            f"{gold_html}"
            f"<div class='case-section-title'>对话回放（截取前 {MAX_DIALOGUE_TURNS_IN_REPORT} 轮）</div>"
            f"<div class='dialogue'>{turns_html or '<div class=\"empty\">无对话</div>'}</div>"
            f"</div>"
        )

    def _recommendations(self) -> list[str]:
        recs: list[str] = []
        details_dim = {d["key"]: d for d in self._dimension_stats()}
        p0_total = sum(int(r.get("p0_total", 0) or 0) for r in self.results)
        p0_passed = sum(int(r.get("p0_passed", 0) or 0) for r in self.results)
        if p0_total and p0_passed < p0_total:
            recs.append(
                f"存在 P0 关键条件未满足：{p0_passed}/{p0_total}。请优先保证关键步骤。"
            )
        for key, en, zh in DIMENSION_KEYS:
            s = details_dim.get(key)
            if not s:
                continue
            if s["mean"] < RISK_THRESHOLD:
                recs.append(f"{zh}（{en}）均分 {s['mean']:.1f}，低于阈值 {RISK_THRESHOLD}，需关注。")
        if not recs:
            recs.append("整体表现良好，建议持续监控并扩大评测样本量。")
        return recs

    # ---------- HTML render ----------

    def _render_html(self) -> str:
        s = self._summary()
        dims = self._dimension_stats()
        worst = self._worst_cases()
        recs = self._recommendations()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        css = self._css()

        sec_target = f"""
        <section class='card'>
          <h2>1. 测试目标</h2>
          <table class='kv'>
            <tr><th>任务名称</th><td>{html_lib.escape(self.task.name)}</td></tr>
            <tr><th>任务 ID</th><td>{html_lib.escape(self.task.task_id)}</td></tr>
            <tr><th>难度</th><td>{html_lib.escape(self.task.difficulty.value if hasattr(self.task.difficulty, 'value') else str(self.task.difficulty))}</td></tr>
            <tr><th>Skill</th><td><code>{html_lib.escape(self.task.skill_name)}</code></td></tr>
            <tr><th>Run ID</th><td><code>{html_lib.escape(self.run_id)}</code></td></tr>
            <tr><th>Run Name</th><td>{html_lib.escape(self.run_name)}</td></tr>
            <tr><th>生成时间</th><td>{html_lib.escape(now)}</td></tr>
          </table>
        </section>"""

        persona_dist: dict[str, int] = {}
        diff_dist: dict[str, int] = {}
        for r in self.results:
            p = r.get("persona_type", "unknown")
            d = r.get("difficulty", "unknown")
            persona_dist[p] = persona_dist.get(p, 0) + 1
            diff_dist[d] = diff_dist.get(d, 0) + 1
        persona_rows = "".join(
            f"<tr><td>{html_lib.escape(str(k))}</td><td>{v}</td></tr>"
            for k, v in sorted(persona_dist.items())
        )
        diff_rows = "".join(
            f"<tr><td>{html_lib.escape(str(k))}</td><td>{v}</td></tr>"
            for k, v in sorted(diff_dist.items())
        )
        sec_scale = f"""
        <section class='card'>
          <h2>2. 评测规模</h2>
          <div class='kpi-row'>
            <div class='kpi'><div class='kpi-num'>{s['total']}</div><div class='kpi-label'>总 Case</div></div>
            <div class='kpi kpi-pass'><div class='kpi-num'>{s['passed']}</div><div class='kpi-label'>通过</div></div>
            <div class='kpi kpi-fail'><div class='kpi-num'>{s['failed']}</div><div class='kpi-label'>失败</div></div>
            <div class='kpi'><div class='kpi-num'>{s['success_rate']:.1f}%</div><div class='kpi-label'>通过率</div></div>
          </div>
          <div class='dist-grid'>
            <div><h4>Persona 分布</h4><table class='dist'><thead><tr><th>类型</th><th>数量</th></tr></thead><tbody>{persona_rows or '<tr><td colspan=2>—</td></tr>'}</tbody></table></div>
            <div><h4>Difficulty 分布</h4><table class='dist'><thead><tr><th>难度</th><th>数量</th></tr></thead><tbody>{diff_rows or '<tr><td colspan=2>—</td></tr>'}</tbody></table></div>
          </div>
        </section>"""

        risk_html = ""
        if s["risks"]:
            risk_items = "".join(f"<li>{html_lib.escape(r)}</li>" for r in s["risks"])
            risk_html = f"<div class='risk-box'><strong>⚠ 风险告警</strong><ul>{risk_items}</ul></div>"
        else:
            risk_html = "<div class='ok-box'>✓ 全部维度均值 ≥ 60 分，无明显短板</div>"
        sec_overall = f"""
        <section class='card'>
          <h2>3. 总体结论</h2>
          <div class='kpi-row'>
            <div class='kpi'><div class='kpi-num'>{s['avg_score']:.1f}</div><div class='kpi-label'>综合均分</div></div>
            <div class='kpi'><div class='kpi-num'>{s['success_rate']:.1f}%</div><div class='kpi-label'>通过率</div></div>
          </div>
          {risk_html}
        </section>"""

        dim_rows = "".join(
            f"<tr>"
            f"<td>{html_lib.escape(d['zh'])}<br/><span class='muted'>{html_lib.escape(d['en'])}</span></td>"
            f"<td>{d['n']}</td>"
            f"<td><strong>{d['mean']:.1f}</strong></td>"
            f"<td>{d['std']:.1f}</td>"
            f"<td>[{d['ci_low']:.1f}, {d['ci_high']:.1f}]</td>"
            f"</tr>"
            for d in dims
        )
        sec_dims = f"""
        <section class='card'>
          <h2>4. 维度分数（含 95% Bootstrap 置信区间）</h2>
          <table class='dims'>
            <thead><tr><th>维度</th><th>N</th><th>均分</th><th>标准差</th><th>95% CI</th></tr></thead>
            <tbody>{dim_rows or '<tr><td colspan=5>无数据</td></tr>'}</tbody>
          </table>
        </section>"""

        if worst:
            cases_html = "".join(self._format_failure_case(c) for c in worst)
        else:
            cases_html = "<div class='ok-box'>✓ 无失败 case</div>"
        sec_failures = f"""
        <section class='card'>
          <h2>5. 典型失败 Case（前 {MAX_FAILURE_CASES} 个最低分）</h2>
          {cases_html}
        </section>"""

        recs_html = "".join(f"<li>{html_lib.escape(r)}</li>" for r in recs)
        sec_recs = f"""
        <section class='card'>
          <h2>6. 改进建议</h2>
          <ul class='recs'>{recs_html}</ul>
        </section>"""

        return f"""<!DOCTYPE html>
<html lang='zh-CN'>
<head>
<meta charset='UTF-8'>
<title>评测报告 · {html_lib.escape(self.task.name)}</title>
<style>{css}</style>
</head>
<body>
<header>
  <h1>外呼 Agent 评测报告</h1>
  <div class='subtitle'>{html_lib.escape(self.task.name)} · {html_lib.escape(self.run_id)}</div>
  <div class='subtitle muted'>生成时间 {html_lib.escape(now)}</div>
</header>
<main>
  {sec_target}
  {sec_scale}
  {sec_overall}
  {sec_dims}
  {sec_failures}
  {sec_recs}
  <footer>
    <p class='muted'>由 OutboundEval 自动生成 · 数据来源：{html_lib.escape(self.run_id)}</p>
  </footer>
</main>
</body>
</html>"""

    def _css(self) -> str:
        return """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans SC', sans-serif; background: #0f1117; color: #e8eaf0; line-height: 1.6; padding: 32px 16px; }
header { max-width: 960px; margin: 0 auto 24px; }
header h1 { font-size: 24px; font-weight: 700; margin-bottom: 4px; }
header .subtitle { font-size: 13px; color: #8b92a8; }
main { max-width: 960px; margin: 0 auto; display: flex; flex-direction: column; gap: 16px; }
.card { background: #161922; border: 1px solid #2a2e3b; border-radius: 8px; padding: 20px 24px; }
.card h2 { font-size: 16px; font-weight: 700; margin-bottom: 12px; color: #e8eaf0; }
.muted { color: #8b92a8; }
table.kv { width: 100%; border-collapse: collapse; }
table.kv th { text-align: left; padding: 4px 12px 4px 0; color: #8b92a8; font-weight: 500; width: 120px; font-size: 13px; }
table.kv td { padding: 4px 0; font-size: 13px; }
table.kv code { background: #0f1117; padding: 2px 6px; border-radius: 3px; font-size: 12px; }
.kpi-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 12px; }
.kpi { background: #1a1d27; border: 1px solid #2a2e3b; border-radius: 6px; padding: 16px; text-align: center; }
.kpi-num { font-size: 28px; font-weight: 700; margin-bottom: 4px; }
.kpi-label { font-size: 12px; color: #8b92a8; }
.kpi-pass .kpi-num { color: #22c55e; }
.kpi-fail .kpi-num { color: #ef4444; }
.dist-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 12px; }
.dist-grid h4 { font-size: 13px; font-weight: 600; margin-bottom: 6px; color: #8b92a8; }
table.dist, table.dims { width: 100%; border-collapse: collapse; font-size: 12px; }
table.dist th, table.dims th { text-align: left; padding: 6px 8px; background: #1a1d27; color: #8b92a8; font-weight: 500; }
table.dist td, table.dims td { padding: 6px 8px; border-top: 1px solid #2a2e3b; }
.risk-box, .ok-box { padding: 10px 14px; border-radius: 6px; font-size: 13px; margin-top: 8px; }
.risk-box { background: #2a1212; border: 1px solid #5a1f1f; color: #fca5a5; }
.risk-box ul { margin: 6px 0 0 18px; }
.ok-box { background: #0f2a1a; border: 1px solid #1a5a2e; color: #86efac; }
.case-card { background: #1a1d27; border: 1px solid #2a2e3b; border-radius: 6px; padding: 14px; margin-bottom: 12px; }
.case-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
.case-tag { background: #ef4444; color: #fff; padding: 2px 8px; border-radius: 3px; font-size: 11px; font-weight: 600; }
.persona-tag, .diff-tag { background: #2a2e3b; color: #8b92a8; padding: 2px 8px; border-radius: 3px; font-size: 11px; margin-left: 6px; }
.case-score { font-size: 12px; color: #8b92a8; }
.case-section-title { font-size: 11px; font-weight: 600; color: #8b92a8; margin: 10px 0 6px; }
.crit-list { display: flex; flex-direction: column; gap: 6px; }
.crit-item { padding: 8px 10px; border-radius: 4px; border-left: 3px solid; }
.crit-fail { background: #2a1212; border-left-color: #ef4444; }
.crit-head { display: flex; justify-content: space-between; align-items: center; gap: 8px; flex-wrap: wrap; }
.crit-name { font-weight: 600; color: #fca5a5; font-size: 13px; }
.crit-meta { font-size: 10px; color: #8b92a8; }
.crit-reason { font-size: 12px; color: #cbd5e1; margin-top: 4px; }
.evidence { font-family: 'SF Mono', Monaco, Menlo, monospace; font-size: 11px; color: #94a3b8; background: #0f1117; padding: 3px 8px; border-radius: 2px; margin-top: 3px; }
.empty { color: #5a6078; font-size: 12px; font-style: italic; padding: 4px 0; }
.viol-list { display: flex; flex-wrap: wrap; gap: 6px; }
.viol-chip { background: #2a1212; color: #fca5a5; padding: 3px 8px; border-radius: 3px; font-size: 11px; border: 1px solid #5a1f1f; }
.ok-chip { background: #0f2a1a; color: #86efac; padding: 3px 8px; border-radius: 3px; font-size: 11px; border: 1px solid #1a5a2e; }
.warn-chip { background: #2a1f0a; color: #fbbf24; padding: 3px 8px; border-radius: 3px; font-size: 11px; border: 1px solid #5a4310; }
.gold-card { background: #1a2333; border: 1px solid #2a3a55; border-radius: 6px; padding: 10px 14px; margin: 10px 0; }
.dialogue { display: flex; flex-direction: column; gap: 6px; max-height: 360px; overflow-y: auto; }
.turn { padding: 8px 12px; border-radius: 6px; font-size: 13px; line-height: 1.55; }
.turn-agent { background: #1a2a3a; border-left: 3px solid #3b82f6; }
.turn-user { background: #2a1f3a; border-left: 3px solid #8b5cf6; }
.turn-label { font-size: 10px; color: #8b92a8; display: block; margin-bottom: 4px; }
.turn-content { color: #e8eaf0; white-space: pre-wrap; }
.recs { margin-left: 20px; }
.recs li { padding: 4px 0; font-size: 13px; }
footer { margin-top: 16px; text-align: center; }
"""

    def _render_markdown(self) -> str:
        s = self._summary()
        dims = self._dimension_stats()
        worst = self._worst_cases()
        recs = self._recommendations()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        diff_display = (
            self.task.difficulty.value
            if hasattr(self.task.difficulty, "value")
            else str(self.task.difficulty)
        )

        out = [
            f"# 评测报告 · {self.task.name}",
            f"_{self.run_id} · 生成时间 {now}_",
            "",
            "## 1. 测试目标",
            f"- 任务：{self.task.name} (`{self.task.task_id}`)",
            f"- 难度：{diff_display}",
            f"- Skill：`{self.task.skill_name}`",
            "",
            "## 2. 评测规模",
            f"- 总 Case：**{s['total']}**",
            f"- 通过：{s['passed']} · 失败：{s['failed']} · 通过率：**{s['success_rate']:.1f}%**",
            "",
            "## 3. 总体结论",
            f"- 综合均分：{s['avg_score']:.1f}",
        ]
        if s["risks"]:
            out.append("- ⚠ 风险告警：")
            for r in s["risks"]:
                out.append(f"  - {r}")
        else:
            out.append("- ✓ 全部维度均值 ≥ 60 分")

        out += ["", "## 4. 维度分数", "| 维度 | N | 均分 | 标准差 | 95% CI |", "|---|---|---|---|---|"]
        for d in dims:
            out.append(
                f"| {d['zh']} ({d['en']}) | {d['n']} | {d['mean']:.1f} | {d['std']:.1f} | "
                f"[{d['ci_low']:.1f}, {d['ci_high']:.1f}] |"
            )

        out += ["", "## 5. 典型失败 Case"]
        if worst:
            for i, c in enumerate(worst, 1):
                out.append(f"### Case {i} · {c.get('persona_type', '')} · 分 {c.get('overall_score', 0):.1f}")
                details = c.get("criterion_details", []) or []
                failed = [d for d in details if not d.get("satisfied", True)]
                # 扣分明细
                if failed:
                    out.append("")
                    out.append("**📉 扣分明细（按扣分从大到小）**")
                    out.append("")
                    out.append("| 扣分 | 条件 | 优先级 | 扣分依据 |")
                    out.append("|---|---|---|---|")
                    sorted_failed = sorted(
                        failed,
                        key=lambda d: -(1 - float(d.get("score", 0))) * 100
                    )
                    for d in sorted_failed[:8]:
                        loss = (1 - float(d.get("score", 0))) * 100
                        out.append(
                            f"| **{loss:.0f}分** | {d.get('name', '')} | {d.get('priority', '')} | {d.get('reasoning', '')} |"
                        )
                # 评分依据追溯（所有 criterion）
                if details:
                    out.append("")
                    out.append("**📊 评分依据追溯（每条判定的方法、得分、证据）**")
                    out.append("")
                    for d in details[:MAX_EVIDENCE_PER_CASE]:
                        state = d.get("satisfied")
                        score_pct = float(d.get("score", 0)) * 100
                        if state:
                            tag = "✅ 满足"
                        elif score_pct >= 40:
                            tag = f"⚠ 部分 {score_pct:.0f}%"
                        else:
                            tag = f"❌ 未满足 {score_pct:.0f}%"
                        out.append(
                            f"- {tag} **{d.get('name', '')}** ({d.get('priority', '')} · {d.get('method', '')} · 得分 {score_pct:.0f})"
                        )
                        if d.get("reasoning"):
                            out.append(f"  - 依据：{d.get('reasoning', '')}")
                        for e in (d.get("evidence") or [])[:3]:
                            out.append(f"  - 证据：`{e}`")
                violations = c.get("failure_violations") or []
                if violations:
                    out.append("")
                    out.append("- 触发红线：" + "、".join(str(v) for v in violations))
                gold = c.get("gold_comparison")
                if gold:
                    out.append(
                        f"- 与标杆对比 (gold={gold.get('matched_gold_id', '-')}, quality={gold.get('gold_quality', '-')}): "
                        f"内容覆盖 {gold.get('coverage', 0) * 100:.0f}% · "
                        f"顺序对齐 {gold.get('sequence_alignment', 0) * 100:.0f}% · "
                        f"结果{'一致' if gold.get('outcome_match') is True else ('不一致' if gold.get('outcome_match') is False else '无信号')}"
                    )
                    for note in (gold.get("notes") or [])[:3]:
                        out.append(f"  - {note}")
        else:
            out.append("✓ 无失败 case")

        out += ["", "## 6. 改进建议"]
        for r in recs:
            out.append(f"- {r}")

        return "\n".join(out) + "\n"
