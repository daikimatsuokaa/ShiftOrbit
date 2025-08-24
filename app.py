# app.py
# -*- coding: utf-8 -*-
"""
シフト作成たたき台（Flask + OR-Tools）
- 3交代(A/B/C)、月次、6-3基本、月内は勤務帯固定、希望休考慮、CSV出力
- pip install flask ortools
"""

from __future__ import annotations
import os
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import date, datetime, time, timedelta
import calendar
from collections import defaultdict

from flask import Flask, request, render_template_string, redirect, url_for, make_response

# ==========
# データ構造
# ==========

@dataclass
class ShiftDef:
    name: str
    start: time
    end: time

    @property
    def hours(self) -> float:
        # 終了が翌日跨ぎ（例: 16:00-00:00）の場合にも対応
        dt0 = datetime.combine(date(2000, 1, 1), self.start)
        dt1 = datetime.combine(date(2000, 1, 1), self.end)
        if dt1 <= dt0:
            dt1 += timedelta(days=1)
        delta = dt1 - dt0
        return round(delta.total_seconds() / 3600.0, 2)

@dataclass
class Employee:
    name: str
    fixed_shift: str  # 'A' / 'B' / 'C'
    # 固定勤務回数（任意）: 指定なければ後で自動設定
    target_shifts: Optional[int] = None
    # 固定勤務時間（任意）: 指定があれば target_shifts に換算
    target_hours: Optional[float] = None
    # 希望休（日にちの配列: 1..31）
    preferred_off_days: List[int] = field(default_factory=list)


# ==========
# ユーティリティ
# ==========

def parse_hhmm(s: str) -> time:
    h, m = s.strip().split(":")
    return time(hour=int(h), minute=int(m))

def month_dates(year: int, month: int) -> List[date]:
    last = calendar.monthrange(year, month)[1]
    return [date(year, month, d) for d in range(1, last + 1)]

def nine_day_pattern(day_idx: int, offset: int) -> int:
    """
    6-3の基本パターン: 9日周期で [1,1,1,1,1,1,0,0,0]
    day_idx: 月初からの通算 0,1,2,...
    offset: 0..8
    return: 1(勤務可) / 0(休み)
    """
    return 1 if ((day_idx + offset) % 9) < 6 else 0

def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# ==========
# 核: スケジューラ（OR-Tools CP-SAT）
# ==========

def solve_month_schedule(
    year: int,
    month: int,
    shifts: Dict[str, ShiftDef],
    employees: List[Employee],
    coverage_per_shift: int = 1,
    preference_penalty: int = 5,
    time_limit_sec: int = 10,
) -> Tuple[Dict[date, Dict[str, List[str]]], Dict[str, str]]:
    """
    戻り値:
      - schedule[date][shift] = [name1, name2, ...] その日に当該シフトで勤務する人のリスト
      - info: 実行上のメッセージ、警告など
    """
    from ortools.sat.python import cp_model

    days = month_dates(year, month)
    D = len(days)

    # シフト→従業員リスト
    emp_by_shift: Dict[str, List[Employee]] = {'A': [], 'B': [], 'C': []}
    for e in employees:
        if e.fixed_shift not in emp_by_shift:
            raise ValueError(f"未知のシフト: {e.fixed_shift}")
        emp_by_shift[e.fixed_shift].append(e)

    info = {}

    # 6-3の最大可用勤務日数（offsetに依らず月内最大はおおむね ceil(D*6/9)）
    max_avail_per_person = (D * 6 + 8) // 9  # ceil(D*6/9)

    # 各シフトの必要総勤務コマ数（= 日数 × 必要人数）
    need_total_per_shift = {s: D * coverage_per_shift for s in ['A', 'B', 'C']}

    # 各シフトで人手が理論上足りるか（可用性基準）
    for s in ['A', 'B', 'C']:
        n = len(emp_by_shift[s])
        if n == 0:
            continue
        max_total_avail = n * max_avail_per_person
        if max_total_avail < need_total_per_shift[s]:
            info[s + "_warning"] = (
                f"{s}シフトの人員が不足しています。"
                f"必要総勤務コマ={need_total_per_shift[s]} に対し、"
                f"6-3可用上限={max_total_avail}（{n}名×{max_avail_per_person}）です。"
                f"→ 在籍者を増やすか、必要人数/6-3設定を見直してください。"
            )

    # 目標勤務回数（target_shifts）を埋める（指定がなければ均等割）
    for s in ['A', 'B', 'C']:
        group = emp_by_shift[s]
        if not group:
            continue

        hours_per_shift = shifts[s].hours
        # 既に target_shifts/target_hours 指定済の合計
        specified = 0
        for e in group:
            if e.target_shifts is None and e.target_hours is not None:
                e.target_shifts = int(round(e.target_hours / hours_per_shift))
            if e.target_shifts is not None:
                # 可用上限を超える指定は切り詰め
                if e.target_shifts > max_avail_per_person:
                    e.target_shifts = max_avail_per_person
                specified += e.target_shifts

        remaining = need_total_per_shift[s] - specified
        unspecified = [e for e in group if e.target_shifts is None]

        if remaining < 0:
            info[s + "_warning_over"] = (
                f"{s}シフト: 事前に指定された勤務回数の合計が必要総数を超えています。"
                f"超過={-remaining}。指定を見直してください。"
            )
            # 超過を許容しつつ目的関数で近づける実装も可能だが、ここは警告のみ。
            remaining = 0

        # 均等割（上限 max_avail_per_person を超えないように配分）
        if unspecified:
            base = remaining // len(unspecified)
            rem = remaining % len(unspecified)
            for i, e in enumerate(unspecified):
                e.target_shifts = min(max_avail_per_person, base + (1 if i < rem else 0))

        # もう一度合計を確認
        total_target = sum(e.target_shifts or 0 for e in group)
        if total_target != need_total_per_shift[s]:
            info[s + "_note_fill"] = (
                f"{s}シフト: 目標勤務回数の合計={total_target} が必要総数={need_total_per_shift[s]}と一致していません。"
                "（人手不足や上限制約のため） → 目的関数で可能な限り充足を試みます。"
            )

    # ---- CP-SAT モデル化 ----
    model = cp_model.CpModel()

    # 従業員インデックス化
    idx_by_shift: Dict[str, Dict[int, Employee]] = {}
    for s in ['A', 'B', 'C']:
        idx_by_shift[s] = {i: e for i, e in enumerate(emp_by_shift[s])}

    # 変数:
    # y[s, i, d] ∈ {0,1}: シフトsのi番目の社員が day d に勤務するか
    y = {}
    for s in ['A', 'B', 'C']:
        for i in idx_by_shift[s]:
            for d in range(D):
                y[(s, i, d)] = model.NewBoolVar(f"y_{s}_{i}_{d}")

    # z[s, i, o] ∈ {0,1}: 社員iの9日オフセットo（0..8）が選ばれたか（6-3パターン）
    z = {}
    for s in ['A', 'B', 'C']:
        for i in idx_by_shift[s]:
            for o in range(9):
                z[(s, i, o)] = model.NewBoolVar(f"z_{s}_{i}_{o}")
            # 1つだけ選ぶ
            model.Add(sum(z[(s, i, o)] for o in range(9)) == 1)

    # 各日×各シフトのカバレッジ（必要人数）を満たす
    for s in ['A', 'B', 'C']:
        for d in range(D):
            model.Add(sum(y[(s, i, d)] for i in idx_by_shift[s]) == coverage_per_shift)

    # 6-3可用性: y <= Σ_o z*pattern[o][d]
    for s in ['A', 'B', 'C']:
        for i in idx_by_shift[s]:
            for d in range(D):
                allowed = []
                for o in range(9):
                    allowed.append(z[(s, i, o)] * nine_day_pattern(d, o))
                model.Add(y[(s, i, d)] <= sum(allowed))

    # 個人の月間固定勤務回数（= target_shifts）
    # 実際には target_shifts 合計を必要総数に一致させてあるが、上限等で一致しない場合も目的関数で近似。
    # ここでは「できるだけ近づける」ために等式ではなくソフト拘束にする（偏差を最小化）。
    deviations = []
    for s in ['A', 'B', 'C']:
        for i, e in idx_by_shift[s].items():
            target = e.target_shifts or 0
            actual = sum(y[(s, i, d)] for d in range(D))
            # 偏差変数 v >= |actual - target|
            v = model.NewIntVar(0, D, f"dev_{s}_{i}")
            # 2つの線形不等式で絶対値を表現
            model.Add(actual - target <= v)
            model.Add(target - actual <= v)
            deviations.append(v)

    # 目的関数: 希望休に勤務が入るペナルティ + 目標勤務回数偏差 の最小化
    penalty_terms = []
    for s in ['A', 'B', 'C']:
        for i, e in idx_by_shift[s].items():
            prefs = set(d for d in e.preferred_off_days if 1 <= d <= D)
            for d in range(D):
                if (d + 1) in prefs:
                    # 希望休に勤務が入ればペナルティ
                    penalty_terms.append(preference_penalty * y[(s, i, d)])
    # 偏差に重み
    penalty_terms += [2 * v for v in deviations]

    model.Minimize(sum(penalty_terms))

    # ソルバー設定
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_sec)
    solver.parameters.num_search_workers = 8  # 並列

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("実行可能な解が見つかりませんでした。人員や条件を見直してください。")

    # 結果を整形
    schedule: Dict[date, Dict[str, List[str]]] = {dt: {'A': [], 'B': [], 'C': []} for dt in days}
    for s in ['A', 'B', 'C']:
        for i, e in idx_by_shift[s].items():
            for d, day in enumerate(days):
                if solver.Value(y[(s, i, d)]) == 1:
                    schedule[day][s].append(e.name)

    # 追加情報（満たせなかった希望休や偏差の統計など）
    violated_prefs = 0
    total_prefs = 0
    for s in ['A', 'B', 'C']:
        for i, e in idx_by_shift[s].items():
            prefs = set(d for d in e.preferred_off_days if 1 <= d <= D)
            total_prefs += len(prefs)
            for d in prefs:
                if solver.Value(y[(s, i, d - 1)]) == 1:
                    violated_prefs += 1
    info["preference"] = f"希望休 総数={total_prefs}, うち違反={violated_prefs}"

    return schedule, info


# ==========
# Flask アプリ
# ==========

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev")

# 生成したCSVを一時保持（ダウンロード用）
CSV_STORE: Dict[str, str] = {}

INDEX_HTML = r"""
<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <title>シフト作成（6-3 / A-B-C）たたき台</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Hiragino Sans", "Noto Sans JP", "Yu Gothic", sans-serif; margin: 24px; }
    header { margin-bottom: 16px; }
    h1 { font-size: 20px; margin: 0 0 8px 0; }
    .card { border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
    label { display: block; font-weight: 600; margin: 8px 0 4px; }
    input[type="text"], input[type="number"], select { padding: 8px; width: 100%; box-sizing: border-box; }
    table { width: 100%; border-collapse: collapse; margin-top: 8px; }
    th, td { border: 1px solid #eee; padding: 6px 8px; text-align: left; }
    th { background: #fafafa; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
    .btn { display: inline-block; padding: 10px 14px; border: 1px solid #444; border-radius: 6px; background: #fff; cursor: pointer; }
    .btn.primary { background: #222; color: #fff; border-color: #222; }
    .btn.small { padding: 6px 8px; font-size: 12px; }
    .muted { color: #666; font-size: 12px; }
    .warn { background: #fff6f6; border-color: #f2dede; color: #b94a48; padding: 10px; border-radius: 6px; margin-top: 8px; }
    .ok { background: #f6fff6; border-color: #def2de; color: #2f7d32; padding: 10px; border-radius: 6px; margin-top: 8px; }
    code.k { background: #f4f4f4; padding: 1px 4px; border-radius: 4px; }
  </style>
</head>
<body>
<header>
  <h1>シフト自動作成（A/B/C・6連勤3休・月内固定）</h1>
  <p class="muted">まずは要件検証用の簡易版です。Codexでのブラッシュアップ前提。</p>
</header>

<form method="post" action="{{ url_for('generate') }}">
  <div class="card">
    <h2>期間・カバレッジ</h2>
    <div class="grid">
      <div>
        <label>年 (YYYY)</label>
        <input type="number" name="year" value="{{year or ''}}" required>
      </div>
      <div>
        <label>月 (1-12)</label>
        <input type="number" name="month" value="{{month or ''}}" min="1" max="12" required>
      </div>
      <div>
        <label>各シフトの必要人数（毎日同一）</label>
        <input type="number" name="coverage" value="1" min="1" required>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>シフト設定（勤務帯）</h2>
    <div class="grid">
      <div>
        <label>A 勤務（開始）</label>
        <input type="text" name="A_start" value="01:00" placeholder="HH:MM" required>
      </div>
      <div>
        <label>A 勤務（終了）</label>
        <input type="text" name="A_end" value="10:00" placeholder="HH:MM" required>
      </div>
      <div>
        <label>B 勤務（開始）</label>
        <input type="text" name="B_start" value="09:00" placeholder="HH:MM" required>
      </div>
      <div>
        <label>B 勤務（終了）</label>
        <input type="text" name="B_end" value="18:00" placeholder="HH:MM" required>
      </div>
      <div>
        <label>C 勤務（開始）</label>
        <input type="text" name="C_start" value="17:00" placeholder="HH:MM" required>
      </div>
      <div>
        <label>C 勤務（終了）</label>
        <input type="text" name="C_end" value="02:00" placeholder="HH:MM" required>
      </div>
    </div>
    <p class="muted">※ 終了が開始以下なら翌日跨ぎとして扱います（例: 16:00→00:00）。</p>
  </div>

  <div class="card">
    <h2>従業員</h2>
    <p class="muted">行を追加して <code class="k">名前</code> / <code class="k">シフト(A/B/C/auto)</code> / <code class="k">月間固定勤務回数(任意)</code> / <code class="k">希望休(例: 3,12,25)</code> を入力してください。</p>
    <table id="emp-table">
      <thead>
        <tr>
          <th>名前</th>
          <th>シフト</th>
          <th>月間固定勤務回数（任意）</th>
          <th>希望休（日にち,カンマ区切り）</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        {% for i in range(4) %}
        <tr>
          <td><input type="text" name="emp_name" placeholder="山田 太郎"></td>
          <td>
            <select name="emp_shift">
              <option value="auto" selected>auto</option>
              <option value="A">A</option>
              <option value="B">B</option>
              <option value="C">C</option>
            </select>
          </td>
          <td><input type="number" name="emp_target_shifts" placeholder="未入力なら自動"></td>
          <td><input type="text" name="emp_prefs" placeholder="例: 3,12,25"></td>
          <td><button class="btn small" onclick="removeRow(event)">削除</button></td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
    <div style="margin-top:8px;">
      <button class="btn small" onclick="addRow(event)">＋ 行を追加</button>
    </div>
  </div>

  <div>
    <button class="btn primary" type="submit">スケジュールを作成</button>
  </div>
</form>

<script>
function addRow(e) {
  e.preventDefault();
  const tbody = document.querySelector("#emp-table tbody");
  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td><input type="text" name="emp_name" placeholder="氏名"></td>
    <td>
      <select name="emp_shift">
        <option value="auto" selected>auto</option>
        <option value="A">A</option>
        <option value="B">B</option>
        <option value="C">C</option>
      </select>
    </td>
    <td><input type="number" name="emp_target_shifts" placeholder="未入力なら自動"></td>
    <td><input type="text" name="emp_prefs" placeholder="例: 5,6,17"></td>
    <td><button class="btn small" onclick="removeRow(event)">削除</button></td>
  `;
  tbody.appendChild(tr);
}
function removeRow(e) {
  e.preventDefault();
  const tr = e.target.closest('tr');
  tr.parentNode.removeChild(tr);
}
</script>

{% if schedule %}
<hr>
<div class="card">
  <h2>作成結果</h2>

  {% if info %}
    {% for k, v in info.items() %}
      {% if "warning" in k %}
        <div class="warn">{{ v }}</div>
      {% else %}
        <div class="ok">{{ v }}</div>
      {% endif %}
    {% endfor %}
  {% endif %}

  <p><a class="btn" href="{{ url_for('download_csv', key=csv_key) }}">CSVをダウンロード</a></p>

  <table>
    <thead>
      <tr>
        <th>日付</th>
        <th>A 勤務</th>
        <th>B 勤務</th>
        <th>C 勤務</th>
      </tr>
    </thead>
    <tbody>
      {% for dt, row in schedule.items() %}
      <tr>
        <td>{{ dt.strftime("%Y-%m-%d (%a)") }}</td>
        <td>{{ ", ".join(row['A']) }}</td>
        <td>{{ ", ".join(row['B']) }}</td>
        <td>{{ ", ".join(row['C']) }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
</div>
{% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML, year=None, month=None, schedule=None)

def _autofill_shifts(emp_rows: List[Tuple[str, str, Optional[int], List[int]]]) -> List[Employee]:
    """
    emp_rows: (name, shift_raw, target_shifts, prefs_days)
    shift_raw が 'auto' の場合は A/B/C へ均等に自動配分
    """
    auto_indices = [i for i, (_, s, _, _) in enumerate(emp_rows) if s == "auto"]
    fixed_counts = {'A': 0, 'B': 0, 'C': 0}
    for _, s, _, _ in emp_rows:
        if s in fixed_counts:
            fixed_counts[s] += 1

    # 少ない順に auto を埋める
    order = []
    for _ in auto_indices:
        s = min(fixed_counts, key=lambda k: fixed_counts[k])
        order.append(s)
        fixed_counts[s] += 1

    employees: List[Employee] = []
    auto_ptr = 0
    for name, s, t, prefs in emp_rows:
        if not name:
            continue
        if s == "auto":
            s = order[auto_ptr]
            auto_ptr += 1
        employees.append(Employee(name=name, fixed_shift=s, target_shifts=t, preferred_off_days=prefs))
    return employees

@app.route("/generate", methods=["POST"])
def generate():
    try:
        year = int(request.form.get("year"))
        month = int(request.form.get("month"))
        coverage = int(request.form.get("coverage"))

        A = ShiftDef("A", parse_hhmm(request.form.get("A_start")), parse_hhmm(request.form.get("A_end")))
        B = ShiftDef("B", parse_hhmm(request.form.get("B_start")), parse_hhmm(request.form.get("B_end")))
        C = ShiftDef("C", parse_hhmm(request.form.get("C_start")), parse_hhmm(request.form.get("C_end")))
        shifts = {"A": A, "B": B, "C": C}

        names = request.form.getlist("emp_name")
        shifts_raw = request.form.getlist("emp_shift")
        targets_raw = request.form.getlist("emp_target_shifts")
        prefs_raw = request.form.getlist("emp_prefs")

        emp_rows: List[Tuple[str, str, Optional[int], List[int]]] = []
        for name, sraw, traw, praw in zip(names, shifts_raw, targets_raw, prefs_raw):
            if not name and not sraw and not traw and not praw:
                continue
            t = int(traw) if traw.strip() != "" else None
            prefs = []
            if praw.strip():
                for token in praw.replace("、", ",").split(","):
                    token = token.strip()
                    if token.isdigit():
                        prefs.append(int(token))
            emp_rows.append((name.strip(), (sraw or "auto"), t, prefs))

        employees = _autofill_shifts(emp_rows)

        # スケジューリング
        schedule, info = solve_month_schedule(
            year=year, month=month, shifts=shifts, employees=employees, coverage_per_shift=coverage
        )

        # CSV 生成（縦持ち）
        lines = ["date,shift,employees,start,end,hours"]
        for dt, row in schedule.items():
            for s in ["A", "B", "C"]:
                staff = ";".join(row[s]) if row[s] else ""
                sd = shifts[s]
                line = f"{dt.isoformat()},{s},{staff},{sd.start.strftime('%H:%M')},{sd.end.strftime('%H:%M')},{sd.hours}"
                lines.append(line)
        csv_text = "\n".join(lines)
        key = uuid.uuid4().hex
        CSV_STORE[key] = csv_text

        return render_template_string(
            INDEX_HTML,
            year=year,
            month=month,
            schedule=schedule,
            info=info,
            csv_key=key
        )

    except Exception as e:
        return render_template_string(
            INDEX_HTML,
            year=request.form.get("year"),
            month=request.form.get("month"),
            schedule=None,
        ) + f"<div class='warn'>エラー: {str(e)}</div>"

@app.route("/download/<key>.csv", methods=["GET"])
def download_csv(key: str):
    csv_text = CSV_STORE.get(key)
    if not csv_text:
        return "Not Found", 404
    resp = make_response(csv_text)
    resp.headers["Content-Type"] = "text/csv; charset=utf-8"
    resp.headers["Content-Disposition"] = f'attachment; filename="schedule_{key}.csv"'
    return resp

if __name__ == "__main__":
    app.run(debug=True)
