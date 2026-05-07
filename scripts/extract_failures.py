import os
import pandas as pd

INPUT = 'testcase_final.csv'
OUT_DIR = 'test_failure_runs'
TOP_N = 5


def find_eval_col(cols):
    for c in cols:
        if str(c).strip().lower() == 'evaluation':
            return c
    for c in cols:
        if 'evaluation' in str(c).lower():
            return c
    return None


def find_total_col(cols):
    for c in cols:
        if 'total response time' in str(c).lower():
            return c
    for c in cols:
        if 'total' in str(c).lower() and 'ms' in str(c).lower():
            return c
    return None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(INPUT, encoding='utf-8-sig', dtype=str)

    eval_col = find_eval_col(df.columns)
    total_col = find_total_col(df.columns)

    if eval_col is None:
        print('No Evaluation column found. Exiting.')
        return

    # Normalize and filter FAIL rows
    df[eval_col] = df[eval_col].fillna('').astype(str)
    fails = df[df[eval_col].str.strip().str.upper() == 'FAIL'].copy()

    # Also include rows where Actual Result indicates 'Không tìm thấy' or similar
    act_cols = [c for c in df.columns if 'actual' in str(c).lower()]
    if act_cols:
        act = act_cols[0]
        fails_extra = df[df[act].fillna('').astype(str).str.contains('không tìm', case=False, na=False)]
        fails = pd.concat([fails, fails_extra]).drop_duplicates()

    if fails.empty:
        print('No failing rows found.')
        return

    # Convert total latency
    if total_col:
        def to_float(x):
            try:
                return float(str(x).replace(',', ''))
            except:
                return 0.0
        fails['_total_ms'] = fails[total_col].fillna('').apply(to_float)
        fails = fails.sort_values('_total_ms', ascending=False)
    else:
        fails = fails

    top = fails.head(TOP_N)

    commands = []
    for _, row in top.iterrows():
        tcid = str(row.get('Test Case ID') or row.get('Test Case Id') or row.get('Test Case') or '').strip()
        if not tcid:
            # fallback to first column value
            tcid = str(row.iloc[0])[:20].replace(' ', '_')

        out_csv = os.path.join(OUT_DIR, f'failing_{tcid}.csv')
        row.to_frame().T.to_csv(out_csv, index=False, encoding='utf-8-sig')

        report = os.path.join(OUT_DIR, f'report_{tcid}.csv')
        # Use venv python command for Windows environment used in this repo
        cmd = f"& .\\.venv312\\Scripts\\python.exe chatbot\\test_e2e.py {out_csv} {report}"
        commands.append(cmd)

    # write commands file
    cmds_path = os.path.join(OUT_DIR, 'run_failed_commands.ps1')
    with open(cmds_path, 'w', encoding='utf-8') as f:
        f.write('# PowerShell commands to re-run failing test cases\n')
        for c in commands:
            f.write(c + '\n')

    print('Wrote per-case CSVs to', OUT_DIR)
    print('Run commands saved to', cmds_path)


if __name__ == '__main__':
    main()
