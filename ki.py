# agent.py — Interaktiver CPU-Agent (Windows/Linux), komplett lokal
# Tools: Shell, Files, Excel (lesen/aggregieren/export), kleine Python-Snippets (pandas)

from smolagents import CodeAgent, TransformersModel, tool
from subprocess import run
from typing import Optional, List
import sys, os, json, io, contextlib
import pandas as pd

# ---------- Shell ----------
@tool
def sh(cmd: str) -> str:
    """Run a shell command.

    Args:
        cmd (str): Command line to execute via the system shell.

    Returns:
        str: Combined stdout and stderr as text.
    """
    try:
        r = run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        return (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        return f"error:{e}"

# ---------- Dateien ----------
@tool
def read_file(path: str, max_chars: int = 20000) -> str:
    """Read a UTF-8 text file and return (optionally truncated) content.

    Args:
        path (str): File path to read.
        max_chars (int): Maximum number of characters to return.

    Returns:
        str: File content (possibly truncated).
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
        return data[:max_chars] + ("...[truncated]" if len(data) > max_chars else "")
    except Exception as e:
        return f"error:{e}"

@tool
def write_file(path: str, content: str) -> str:
    """Write UTF-8 text to a file (create folders as needed).

    Args:
        path (str): Target file path.
        content (str): Text to write.

    Returns:
        str: Status string with saved path or error.
    """
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"saved:{path}"
    except Exception as e:
        return f"error:{e}"

# ---------- Excel / Daten ----------
@tool
def read_excel(path: str, sheet: Optional[str] = None, head: int = 5) -> str:
    """Load an Excel sheet and return structure info + preview as JSON.

    Args:
        path (str): Excel file path.
        sheet (str, optional): Sheet name or None for first sheet.
        head (int): Number of preview rows to include.

    Returns:
        str: JSON string with rows, columns, dtypes, na_counts, preview_markdown.
    """
    try:
        df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
        info = {
            "rows": int(len(df)),
            "columns": list(map(str, df.columns)),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
            "na_counts": {c: int(df[c].isna().sum()) for c in df.columns},
            "preview_markdown": df.head(head).to_markdown(index=False)
        }
        return json.dumps(info, ensure_ascii=False)
    except Exception as e:
        return f"error:{e}"

@tool
def excel_groupby(path: str, sheet: Optional[str], by: List[str], value: str, agg: str = "sum") -> str:
    """Group an Excel sheet by columns and aggregate a value column.

    Args:
        path (str): Excel file path.
        sheet (str, optional): Sheet name or None for first sheet.
        by (list[str]): Column names to group by.
        value (str): Value column to aggregate.
        agg (str): Aggregation function (e.g., 'sum', 'mean', 'count').

    Returns:
        str: Result table as Markdown.
    """
    try:
        df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
        g = df.groupby(by)[value].agg(agg).reset_index()
        return g.to_markdown(index=False)
    except Exception as e:
        return f"error:{e}"

@tool
def to_csv_from_excel(path: str, sheet: Optional[str], out_csv: str) -> str:
    """Export an Excel sheet to CSV.

    Args:
        path (str): Excel file path.
        sheet (str, optional): Sheet name or None for first sheet.
        out_csv (str): Output CSV path.

    Returns:
        str: Status string with saved path or error.
    """
    try:
        df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
        df.to_csv(out_csv, index=False)
        return f"saved:{out_csv}"
    except Exception as e:
        return f"error:{e}"

# ---------- Optional: kleine Python-Snippets ----------
@tool
def py(code: str) -> str:
    """Execute a short Python snippet with pandas available; return printed output.

    Args:
        code (str): Python code to execute (use 'pd' for pandas).

    Returns:
        str: Captured stdout or 'ok'.
    """
    ns = {"pd": pd, "json": json, "os": os}
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, ns, ns)
        out = buf.getvalue()
        return out if out else "ok"
    except Exception as e:
        return f"error:{e}"

# ---------- Modell (CPU, offline-freundlich) ----------
model = TransformersModel(
    model_id="Qwen/Qwen2.5-Coder-1.5B-Instruct",      # klein & flott auf CPU
    max_new_tokens=1024,
    # Kein device_map => keine Abhängigkeit zu 'accelerate'
    model_kwargs={
        "trust_remote_code": True,
        "torch_dtype": "float32"
    },
    tokenizer_kwargs={
        "trust_remote_code": True
    }
)

agent = CodeAgent(
    model=model,
    tools=[sh, read_file, write_file, read_excel, excel_groupby, to_csv_from_excel, py],
    add_base_tools=False
)

GUIDANCE = (
    "Use sh to explore the filesystem (Windows: 'dir', or PowerShell via 'powershell -Command ...'). "
    "Use read_file/write_file for text; read_excel/excel_groupby/to_csv_from_excel for spreadsheets; "
    "use py for short pandas snippets."
)

def main() -> int:
    # Interaktiv: Prompt abfragen
    print("🔧 Lokaler Agent bereit. Tipp 'exit' zum Beenden.")
    user_prompt = input("👉 Prompt: ").strip()
    if not user_prompt or user_prompt.lower() in ("exit", "quit"):
        print("Abbruch.")
        return 0

    print("\n⏳ Arbeite…\n")
    result = agent.run(user_prompt + "\n\n" + GUIDANCE)
    print("\n✅ Ergebnis:\n")
    print(result)
    return 0

if __name__ == "__main__":
    sys.exit(main())
