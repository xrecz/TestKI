# agent.py — Interaktiver Offline-Agent (Ubuntu/CPU)
from smolagents import CodeAgent, OpenAIServerModel, tool
from subprocess import run
from typing import Optional, List
import sys, os, json, io, contextlib
import pandas as pd

@tool
def sh(cmd: str) -> str:
    """Run a shell command.
    Args:
        cmd (str): Command line to execute via the system shell.
    Returns:
        str: Combined stdout+stderr.
    """
    try:
        r = run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        return (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        return f"error:{e}"

@tool
def read_file(path: str, max_chars: int = 20000) -> str:
    """Read a UTF-8 text file (truncated).
    Args:
        path (str): File path.
        max_chars (int): Max chars to return.
    Returns:
        str: File content.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
        return data[:max_chars] + ("...[truncated]" if len(data) > max_chars else "")
    except Exception as e:
        return f"error:{e}"

@tool
def write_file(path: str, content: str) -> str:
    """Write UTF-8 text to a file.
    Args:
        path (str): Target file path.
        content (str): Text to write.
    Returns:
        str: Status.
    """
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"saved:{path}"
    except Exception as e:
        return f"error:{e}"

@tool
def read_excel(path: str, sheet: Optional[str] = None, head: int = 5) -> str:
    """Load Excel and return structure+preview as JSON.
    Args:
        path (str): Excel file path.
        sheet (str, optional): Sheet name or None.
        head (int): Preview rows.
    Returns:
        str: JSON with rows, columns, dtypes, na_counts, preview_markdown.
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
    """Group Excel by columns and aggregate a value column.
    Args:
        path (str): Excel file path.
        sheet (str, optional): Sheet name or None.
        by (list[str]): Group-by columns.
        value (str): Value column.
        agg (str): Aggregation ('sum','mean','count',...).
    Returns:
        str: Markdown table.
    """
    try:
        df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
        g = df.groupby(by)[value].agg(agg).reset_index()
        return g.to_markdown(index=False)
    except Exception as e:
        return f"error:{e}"

@tool
def to_csv_from_excel(path: str, sheet: Optional[str], out_csv: str) -> str:
    """Export Excel sheet to CSV.
    Args:
        path (str): Excel file path.
        sheet (str, optional): Sheet name or None.
        out_csv (str): Output CSV path.
    Returns:
        str: Status.
    """
    try:
        df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
        df.to_csv(out_csv, index=False)
        return f"saved:{out_csv}"
    except Exception as e:
        return f"error:{e}"

model = OpenAIServerModel(
    model_id="local-gguf",                  # muss zu --alias passen
    api_base="http://127.0.0.1:8080/v1",    # lokaler llama.cpp-Server
    api_key="sk-local",                     # Dummy, wird nicht geprüft
    temperature=0.2,
    max_tokens=1024,
)

agent = CodeAgent(
    model=model,
    tools=[sh, read_file, write_file, read_excel, excel_groupby, to_csv_from_excel],
    add_base_tools=False
)

GUIDANCE = (
    "Use sh to explore the filesystem. "
    "Use read_file/write_file for text; read_excel/excel_groupby/to_csv_from_excel for spreadsheets."
)

def main() -> int:
    print("🔒 Offline-Agent bereit (localhost). 'exit' zum Beenden.")
    prompt = input("👉 Prompt: ").strip()
    if not prompt or prompt.lower() in ("exit","quit"):
        print("Abbruch."); return 0
    print("\n⏳ Arbeite…\n")
    out = agent.run(prompt + "\n\n" + GUIDANCE)
    print("\n✅ Ergebnis:\n")
    print(out)
    return 0

if __name__ == "__main__":
    sys.exit(main())
