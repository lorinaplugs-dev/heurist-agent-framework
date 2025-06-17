import asyncio
import subprocess
from pathlib import Path


async def run_file(file_path):
    print(f"Executing {file_path.name}")
    process = await asyncio.create_subprocess_exec(
        "python", str(file_path), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    await process.wait()
    print(f"Completed {file_path.name}")


async def main():
    current_dir = Path(__file__).parent
    py_files = [f for f in current_dir.glob("*.py") if f.name != "__init__.py" and f.name != Path(__file__).name]

    for py_file in py_files:
        await run_file(py_file)


if __name__ == "__main__":
    asyncio.run(main())
