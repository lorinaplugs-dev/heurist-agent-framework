import importlib
import pkgutil
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

package_name = "mesh.agents"
modules_checked = 0

try:
    package = importlib.import_module(package_name)
    module_infos = list(pkgutil.iter_modules([str(Path(package.__file__).parent)]))

    non_pkg_modules = [info for info in module_infos if not info.ispkg]
    total_modules = len(non_pkg_modules)

    if total_modules == 0:
        print("No modules found. ✅")
        sys.exit(0)

    for _, module_name, _ in non_pkg_modules:
        importlib.import_module(f"{package_name}.{module_name}")
        modules_checked += 1

    print(f"✅ All {modules_checked} modules imported successfully.")
    sys.exit(0)

except Exception as e:
    print(f"\n❌ FAILED after {modules_checked}/{total_modules} imports: {e}")
    sys.exit(1)
