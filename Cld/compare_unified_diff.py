from pathlib import Path
import difflib
p1 = Path('unified_adv_diff_gpr_residual.py').read_text().splitlines()
p2 = Path('unified_adv_diff_gpr_residual_complex.py').read_text().splitlines()
for line in difflib.unified_diff(p1, p2, fromfile='unified_adv_diff_gpr_residual.py', tofile='unified_adv_diff_gpr_residual_complex.py', n=3):
    print(line)
