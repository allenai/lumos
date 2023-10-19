import os
import re
import sys
import math
import wolframalpha
from eval.svamp.utils import *


class MathActions:
    def Calculator(x, isfunc):
        if isfunc:      # when facing "R1(a=1)"", or "R1(R2)", substitute
            func_args, func_exp = parse_func_args(x)
            x = func_exp + ", where " + func_args
        elif '):' in x:     # when facing operations between equations
            return combine_eqs(x)

        if x.isdigit():
            return x
        elif not isalphain(x):
            return WA_result(x)
        else:
            ops = ["Expand", "Calculate", ""]
            for op in ops:
                try:
                    result = WA_result(f"{op} {x}")
                    if "factor" in result.lower():
                        pos_explanation = result.find(" (")
                        result = result[:pos_explanation]
                    return result
                except:
                    continue

    def SetEquation(x):
        if ',' in x:
            x = x.replace(',', ", where")
            return MathActions.Calculator(x, False)
        return x
    
    def SolveEquation(x):
        return WA_result(f"Solve {x}")

    def SolveInequality(x):
        return WA_result(f"Solve {x}")

    def Define(x):
        return x

    def Count(x):
        items = x.split(', ')
        if "..." in items:      # assume as arithmetic sequence
            diff = WA_result(f"({items[1]}) - ({items[0]})")
            return WA_result(f"{WA_result(f'({items[-1]}) - ({items[0]})')} / {diff} + 1")
        else:
            return len(items)
    

if __name__ == "__main__":
    main()