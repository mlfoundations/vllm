"""#237 Stage 3 G-CAPTURE comparator: np.array_equal(pre, post) for every
captured routed_experts array. Pre = stock SIF capturer, Post = FIX-B patched.
Exit 0 + RESULT=EQUAL only if every shared key matches exactly (int ids, no tol).
"""

import sys

import numpy as np


def main(pre_path: str, post_path: str) -> int:
    pre = np.load(pre_path)
    post = np.load(post_path)
    pre_keys = set(pre.files)
    post_keys = set(post.files)
    if pre_keys != post_keys:
        print(f"RESULT=KEYMISMATCH pre={sorted(pre_keys)} post={sorted(post_keys)}")
        return 1
    if not pre_keys:
        print("RESULT=EMPTY no captured arrays in either side")
        return 1
    all_eq = True
    for k in sorted(pre_keys):
        a, b = pre[k], post[k]
        if a.shape != b.shape:
            print(f"  {k}: SHAPE pre={a.shape} post={b.shape} -> MISMATCH")
            all_eq = False
            continue
        eq = np.array_equal(a, b)
        ndiff = int((a != b).sum()) if not eq else 0
        print(f"  {k}: shape={a.shape} array_equal={eq} ndiff={ndiff}")
        all_eq = all_eq and eq
    print(f"RESULT={'EQUAL' if all_eq else 'MISMATCH'}")
    return 0 if all_eq else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
