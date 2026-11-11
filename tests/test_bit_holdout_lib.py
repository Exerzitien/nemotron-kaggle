import csv
import json
import os

import bit_holdout_lib as L

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS = os.path.join(_ROOT, "research", "majortom_results.csv")
_TYPED = os.path.join(_ROOT, "research", "train_with_type.csv")


def test_bits_roundtrip():
    for v in (0, 1, 85, 170, 255):
        bits = L.int_to_bits(v)
        assert len(bits) == 8
        assert L.str_to_bits(L.bits_to_str(bits)) == bits


def test_signature_length_and_determinism():
    e2f = L.build_expr2func()
    pf = L.make_predictor("{A}", {"{A}": ("rot", 0)}, e2f)
    sig = L.signature(pf)
    assert len(sig) == 256 * 8
    assert L.signature(pf) == sig  # deterministic


def test_predictor_matches_solver_csv():
    """重建的 predictor 必须复现 solver 当初写进 CSV 的 query 预测。"""
    e2f = L.build_expr2func()
    rows = list(csv.DictReader(open(_RESULTS)))
    typed = list(csv.DictReader(open(_TYPED)))
    id2prompt = {r["id"]: r["prompt"] for r in typed if r["type"] == "bit_manipulation"}
    import majortom_bit_solver as S
    ok = tested = 0
    for r in rows[:60]:
        if not r["expr"]:
            continue
        trans = L.parse_trans(r["trans"])
        _, _, query = S.parse_bit_problem(id2prompt[r["id"]])
        if query is None:
            continue
        pred = L.make_predictor(r["expr"], trans, e2f)(query)
        tested += 1
        ok += (pred == r["pred"])
    assert tested >= 50
    assert ok == tested


def test_format_matches_real_prompt_byte_exact():
    """用真实 bit 题（id 00066667）的样例重建 prompt，必须逐字节相等。"""
    real = (
        "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit "
        "binary numbers. The transformation involves operations like bit shifts, "
        "rotations, XOR, AND, OR, NOT, and possibly majority or choice functions."
        "\n\nHere are some examples of input -> output:\n"
        "01010001 -> 11011101\n00001001 -> 01101101\n00010101 -> 01010101\n"
        "11111111 -> 10000001\n10011101 -> 01000101\n00111011 -> 00001001\n"
        "10111101 -> 00000101\n00100110 -> 10110011"
        "\n\nNow, determine the output for: 00110100"
    )
    examples = [
        ("01010001", "11011101"), ("00001001", "01101101"),
        ("00010101", "01010101"), ("11111111", "10000001"),
        ("10011101", "01000101"), ("00111011", "00001001"),
        ("10111101", "00000101"), ("00100110", "10110011"),
    ]
    assert L.format_bit_prompt(examples, "00110100") == real


def test_is_determinate_false_when_underdetermined():
    """单个全零样例无法定规则 -> 不确定。"""
    e2f = L.build_expr2func()
    id_sig = L.signature(L.make_predictor("{A}", {"{A}": ("rot", 0)}, e2f))
    assert L.is_determinate([("00000000", "00000000")], id_sig, e2f) is False


def test_is_determinate_true_for_real_rule():
    """10 个随机样例足以定一个真实 2-var 规则。"""
    import random
    e2f = L.build_expr2func()
    expr, trans = "XOR({B}, {A})", {"{A}": ("shl", 1), "{B}": ("shl", 7)}
    pf = L.make_predictor(expr, trans, e2f)
    tgt = L.signature(pf)
    rng = random.Random(1)
    exs = []
    for v in rng.sample(range(256), 10):
        q = L.int_to_bits(v)
        exs.append((L.bits_to_str(q), pf(q)))
    assert L.is_determinate(exs, tgt, e2f) is True


def test_load_train_signatures():
    e2f = L.build_expr2func()
    sigs = L.load_train_signatures(_RESULTS, e2f)
    assert isinstance(sigs, set)
    assert 700 <= len(sigs) <= 900  # 实测 765 distinct
    assert all(len(s) == 256 * 8 for s in list(sigs)[:5])


def test_generate_one_problem_valid():
    import random
    import majortom_bit_solver as S
    e2f = L.build_expr2func()
    rng = random.Random(7)
    # 契约：样例未唯一确定规则时返回 None，调用方重试（Task 5 编排即如此消费）。
    prob = None
    for _ in range(50):
        prob = L.generate_one_problem("XOR({B}, {A})",
                                      {"{A}": ("shl", 1), "{B}": ("shl", 7)},
                                      rng, e2f)
        if prob is not None:
            break
    assert prob is not None
    # 格式可被 solver 解析
    ins, outs, query = S.parse_bit_problem(prob["prompt"])
    assert query is not None and 7 <= len(ins) <= 10
    # 答案是 8-bit、与样例同源
    assert len(prob["answer"]) == 8 and set(prob["answer"]) <= {"0", "1"}
    # query 不在样例输入里（确属"新输入"）
    ex_inputs = {L.bits_to_str(x) for x in ins}
    assert L.bits_to_str(query) not in ex_inputs
    # 该题确定可解
    assert L.is_determinate(list(zip([L.bits_to_str(x) for x in ins],
                                     [L.bits_to_str(x) for x in outs])),
                            prob["target_sig"], e2f) is True
