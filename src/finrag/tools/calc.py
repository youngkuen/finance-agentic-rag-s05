# -*- coding: utf-8 -*-
"""결정적 계산 함수 모음.

원칙: LLM 은 산식을 만들지 않는다. LLM 이 하는 일은 문서에서 파라미터(요율·구간·
최저이율·절사 규칙·상한)와 근거 조항 ID 를 뽑는 것뿐이고, 계산은 여기 있는 순수
함수가 한다. 같은 "중도해지이율"이라도 상품마다 산식이 세 가지라서, 모델이 산식을
지어내면 그럴듯하게 틀린다.

절사 규칙과 일수 규칙은 기본값에 기대지 말고 명시 파라미터로 받는다. 답이 바뀐다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

from . import constants as C

Rounding = Literal["truncate_won", "round_won", "none"]


# ── 절사·반올림 ──────────────────────────────────────────────────────
def trunc_won(x: float) -> int:
    return int(math.floor(x + 1e-9))


def apply_rounding(x: float, rule: Rounding = "truncate_won") -> float:
    if rule == "truncate_won":
        return trunc_won(x)
    if rule == "round_won":
        return int(round(x))
    return x


def trunc_rate(x: float, precision: int = 3) -> float:
    """이율 절사. 농협 설명서: '소수 셋째자리까지 적용(넷째자리에서 절사)'."""
    f = 10 ** precision
    return math.floor(x * f + 1e-9) / f


@dataclass
class CalcResult:
    value: float | int | None
    unit: str
    steps: list[str] = field(default_factory=list)
    rules: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    abstain_reason: str | None = None

    def to_dict(self) -> dict:
        return {"value": self.value, "unit": self.unit, "steps": self.steps,
                "rules": self.rules, "warnings": self.warnings,
                "abstain_reason": self.abstain_reason}


def _missing(*pairs) -> list[str]:
    return [name for name, v in pairs if v is None]


# ── 적금·예금 (5회차 라이브 구현 대상) ────────────────────────────────
def savings_early_termination_rate(
    *, base_rate_pct: float | None, months_elapsed: int | None = None,
    months_contract: int | None = None, days_elapsed: int | None = None,
    days_contract: int | None = None, ratio: float | None = None,
    fixed_rate_pct: float | None = None, min_rate_pct: float = 0.0,
    deduction_pct: float | None = None, rate_precision: int = 3,
    month_rule: Literal["floor", "ceil"] = "floor",
) -> CalcResult:
    """중도해지이율. 산식 형태 세 가지를 한 함수로 받는다.

      A 차감률형   : 기본이율 × (1 − 차감률) × 경과월수/계약월수   (신한 오락실적금)
      B 구간 적용률형: 기준이율 × 적용률 × 경과월수/계약월수        (농협·KB Star)
      C 일할 적용률형: 기본금리 × 적용률 × 경과일수/계약일수        (카카오뱅크)

    구간표의 첫 행이 산식이 아니라 고정값인 경우가 많다(농협 '3개월 미만 0.1%').
    그때는 fixed_rate_pct 로 받는다. 산식을 억지로 적용하면 틀린다.
    """
    rules = {"rate_precision": rate_precision, "month_rule": month_rule, "min_rate_pct": min_rate_pct}
    steps: list[str] = []
    warnings: list[str] = []

    # 구간표 첫 행이 고정값이면 산식을 돌리지 않는다.
    if fixed_rate_pct is not None:
        value = max(fixed_rate_pct, min_rate_pct)
        steps.append(f"해당 구간은 산식이 아니라 고정이율 {fixed_rate_pct}% 를 적용합니다")
        if value != fixed_rate_pct:
            warnings.append(f"고정이율 {fixed_rate_pct}% 가 최저보장 {min_rate_pct}% 보다 작아 최저보장을 적용했습니다")
            steps.append(f"최저보장 {min_rate_pct}% 적용 → {value}%")
        return CalcResult(value, "%", steps, {**rules, "form": "fixed"}, warnings)

    if base_rate_pct is None:
        return CalcResult(None, "%", rules=rules,
                          abstain_reason="기본이율(기준이율)을 문서에서 찾지 못했습니다. 개별 상품 확인이 필요합니다.")

    # 곱하는 수: 적용률(형태 B·C) 또는 1 − 차감률(형태 A)
    if ratio is not None:
        mult, mult_label, form = ratio, f"적용률 {ratio:.0%}", "B/C"
    elif deduction_pct is not None:
        mult = 1 - deduction_pct / 100
        mult_label, form = f"(1 − 차감률 {deduction_pct}%)", "A"
    else:
        return CalcResult(None, "%", rules=rules,
                          abstain_reason="적용률·차감률·고정이율 중 어느 것도 문서에서 찾지 못했습니다.")

    # 기간 비율: 일할(형태 C)이 있으면 일수, 아니면 월수
    if days_elapsed is not None and days_contract:
        period, period_label = days_elapsed / days_contract, f"{days_elapsed}일/{days_contract}일"
        form = "C" if form != "A" else form
    elif months_elapsed is not None and months_contract:
        m = math.floor(months_elapsed + 1e-9) if month_rule == "floor" else math.ceil(months_elapsed - 1e-9)
        if m != months_elapsed:
            steps.append(f"경과월수 {months_elapsed} → 월 미만 {'버림' if month_rule == 'floor' else '절상'} {m}개월")
        period, period_label = m / months_contract, f"{m}개월/{months_contract}개월"
        form = "B" if form == "B/C" else form
    else:
        missing = _missing(("경과기간", days_elapsed if days_elapsed is not None else months_elapsed),
                           ("계약기간", days_contract or months_contract))
        return CalcResult(None, "%", rules=rules,
                          abstain_reason=f"{', '.join(missing) or '기간'} 을(를) 문서·질문에서 찾지 못했습니다.")

    raw = base_rate_pct * mult * period
    value = trunc_rate(raw, rate_precision)
    steps.append(f"{base_rate_pct}% × {mult_label} × {period_label} = {raw:.6f}%")
    steps.append(f"소수 {rate_precision}째 자리까지 (아래 절사) → {value}%")
    if value < min_rate_pct:
        warnings.append(f"산출 이율 {value}% 가 최저보장 {min_rate_pct}% 보다 작아 최저보장을 적용했습니다")
        steps.append(f"최저보장 {min_rate_pct}% 적용 → {min_rate_pct}%")
        value = min_rate_pct
    return CalcResult(value, "%", steps, {**rules, "form": form}, warnings)


def savings_maturity_interest(
    *, amount: int, rate_pct: float, deposit_dates: list[str], maturity: str,
    day_count: int = 365, per_deposit_truncate: bool = False,
    tax: Literal["none", "lump", "per_component"] = "none",
) -> CalcResult:
    """적금 만기이자. 입금 건별로 일할 계산해 더한다.

    농협 산식: 입금금액 × 약정이자율 × 일수(입금일~만기일 전일) / 365, 합계 원미만 절사.
    "합계 절사"와 "건별 절사 후 합계"가 다르다(97,890 vs 97,884). 문서가 어느 쪽인지
    명시하지 않으면 둘 다 제시하고 어느 규칙을 썼는지 밝힌다.
    """
    mat = date.fromisoformat(maturity)
    raw_total, trunc_total, steps = 0.0, 0, []
    for d in deposit_dates:
        days = (mat - date.fromisoformat(d)).days
        v = amount * rate_pct / 100 * days / day_count
        raw_total += v
        trunc_total += trunc_won(v)
    gross = trunc_total if per_deposit_truncate else trunc_won(raw_total)
    steps.append(f"입금 {len(deposit_dates)}건, 일수 합계 "
                 f"{sum((mat - date.fromisoformat(d)).days for d in deposit_dates)}일")
    steps.append(("건별 절사 후 합계" if per_deposit_truncate else "합계 후 원미만 절사")
                 + f" → {gross:,}원")

    warnings = [f"절사 규칙에 따라 총액 절사 {trunc_won(raw_total):,}원 / "
                f"건별 절사 {trunc_total:,}원으로 갈립니다."] if trunc_won(raw_total) != trunc_total else []

    value = gross
    if tax != "none":
        rate = C.get("tax.interest_income.total_rate", 0.154)
        if tax == "lump":
            value = gross - trunc_won(gross * rate)
            steps.append(f"이자소득세 {rate:.1%} 일괄 절사 → {value:,}원")
        else:
            income = trunc_won(gross * C.get("tax.interest_income.components.income_tax", 0.14))
            local = trunc_won(income * 0.10)
            value = gross - income - local
            steps.append(f"소득세 {income:,}원 + 지방소득세 {local:,}원 차감 → {value:,}원")
        warnings.append(C.disclaimer())
    return CalcResult(value, "원", steps,
                      {"day_count": day_count, "per_deposit_truncate": per_deposit_truncate,
                       "tax": tax}, warnings)


def deposit_interest(*, principal: int, rate_pct: float, months: int,
                     mode: Literal["simple", "monthly_compound"] = "simple",
                     tax: Literal["none", "lump"] = "none") -> CalcResult:
    """예금 이자. 단리/월복리.

    우리금융저축은행 산식:
      단리   원금 × 약정이율 × 1/12 (월단위)
      월복리 원금 × {(1 + 약정이율/12)^n − 1}
    """
    r = rate_pct / 100
    if mode == "simple":
        raw = principal * r * months / 12
        steps = [f"{principal:,} × {rate_pct}% × {months}/12 = {raw:,.2f}"]
    else:
        raw = principal * ((1 + r / 12) ** months - 1)
        steps = [f"{principal:,} × {{(1 + {rate_pct}%/12)^{months} − 1}} = {raw:,.4f}"]
    gross = trunc_won(raw)
    steps.append(f"원미만 절사 → {gross:,}원")
    value, warnings = gross, []
    if tax == "lump":
        rate = C.get("tax.interest_income.total_rate", 0.154)
        value = gross - trunc_won(gross * rate)
        steps.append(f"이자소득세 {rate:.1%} → {value:,}원")
        warnings.append(C.disclaimer())
    return CalcResult(value, "원", steps, {"mode": mode, "tax": tax}, warnings)


# ── 대출 (5회차 과제 / 완성본 제공) ───────────────────────────────────
def prepayment_fee(
    *, amount: int, fee_rate_pct: float | None, days_remaining: int | None = None,
    days_total: int | None = None, remaining_ratio: float | None = None,
    leap_year: bool = False, waiver_months: int | None = None,
    months_to_maturity: float | None = None, exempt_amount: int = 0,
    rounding: Rounding = "truncate_won",
) -> CalcResult:
    """중도상환수수료(중도상환해약금).

        수수료 = 중도상환금액 × 요율 × (대출잔여일수 ÷ 대출기간)

    면제 판정을 계산보다 먼저 한다. "만기까지 3개월 미만이면 면제"인데 산식부터
    돌리면 0원이어야 할 답에 숫자가 나온다. 면제 기준은 은행마다 다르다
    (하나 3개월, iM 1개월, 삼성화재 3개월).
    """
    if waiver_months is not None and months_to_maturity is not None and months_to_maturity < waiver_months:
        return CalcResult(0, "원", [f"만기까지 {months_to_maturity}개월 < 면제 기준 {waiver_months}개월"],
                          {"waived": True})
    if fee_rate_pct is None:
        return CalcResult(None, "원",
                          abstain_reason="중도상환수수료율이 문서에 비어 있습니다('( )%'). "
                                         "개별 계약서 또는 은행 확인이 필요합니다.")
    base = max(amount - exempt_amount, 0)
    if remaining_ratio is None:
        if days_remaining is None or not days_total:
            return CalcResult(None, "원", abstain_reason="잔여일수 또는 대출기간을 찾지 못했습니다.")
        if leap_year:
            days_remaining, days_total = days_remaining + 1, days_total + 1
        remaining_ratio = days_remaining / days_total
        ratio_label = f"{days_remaining}/{days_total}"
    else:
        ratio_label = f"{remaining_ratio:.4f}"

    raw = base * fee_rate_pct / 100 * remaining_ratio
    value = apply_rounding(raw, rounding)
    steps = []
    if exempt_amount:
        steps.append(f"면제금액 {exempt_amount:,}원 제외 → 부과대상 {base:,}원")
    steps.append(f"{base:,} × {fee_rate_pct}% × {ratio_label} = {raw:,.2f}")
    steps.append(f"{rounding} → {int(value):,}원")
    return CalcResult(int(value), "원", steps,
                      {"leap_year": leap_year, "rounding": rounding,
                       "exempt_amount": exempt_amount})


def delinquency_interest(
    *, principal: int, contract_rate_pct: float, penalty_add_pct: float,
    overdue_interest_amount: int = 0, months_overdue: int = 1,
    cap_rate_pct: float | None = None, rounding: Rounding = "truncate_won",
) -> CalcResult:
    """연체이자. 연체이자율 = 약정이자율 + 연체가산이자율.

    iM뱅크 설명서 예시 방식을 따른다(월 단위 단순 계산).
      1개월차: 지체된 약정이자 × 연체이자율 × 1/12
      2개월차: 원금 × 연체이자율 × 1/12
    """
    rate = contract_rate_pct + penalty_add_pct
    warnings: list[str] = []
    if cap_rate_pct is not None and rate > cap_rate_pct:
        warnings.append(f"연체이자율 {rate}% 가 상한 {cap_rate_pct}% 를 넘어 상한을 적용했습니다.")
        rate = cap_rate_pct
    steps = [f"연체이자율 = 약정 {contract_rate_pct}% + 가산 {penalty_add_pct}% = {rate}%"]
    total = 0
    if overdue_interest_amount:
        v = apply_rounding(overdue_interest_amount * rate / 100 / 12, rounding)
        total += int(v)
        steps.append(f"지체 약정이자 {overdue_interest_amount:,} × {rate}% × 1/12 = {int(v):,}원")
    for _ in range(max(months_overdue - 1, 0)):
        v = apply_rounding(principal * rate / 100 / 12, rounding)
        total += int(v)
        steps.append(f"원금 {principal:,} × {rate}% × 1/12 = {int(v):,}원")
    steps.append(f"합계 {total:,}원")
    return CalcResult(total, "원", steps, {"rate_pct": rate, "rounding": rounding}, warnings)


def delinquency_rate(*, normal_rate_pct: float, add_pct: float | None = None,
                     legal_cap_pct: float | None = None) -> CalcResult:
    """연체이자율만 구한다. 법정최고금리 캡이 걸리면 그 사실을 답변에 남긴다."""
    add = add_pct if add_pct is not None else C.get("delinquency.add_on_rate", 0.03) * 100
    cap = legal_cap_pct if legal_cap_pct is not None else C.get("interest_caps.legal_max_rate", 0.20) * 100
    raw = normal_rate_pct + add
    steps = [f"{normal_rate_pct}% + {add}%p = {raw}%"]
    warnings = []
    if raw > cap:
        steps.append(f"법정최고금리 {cap}% 캡 적용")
        warnings.append(f"산출값 {raw}% 가 법정최고 {cap}% 를 넘어 {cap}% 로 제한했습니다.")
        warnings.append(C.disclaimer())
        raw = cap
    return CalcResult(raw, "%", steps, {"legal_cap_pct": cap}, warnings)


# ── 카드·보험 (골든셋 채점용 참조 구현) ───────────────────────────────
def card_overseas_charge(*, usd: float, tt_rate: float, brand_fee_pct: float,
                         service_fee_pct: float, rounding: Rounding = "truncate_won") -> CalcResult:
    """해외이용 원화청구금액 = ①미화×전신환매도율 + ②브랜드수수료분 + ③해외서비스수수료분."""
    a = usd * tt_rate
    b = usd * brand_fee_pct / 100 * tt_rate
    c = usd * service_fee_pct / 100 * tt_rate
    total = apply_rounding(a + b + c, rounding)
    return CalcResult(int(total), "원",
                      [f"① {usd:,.0f} × {tt_rate:,.0f} = {a:,.0f}",
                       f"② 브랜드 {brand_fee_pct}% → {b:,.0f}",
                       f"③ 해외서비스 {service_fee_pct}% → {c:,.0f}",
                       f"합계 {int(total):,}원"], {"rounding": rounding})


def silson_outpatient_payout(*, copay: int, fixed_deductible: int,
                             coinsurance_pct: float = 20.0,
                             health_copay_rate: float | None = None) -> CalcResult:
    """실손 급여 통원 보험금 = 본인부담금 − 공제금액.

    공제금액은 '중 큰 금액'이다. 항목 수가 세대에 따라 다르다.
      4세대: max(정액, 보장대상의료비 × 20%)
      5세대: max(정액, 보장대상의료비 × 20%, 보장대상의료비 × 건강보험 본인부담률)
    health_copay_rate 를 주면 5세대, 안 주면 4세대로 계산한다. 답이 달라지므로
    질문에 세대·시행일이 없으면 되물어야 한다.
    """
    parts = [(f"정액 {fixed_deductible:,}", float(fixed_deductible)),
             (f"의료비의 {coinsurance_pct}%", copay * coinsurance_pct / 100)]
    gen = "4세대"
    if health_copay_rate is not None:
        parts.append((f"본인부담률 {health_copay_rate:.0%}", copay * health_copay_rate))
        gen = "5세대"
    label, deductible = max(parts, key=lambda x: x[1])
    payout = max(trunc_won(copay - deductible), 0)
    return CalcResult(payout, "원",
                      [f"공제 후보: " + ", ".join(f"{n}={v:,.0f}" for n, v in parts),
                       f"중 큰 금액 = {label} → {deductible:,.0f}원 공제",
                       f"{copay:,} − {deductible:,.0f} = {payout:,}원"],
                      {"generation": gen},
                      [f"{gen} 기준입니다. 세대가 다르면 공제 항목이 달라 답이 바뀝니다."])


def silson_noncovered_payout(*, amount: int, kind: Literal["outpatient", "inpatient"],
                             fixed_deductible: int = 0, coinsurance_pct: float = 0.0,
                             payout_pct: float = 0.0) -> CalcResult:
    """실손 비급여. 통원은 공제 후 지급, 입원은 비율 지급."""
    if kind == "outpatient":
        deductible = max(float(fixed_deductible), amount * coinsurance_pct / 100)
        payout = max(trunc_won(amount - deductible), 0)
        steps = [f"공제 = max({fixed_deductible:,}, {amount:,}×{coinsurance_pct}%) = {deductible:,.0f}",
                 f"{amount:,} − {deductible:,.0f} = {payout:,}원"]
    else:
        payout = trunc_won(amount * payout_pct / 100)
        steps = [f"{amount:,} × {payout_pct}% = {payout:,}원"]
    return CalcResult(payout, "원", steps, {"kind": kind},
                      ["회당·연간 한도(통원 100회 등)는 별도로 확인해야 합니다."])


def cancer_benefit(*, sum_insured: int, months_since_start: float,
                   days_since_start: int | None = None, waiting_days: int = 90,
                   reduction_pct: float = 50.0, reduction_within_months: int = 12) -> CalcResult:
    """암 진단급여금. 보장개시 판정 → 감액 판정 순서로 본다."""
    if days_since_start is not None and days_since_start <= waiting_days:
        return CalcResult(0, "원",
                          [f"진단 시점 {days_since_start}일 ≤ 보장개시 {waiting_days}일 → 보장개시 전"],
                          {"before_waiting": True},
                          ["보장개시일 전 진단은 암진단급여금 지급 대상이 아닙니다. "
                           "계약 소멸·보험료 반환 여부는 별도 조항 확인이 필요합니다."])
    if months_since_start < reduction_within_months:
        v = trunc_won(sum_insured * reduction_pct / 100)
        return CalcResult(v, "원",
                          [f"가입 후 {months_since_start}개월 < {reduction_within_months}개월 → "
                           f"{reduction_pct}% 감액", f"{sum_insured:,} × {reduction_pct}% = {v:,}원"],
                          {"reduced": True})
    return CalcResult(sum_insured, "원", [f"감액 기간 경과 → 전액 {sum_insured:,}원"], {"reduced": False})
