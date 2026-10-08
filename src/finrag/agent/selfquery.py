# -*- coding: utf-8 -*-
"""Self-Query: 질문에서 메타데이터 필터를 뽑는다.

LLM 이 하는 일은 Pydantic 스키마를 채우는 것뿐이다. Qdrant Filter 로 옮기는 일과
0건일 때 완화하는 일은 코드가 한다. 모델에게 필터 문법을 시키면 틀린 필드명을
지어내고, 그 순간 검색이 조용히 0건이 된다.
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from ..llm import get_llm, is_fake
from ..retrieval.filters import QueryFilters

SYSTEM = """너는 한국 금융문서 검색의 질의 분석기다.
사용자 질문에서 확실히 알 수 있는 조건만 뽑아라. 모르면 비워 둔다(추측 금지).

doc_type 은 다음 중 하나만: 약관, 표준약관, 특약, 부속약관, 상품설명서, 개정대비표,
법령, 분쟁조정사례, 공시, 이용안내, 사업비안내, 경영공시, FAQ
issuer 예: 하나은행, 카카오뱅크, iM뱅크, 신한은행, NH농협은행, KB국민카드, 삼성화재, ABL생명, 금융감독원, 법제처
product 예: 정기예금, 적금, 주택담보대출, 가계대출, 실손의료보험, 암보험, 신용카드, 여신거래, 예금거래
generation: 실손보험이면 4세대 또는 5세대
effective_on: 질문이 특정 시점을 지목하면 YYYY-MM-DD, "현재/현행"이면 latest, 없으면 비움

주의: '2013년 판 기준' 처럼 과거 시점을 지목하면 effective_on 에 그 날짜를 넣는다.
'제31조' 같은 조항번호는 필터가 아니라 검색어다. 넣지 마라."""

PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", "{question}")])


def extract_filters(question: str) -> QueryFilters:
    """질문에서 메타데이터 필터(발행사·문서 종류·상품·세대·기준일)를 뽑는다.

    LLM 에게 QueryFilters 스키마를 채우게 한다. 모르는 칸은 비워 둔다(추측 금지).
    Qdrant Filter 로 바꾸는 일(to_qdrant)과 0건일 때 푸는 일(relax)은 filters.py 가 한다.
    """
    llm = get_llm("selfquery", size="small")
    if is_fake(llm):
        return QueryFilters()                          # 필터 없이 검색이 조용히 0건보다 낫다
    try:
        out = (PROMPT | llm.with_structured_output(QueryFilters, method="json_schema")).invoke(
            {"question": question})
        return out if isinstance(out, QueryFilters) else QueryFilters(**(out or {}))
    except Exception:
        return QueryFilters()
