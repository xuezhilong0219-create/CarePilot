import pytest

from carepilot.graph.nodes import resolve_citations

IDS = ["企业版/采购#10", "企业版/采购入库单#1", "WMS/基础设置#0", "平台对接_店铺授权/淘宝#3"]


@pytest.mark.parametrize(
    "cited, expected",
    [
        (["1"], [IDS[0]]),
        (["[2]"], [IDS[1]]),
        (["资料3"], [IDS[2]]),
        (["资料#4"], [IDS[3]]),
        (["1", "3"], [IDS[0], IDS[2]]),
        ([IDS[1]], [IDS[1]]),  # 完整 id 仍然接受
        (["2", IDS[1], "[2]"], [IDS[1]]),  # 同一条的几种写法去重
        (["#0"], []),  # 实测里模型返回过的缩写：不猜
        (["0"], []),
        (["10"], []),  # 越界（只有 4 条资料）
        (["5"], []),
        (["企业版/不存在#9"], []),
        ([], []),
        (["", "  "], []),
    ],
)
def test_resolve_citations(cited, expected):
    assert resolve_citations(cited, IDS) == expected


def test_resolve_citations_never_maps_a_hallucinated_number_beyond_n():
    assert resolve_citations(["7"], IDS[:2]) == []
