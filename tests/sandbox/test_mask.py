import pytest

from carepilot.sandbox.mask import mask_address, mask_phone, mask_tracking_no


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("13812345678", "138****5678"),
        ("138 1234 5678", "138****5678"),
        ("138-1234-5678", "138****5678"),
        ("+86 13812345678", "138****5678"),
        ("+8613812345678", "138****5678"),
        ("8613812345678", "138****5678"),
        ("138****5678", "138****5678"),
        ("", "****"),
        (None, "****"),
        ("12345", "****"),
        ("138123456789", "****"),
        ("abcdefghijk", "****"),
        ("1381234567a", "****"),
    ],
)
def test_mask_phone_boundaries(raw, expected):
    assert mask_phone(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("浙江省杭州市西湖区文三路1号", "浙江省杭州市***"),
        ("北京市朝阳区建国路8号", "北京市***"),
        ("上海市浦东新区张江路2号", "上海市***"),
        ("广东省深圳市南山区科技园", "广东省深圳市***"),
        ("浙江省杭州市西湖区***", "浙江省杭州市西湖区***"),
        ("没有行政标记的地址", "***"),
        ("", "***"),
        (None, "***"),
    ],
)
def test_mask_address_boundaries(raw, expected):
    assert mask_address(raw) == expected


def test_mask_tracking_no_hides_middle():
    masked = mask_tracking_no("SF123456789012")
    assert masked == "SF12****9012"
    assert mask_tracking_no("") == "****"
    assert mask_tracking_no(None) == "****"
