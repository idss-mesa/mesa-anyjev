from __future__ import annotations

from mesa_anyjev.cards import DatasetCard, is_identifier, parse_card


def test_parse_card_header_sites_rows_columns(card: DatasetCard) -> None:
    assert card.product_code == "DP1.10003.001"
    assert card.table == "brd_countdata"
    assert card.name == "DP1.10003.001.brd_countdata"
    assert card.product_title == "Breeding landbird point counts"
    assert [s.code for s in card.sites] == ["HARV", "SRER"]
    assert card.sites[1].domain == "D14"
    assert card.sites[1].habitat == "semi-arid desert grassland/shrubland"
    assert card.rows == 15484
    assert (card.months_from, card.months_to) == ("2020-04", "2024-06")
    assert len(card.columns) == 7
    assert card.column("observerDistance").unit == "meter"
    assert card.column("siteID").unit == ""
    assert len(card.sha256) == 64


def test_identifier_rule(card: DatasetCard) -> None:
    flagged = {c.name for c in card.columns if is_identifier(c)}
    assert flagged == {"uid", "siteID", "startDate", "identificationHistoryID"}
    assert not is_identifier(card.column("scientificName"))


def test_not_a_card_raises() -> None:
    try:
        parse_card("# Something else\n")
    except ValueError as exc:
        assert "Dataset" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
