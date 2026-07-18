from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tailguard.bom import (
    BOMValidationError,
    SupplierOption,
    bom_from_dataframe,
    bom_to_dataframe,
    count_by_source_type,
    flatten_bom,
    load_bom_csv,
    load_bom_csv_bytes,
)


def test_checked_in_template_parses_with_documentation_comments():
    bom = load_bom_csv(Path("data/bom/bom_template.csv"))
    options = flatten_bom(bom)
    assert len(options) == 2
    assert {option.source_type for option in options} == {"offshore", "domestic"}


def test_all_checked_in_bom_fixtures_parse():
    for path in Path("data/bom").glob("*.csv"):
        assert flatten_bom(load_bom_csv(path)), path


def test_local_bom_loader_translates_invalid_filesystem_paths():
    with pytest.raises(BOMValidationError, match="could not read"):
        load_bom_csv("bad\0path")


def test_type_and_lead_time_are_preserved_independently():
    frame = pd.DataFrame(
        [
            ["A", "one_day_offshore", 10.25, 2.5, 1, "offshore"],
            ["A", "long_domestic", 12.75, 1.25, 9, "domestic"],
        ],
        columns=["component", "supplier", "base_cost", "kappa", "lead_time", "type"],
    )
    options = flatten_bom(bom_from_dataframe(frame))
    assert options[0].source_type == "offshore"
    assert options[0].lead_time == 1
    assert options[1].source_type == "domestic"
    assert options[1].lead_time == 9
    assert options[0].base_cost == 10.25


def test_direct_supplier_option_rejects_boolean_values_and_invalid_lead_cost():
    with pytest.raises(BOMValidationError):
        SupplierOption("A", "S", True, 0, 1, "offshore")
    option = SupplierOption("A", "S", 1, 0, 1, "offshore")
    with pytest.raises(BOMValidationError, match="lead_time_cost"):
        option.deterministic_cost(-1)
    with pytest.raises(BOMValidationError, match="base_cost"):
        SupplierOption("A", "S", np.asarray([1.0]), 0, 1, "offshore")
    with pytest.raises(BOMValidationError, match="lead_time_cost"):
        option.deterministic_cost(np.asarray([1.0]))
    with pytest.raises(BOMValidationError, match="source_type"):
        SupplierOption("A", "S", 1, 0, 1, [])
    with pytest.raises(BOMValidationError, match="lead_time_cost"):
        option.deterministic_cost(np.bool_(True))


def test_deterministic_cost_rounds_the_exact_sum_once_and_rejects_underflow():
    minimum_positive = np.nextafter(0.0, 1.0)
    option = SupplierOption("A", "S", minimum_positive, 0, 0.5, "offshore")

    assert option.deterministic_cost(minimum_positive) == 2 * minimum_positive

    zero_base = SupplierOption("A", "S", 0, 0, 0.5, "offshore")
    with pytest.raises(BOMValidationError, match="too small"):
        zero_base.deterministic_cost(minimum_positive)


@pytest.mark.parametrize("bad_type", ["off-shore", "domsetic", "", "unknown"])
def test_unknown_source_type_is_rejected(bad_type):
    frame = pd.DataFrame(
        [["A", "S", 1, 0, 1, bad_type]],
        columns=["component", "supplier", "base_cost", "kappa", "lead_time", "type"],
    )
    with pytest.raises(BOMValidationError):
        bom_from_dataframe(frame)


def test_negative_numeric_input_is_rejected_without_mutating_frame():
    frame = pd.DataFrame(
        [["A", "S", -1, 0, 1, "offshore"]],
        columns=[" Component ", "supplier", "base_cost", "kappa", "lead_time", "type"],
    )
    original_columns = list(frame.columns)
    with pytest.raises(BOMValidationError):
        bom_from_dataframe(frame)
    assert list(frame.columns) == original_columns


@pytest.mark.parametrize("bad_text", [np.asarray([1, 2]), [], {}, 123])
def test_dataframe_identifiers_must_be_scalar_text(bad_text):
    frame = pd.DataFrame(
        [["A", "S", 1, 0, 1, "offshore"]],
        columns=["component", "supplier", "base_cost", "kappa", "lead_time", "type"],
    )
    frame.at[0, "component"] = bad_text
    with pytest.raises(BOMValidationError, match="must be text"):
        bom_from_dataframe(frame)


@pytest.mark.parametrize("field", ["component", "supplier"])
def test_nul_identifiers_are_rejected_by_direct_and_dataframe_apis(field):
    direct = {
        "component": "A",
        "supplier": "S",
        "base_cost": 1,
        "kappa": 0,
        "lead_time": 1,
        "source_type": "offshore",
    }
    direct[field] += "\x00hidden"
    with pytest.raises(BOMValidationError, match="NUL"):
        SupplierOption(**direct)

    row = {
        "component": "A",
        "supplier": "S",
        "base_cost": 1,
        "kappa": 0,
        "lead_time": 1,
        "type": "offshore",
    }
    row[field] += "\x00hidden"
    with pytest.raises(BOMValidationError, match="NUL"):
        bom_from_dataframe(pd.DataFrame([row]))


@pytest.mark.parametrize("field", ["component", "supplier"])
def test_lone_surrogate_identifiers_are_rejected_by_programmatic_apis(field):
    direct = {
        "component": "A",
        "supplier": "S",
        "base_cost": 1,
        "kappa": 0,
        "lead_time": 1,
        "source_type": "offshore",
    }
    direct[field] = "synthetic\ud800identifier"
    with pytest.raises(BOMValidationError, match="UTF-8"):
        SupplierOption(**direct)

    row = {
        "component": "A",
        "supplier": "S",
        "base_cost": 1,
        "kappa": 0,
        "lead_time": 1,
        "type": "offshore",
    }
    row[field] = "synthetic\ud800identifier"
    with pytest.raises(BOMValidationError, match="UTF-8"):
        bom_from_dataframe(pd.DataFrame([row]))


def test_valid_utf8_identifiers_round_trip_through_csv_serialization():
    component = "synthetic_组件"
    supplier = "synthetic,option\nα"
    option = SupplierOption(component, supplier, 1.25, 0.5, 2, "offshore")
    frame = bom_to_dataframe({component: (option,)})

    restored = load_bom_csv_bytes(frame.to_csv(index=False).encode("utf-8"))

    assert flatten_bom(restored)[0] == option


def test_dataframe_numeric_fields_reject_numpy_booleans():
    frame = pd.DataFrame(
        [["A", "S", np.bool_(True), 0, 1, "offshore"]],
        columns=["component", "supplier", "base_cost", "kappa", "lead_time", "type"],
        dtype=object,
    )
    with pytest.raises(BOMValidationError, match="must be numeric"):
        bom_from_dataframe(frame)


def test_dataframe_numeric_fields_reject_custom_float_only_objects():
    class FloatOnly:
        def __float__(self):
            return 0.0

    frame = pd.DataFrame(
        [["A", "S", FloatOnly(), 0, 1, "offshore"]],
        columns=["component", "supplier", "base_cost", "kappa", "lead_time", "type"],
    )
    with pytest.raises(BOMValidationError, match="must be numeric"):
        bom_from_dataframe(frame)


@pytest.mark.parametrize(
    "value",
    ["1.25", Decimal("1.25"), Fraction(5, 4), np.float64(1.25)],
)
def test_dataframe_numeric_fields_accept_supported_scalar_types(value):
    frame = pd.DataFrame(
        [["A", "S", value, 0, 1, "offshore"]],
        columns=["component", "supplier", "base_cost", "kappa", "lead_time", "type"],
    )
    assert bom_from_dataframe(frame)["A"][0].base_cost == 1.25


def test_component_need_not_have_both_reporting_labels():
    frame = pd.DataFrame(
        [["A", "S", 1, 0, 1, "offshore"]],
        columns=["component", "supplier", "base_cost", "kappa", "lead_time", "type"],
    )
    assert len(bom_from_dataframe(frame)["A"]) == 1


def test_source_type_counts_have_deterministic_key_order():
    offshore = SupplierOption("A", "O", 1, 0, 1, "offshore")
    domestic = SupplierOption("A", "D", 1, 0, 1, "domestic")

    assert list(count_by_source_type([domestic, offshore, offshore]).items()) == [
        ("offshore", 2),
        ("domestic", 1),
    ]


def test_dataframe_validation_handles_non_integer_index_and_duplicate_normalized_headers():
    frame = pd.DataFrame(
        [["A", "S", 1, 0, 1, "offshore"]],
        columns=["component", "supplier", "base_cost", "kappa", "lead_time", "type"],
        index=["row-a"],
    )
    assert len(bom_from_dataframe(frame)["A"]) == 1

    duplicate_header = frame.copy()
    duplicate_header.columns = ["component", " Component ", "base_cost", "kappa", "lead_time", "type"]
    with pytest.raises(BOMValidationError, match="duplicate columns"):
        bom_from_dataframe(duplicate_header)


def test_hand_built_bom_cannot_repeat_a_supplier_within_component():
    duplicate = SupplierOption("A", "S", 1, 0, 1, "offshore")
    with pytest.raises(BOMValidationError, match="duplicate supplier"):
        flatten_bom({"A": (duplicate, duplicate)})


def test_raw_csv_duplicate_headers_cannot_be_silently_mangled_by_pandas(tmp_path):
    path = tmp_path / "duplicate-header.csv"
    path.write_text(
        "component,component,supplier,base_cost,kappa,lead_time,type\n"
        "A,ignored,S,1,0,1,offshore\n",
        encoding="utf-8",
    )
    with pytest.raises(BOMValidationError, match="duplicate columns"):
        load_bom_csv(path)


def test_in_memory_bom_upload_parses_without_a_filename_or_local_file():
    content = (
        "# illustrative upload\n"
        " Component ,Supplier,base_cost,kappa,lead_time,type\n"
        "A,one,10.25,2.5,4,offshore\n"
        "A,two,12.75,1.25,2,domestic\n"
    ).encode("utf-8")
    bom = load_bom_csv_bytes(memoryview(content))
    options = flatten_bom(bom)
    assert [option.supplier for option in options] == ["one", "two"]
    assert options[0].base_cost == 10.25


def test_documentation_comments_do_not_truncate_hash_characters_inside_csv_fields():
    content = (
        "  # full-line documentation\n"
        "component,supplier,base_cost,kappa,lead_time,type\n"
        "A,option#1,1,0,1,offshore\n"
    ).encode("utf-8")
    option = flatten_bom(load_bom_csv_bytes(content))[0]
    assert option.supplier == "option#1"


def test_documentation_filter_preserves_hash_lines_inside_quoted_multiline_fields():
    content = (
        "# complete documentation record\n"
        "component,supplier,base_cost,kappa,lead_time,type\n"
        'A,"line one\n# comment-looking but valid field data\nline three",1,0,1,offshore\n'
    ).encode("utf-8")
    option = flatten_bom(load_bom_csv_bytes(content))[0]
    assert option.supplier == "line one\n# comment-looking but valid field data\nline three"


def test_only_leading_comments_are_filtered_and_identifier_text_is_preserved():
    content = (
        "# leading documentation\n"
        "component,supplier,base_cost,kappa,lead_time,type\n"
        "#A,0007,1,0,1,offshore\n"
        "001,NA,2,0,1,domestic\n"
        "NULL,S3,3,0,1,offshore\n"
    ).encode("utf-8")
    options = flatten_bom(load_bom_csv_bytes(content))
    assert [(option.component, option.supplier) for option in options] == [
        ("#A", "0007"),
        ("001", "NA"),
        ("NULL", "S3"),
    ]


def test_in_memory_bom_upload_rejects_invalid_encoding_and_non_bytes():
    with pytest.raises(BOMValidationError, match="UTF-8"):
        load_bom_csv_bytes(b"\xff\xfe\x00")
    with pytest.raises(TypeError, match="bytes-like"):
        load_bom_csv_bytes("component,supplier")
    released = memoryview(b"released")
    released.release()
    with pytest.raises(BOMValidationError, match="could not read"):
        load_bom_csv_bytes(released)
    with pytest.raises(BOMValidationError, match="NUL"):
        load_bom_csv_bytes(
            b"component,supplier,base_cost,kappa,lead_time,type\nA,S\x00hidden,1,0,1,offshore\n"
        )
    with pytest.raises(BOMValidationError, match="same number of fields"):
        load_bom_csv_bytes(
            b"component,supplier,base_cost,kappa,lead_time,type\nEXTRA,A,S,1,2,3,domestic\n"
        )


def test_hand_built_bom_rejects_an_empty_iterator_component():
    with pytest.raises(BOMValidationError, match="no supplier options"):
        flatten_bom({"A": iter(())})


def test_hand_built_bom_must_be_a_mapping():
    with pytest.raises(BOMValidationError, match="mapping"):
        flatten_bom([1])


def test_supplier_option_rejects_an_unrepresentable_python_integer_cleanly():
    with pytest.raises(BOMValidationError, match="finite non-negative"):
        SupplierOption("A", "S", 10**10_000, 0, 1, "offshore")


def test_bom_rejects_underflowed_costs_and_unordered_option_collections():
    with pytest.raises(BOMValidationError, match="finite non-negative"):
        SupplierOption("A", "S", Fraction(-1, 10**1000), 0, 1, "offshore")
    with pytest.raises(BOMValidationError, match="too small"):
        load_bom_csv_bytes(
            b"component,supplier,base_cost,kappa,lead_time,type\nA,S,1e-400,0,1,offshore\n"
        )
    for tiny_decimal in (Decimal("1e-10000"), Decimal("-1e-10000")):
        frame = pd.DataFrame(
            {
                "component": ["A"],
                "supplier": ["S"],
                "base_cost": [tiny_decimal],
                "kappa": [0],
                "lead_time": [1],
                "type": ["offshore"],
            }
        )
        with pytest.raises(BOMValidationError):
            bom_from_dataframe(frame)
    option = SupplierOption("A", "S", 1, 0, 1, "offshore")
    with pytest.raises(TypeError, match="ordered"):
        flatten_bom({"A": {option}})
    with pytest.raises(BOMValidationError, match="finite non-negative"):
        SupplierOption("A", "S", np.complex128(1 + 2j), 0, 1, "offshore")
    hidden_scalar = frame.copy()
    hidden_scalar.loc[0, "base_cost"] = np.array(True)
    with pytest.raises(BOMValidationError, match="numeric"):
        bom_from_dataframe(hidden_scalar)
