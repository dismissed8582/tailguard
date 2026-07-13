from pathlib import Path

import pandas as pd
import pytest

import generate_bom


def test_random_generator_validates_shape_and_keeps_supplier_names_unique():
    with pytest.raises(ValueError):
        generate_bom.generate_random_bom(0, 3)
    with pytest.raises(ValueError):
        generate_bom.generate_random_bom(1, 1)
    with pytest.raises(ValueError):
        generate_bom.generate_random_bom(1, 2, seed=-1)
    with pytest.raises(ValueError):
        generate_bom.generate_random_bom(True, 2)
    frame = generate_bom.generate_random_bom(6, 3, seed=11)
    assert frame["supplier"].is_unique
    assert set(frame.groupby("component").size()) == {3}


def test_custom_cli_is_deterministic_includes_seed_and_refuses_overwrite(tmp_path):
    args = ["--custom", "2", "3", "--seed", "9", "--out", str(tmp_path)]
    generate_bom.main(args)
    path = tmp_path / "bom_random_2_comp_3_opt_seed_9.csv"
    first = pd.read_csv(path, comment="#")
    assert path.exists()
    assert path.read_text(encoding="utf-8").splitlines()[0].startswith("# Synthetic Tailguard")
    with pytest.raises(SystemExit):
        generate_bom.main(args)
    generate_bom.main(args + ["--force"])
    assert first.equals(pd.read_csv(path, comment="#"))


def test_custom_cli_hashes_oversized_seed_in_filename(tmp_path):
    seed = 10**400
    generate_bom.main(["--custom", "1", "2", "--seed", str(seed), "--out", str(tmp_path)])

    paths = list(tmp_path.glob("bom_random_1_comp_2_opt_seed_sha256_*.csv"))
    assert len(paths) == 1
    assert len(paths[0].name.encode()) < 255
    assert len(pd.read_csv(paths[0], comment="#")) == 2


def test_cli_omits_output_path_from_filesystem_errors(tmp_path, monkeypatch, capsys):
    def fail_preflight(paths, force=False):
        raise OSError("cannot access {}".format(tmp_path))

    monkeypatch.setattr(generate_bom, "_preflight_output_paths", fail_preflight)
    with pytest.raises(SystemExit):
        generate_bom.main(["--industry", "auto", "--out", str(tmp_path)])

    captured = capsys.readouterr()
    assert str(tmp_path) not in captured.err
    assert "Traceback" not in captured.err
    assert "check --out" in captured.err


def test_cli_translates_an_invalid_output_path_without_a_traceback(capsys):
    with pytest.raises(SystemExit):
        generate_bom.main(["--custom", "1", "2", "--out", "bad\0path"])
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert "check --out" in captured.err


@pytest.mark.parametrize("writer", ["csv", "template"])
def test_non_force_publication_cannot_clobber_a_racing_destination(
    writer, tmp_path, monkeypatch
):
    destination = tmp_path / f"{writer}.csv"
    real_link = generate_bom.os.link

    def racing_link(source, target):
        Path(target).write_text("created by another process\n", encoding="utf-8")
        return real_link(source, target)

    monkeypatch.setattr(generate_bom.os, "link", racing_link)
    with pytest.raises(FileExistsError):
        if writer == "csv":
            generate_bom._write_csv(
                pd.DataFrame({"component": ["A"], "supplier": ["S"]}),
                destination,
            )
        else:
            generate_bom._write_template(destination)

    assert destination.read_text(encoding="utf-8") == "created by another process\n"
    assert not list(tmp_path.glob(f".{destination.name}.*.tmp"))


@pytest.mark.parametrize("values", [("0", "3"), ("1", "1")])
def test_custom_cli_rejects_invalid_counts(values, tmp_path):
    with pytest.raises(SystemExit):
        generate_bom.main(["--custom", *values, "--out", str(tmp_path)])


def test_industry_and_custom_are_mutually_exclusive(tmp_path):
    with pytest.raises(SystemExit):
        generate_bom.main(["--industry", "auto", "--custom", "2", "3", "--out", str(tmp_path)])


def test_checked_in_industry_fixtures_match_generator_templates():
    for key in generate_bom.INDUSTRIES:
        expected = generate_bom.industry_to_df(key)
        actual = pd.read_csv(f"data/bom/bom_{key}.csv", comment="#")
        pd.testing.assert_frame_equal(actual, expected)


def test_industry_generation_preflights_every_output_before_writing(tmp_path):
    existing_template = tmp_path / "bom_template.csv"
    existing_template.write_text("already here\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        generate_bom.main(["--industry", "auto", "--out", str(tmp_path)])
    assert not (tmp_path / "bom_auto.csv").exists()
