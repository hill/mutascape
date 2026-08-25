import numpy as np

from mutascape.benchmarks.topology import CubeSpec, navigability_summary


def test_navigability_summary_follows_steepest_route_from_wild_type():
    cube = CubeSpec(
        positions=(1, 2),
        alphabets={1: ["A", "B"], 2: ["A", "B"]},
        wild_type={1: "A", 2: "A"},
    )
    tensor = np.asarray([[0.0, 1.0], [0.5, 2.0]])

    result = navigability_summary(tensor, cube)

    assert result["beneficial_fraction"] == 0.75
    assert result["greedy_peak_percentile"] == 1.0
    assert result["greedy_normalised_regret"] == 0.0
    assert result["greedy_path_length"] == 2.0
