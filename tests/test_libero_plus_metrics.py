import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "LIBERO-plus"
    / "eval_files"
    / "libero_plus_metrics.py"
)


def load_metrics_module():
    spec = importlib.util.spec_from_file_location("libero_plus_metrics", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_recorder_tracks_overall_category_and_difficulty():
    metrics = load_metrics_module()
    mapping = [
        {"id": 1, "name": "task_a", "category": "Camera Viewpoints", "difficulty_level": 2},
        {"id": 2, "name": "task_b", "category": "Sensor Noise", "difficulty_level": 5},
    ]

    recorder = metrics.LiberoPlusMetricRecorder("libero_goal", mapping, task_range=(0, 2))
    recorder.record_episode(task_id=0, success=True)
    recorder.record_episode(task_id=1, success=False)
    result = recorder.to_dict()

    assert result["suite"] == "libero_goal"
    assert result["task_range"] == [0, 2]
    assert result["overall"] == {"total_count": 2, "success_count": 1, "success_rate": 0.5}
    assert result["by_category"]["Camera Viewpoints"] == {
        "total_count": 1,
        "success_count": 1,
        "success_rate": 1.0,
    }
    assert result["by_category"]["Sensor Noise"] == {
        "total_count": 1,
        "success_count": 0,
        "success_rate": 0.0,
    }
    assert result["by_difficulty_level"]["2"] == {
        "total_count": 1,
        "success_count": 1,
        "success_rate": 1.0,
    }
    assert result["by_difficulty_level"]["5"] == {
        "total_count": 1,
        "success_count": 0,
        "success_rate": 0.0,
    }


def test_merge_results_sums_nested_metric_groups():
    metrics = load_metrics_module()
    first = {
        "suite": "libero_goal",
        "task_range": [0, 1],
        "overall": {"total_count": 1, "success_count": 1, "success_rate": 1.0},
        "by_category": {
            "Camera Viewpoints": {"total_count": 1, "success_count": 1, "success_rate": 1.0},
        },
        "by_difficulty_level": {
            "2": {"total_count": 1, "success_count": 1, "success_rate": 1.0},
        },
    }
    second = {
        "suite": "libero_goal",
        "task_range": [1, 2],
        "overall": {"total_count": 1, "success_count": 0, "success_rate": 0.0},
        "by_category": {
            "Camera Viewpoints": {"total_count": 1, "success_count": 0, "success_rate": 0.0},
        },
        "by_difficulty_level": {
            "2": {"total_count": 1, "success_count": 0, "success_rate": 0.0},
        },
    }

    merged = metrics.merge_metric_results([first, second])

    assert merged["overall"] == {"total_count": 2, "success_count": 1, "success_rate": 0.5}
    assert merged["by_category"]["Camera Viewpoints"] == {
        "total_count": 2,
        "success_count": 1,
        "success_rate": 0.5,
    }
    assert merged["by_difficulty_level"]["2"] == {
        "total_count": 2,
        "success_count": 1,
        "success_rate": 0.5,
    }


def test_recorder_normalizes_missing_difficulty_level():
    metrics = load_metrics_module()
    mapping = [
        {"id": 1, "name": "task_a", "category": "Light Conditions", "difficulty_level": None},
    ]

    recorder = metrics.LiberoPlusMetricRecorder("libero_goal", mapping)
    recorder.record_episode(task_id=0, success=True)
    result = recorder.to_dict()

    assert result["by_difficulty_level"]["unknown"] == {
        "total_count": 1,
        "success_count": 1,
        "success_rate": 1.0,
    }


def test_task_meta_and_safe_filename_part():
    metrics = load_metrics_module()
    mapping = [
        {"id": 1, "name": "open drawer/table 1", "category": "Camera Viewpoints", "difficulty_level": 3},
    ]

    recorder = metrics.LiberoPlusMetricRecorder("libero_goal", mapping)

    assert recorder.task_meta(0)["category"] == "Camera Viewpoints"
    assert metrics.safe_filename_part("Camera Viewpoints") == "Camera_Viewpoints"
    assert metrics.safe_filename_part("open drawer/table 1") == "open_drawer_table_1"

def test_uniform_sample_ids_spreads_selection_across_sequence():
    metrics = load_metrics_module()

    assert metrics.uniform_sample_ids(list(range(10)), 5) == [0, 2, 4, 7, 9]


def test_select_task_ids_by_category_fraction_samples_each_category_evenly():
    metrics = load_metrics_module()
    mapping = [
        {"id": 1, "name": "a1", "category": "A", "difficulty_level": 1},
        {"id": 2, "name": "a2", "category": "A", "difficulty_level": 1},
        {"id": 3, "name": "a3", "category": "A", "difficulty_level": 1},
        {"id": 4, "name": "a4", "category": "A", "difficulty_level": 1},
        {"id": 5, "name": "a5", "category": "A", "difficulty_level": 1},
        {"id": 6, "name": "a6", "category": "A", "difficulty_level": 1},
        {"id": 7, "name": "b1", "category": "B", "difficulty_level": 1},
        {"id": 8, "name": "b2", "category": "B", "difficulty_level": 1},
        {"id": 9, "name": "b3", "category": "B", "difficulty_level": 1},
        {"id": 10, "name": "b4", "category": "B", "difficulty_level": 1},
    ]

    task_ids = metrics.select_task_ids_by_category_fraction(
        mapping,
        start_idx=0,
        end_idx=10,
        category_fraction=0.5,
    )

    assert task_ids == [0, 2, 5, 6, 9]
    assert metrics.count_task_ids_by_category(mapping, task_ids) == {"A": 3, "B": 2}

