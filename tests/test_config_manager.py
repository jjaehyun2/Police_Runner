"""ConfigManager 및 SimulationConfig 단위 테스트."""

import os
import tempfile

import pytest
import yaml

from pursuit_evasion_rl.utils.config_manager import ConfigManager, SimulationConfig


@pytest.fixture
def default_yaml_path():
    """기본 설정 파일 경로를 반환한다."""
    return os.path.join(
        os.path.dirname(__file__),
        "..",
        "pursuit_evasion_rl",
        "configs",
        "default.yaml",
    )


@pytest.fixture
def valid_config_file(tmp_path):
    """유효한 YAML 설정 파일을 생성한다."""
    config = {
        "network": {"mode": "random", "num_nodes": 30, "density": 0.4, "fixed_edges": None},
        "agents": {"num_police": 5},
        "environment": {"max_steps": 300},
        "training": {
            "algorithm": "MADDPG",
            "learning_rate": 0.001,
            "discount_factor": 0.95,
            "epsilon": 0.2,
            "batch_size": 128,
            "replay_buffer_size": 50000,
            "max_episodes": 5000,
            "checkpoint_interval": 50,
        },
        "rewards": {
            "terminal_reward": 2.0,
            "shaping_scale": 0.2,
            "cooperation_bonus": 0.1,
            "invalid_action_penalty": -0.05,
        },
    }
    filepath = tmp_path / "config.yaml"
    filepath.write_text(yaml.dump(config), encoding="utf-8")
    return str(filepath)


class TestSimulationConfig:
    """SimulationConfig dataclass 테스트."""

    def test_default_values(self):
        """기본값이 올바르게 설정되는지 확인한다."""
        config = SimulationConfig()
        assert config.network_mode == "random"
        assert config.num_nodes == 20
        assert config.density == 0.3
        assert config.fixed_edges is None
        assert config.num_police == 3
        assert config.max_steps == 500
        assert config.algorithm == "MAPPO"
        assert config.learning_rate == 3e-4
        assert config.discount_factor == 0.99
        assert config.epsilon == 0.1
        assert config.batch_size == 64
        assert config.replay_buffer_size == 100000
        assert config.max_episodes == 10000
        assert config.checkpoint_interval == 100
        assert config.terminal_reward == 1.0
        assert config.shaping_scale == 0.1
        assert config.cooperation_bonus == 0.05
        assert config.invalid_action_penalty == -0.01


class TestConfigManagerLoad:
    """ConfigManager.load() 테스트."""

    def test_load_valid_config(self, valid_config_file):
        """유효한 설정 파일 로드."""
        manager = ConfigManager(valid_config_file)
        config = manager.load()
        assert config.network_mode == "random"
        assert config.num_nodes == 30
        assert config.density == 0.4
        assert config.num_police == 5
        assert config.max_steps == 300
        assert config.algorithm == "MADDPG"
        assert config.learning_rate == 0.001

    def test_load_default_yaml(self, default_yaml_path):
        """프로젝트 기본 YAML 파일 로드."""
        manager = ConfigManager(default_yaml_path)
        config = manager.load()
        assert config.network_mode == "random"
        assert config.num_nodes == 20
        assert config.density == 0.3

    def test_missing_fields_use_defaults(self, tmp_path):
        """누락된 항목에 기본값이 적용되는지 확인한다."""
        # 일부만 포함하는 설정 파일
        config = {"network": {"mode": "random", "num_nodes": 10, "density": 0.5}}
        filepath = tmp_path / "partial.yaml"
        filepath.write_text(yaml.dump(config), encoding="utf-8")

        manager = ConfigManager(str(filepath))
        result = manager.load()
        # 명시된 필드
        assert result.num_nodes == 10
        assert result.density == 0.5
        # 누락된 필드는 기본값
        assert result.num_police == 3
        assert result.max_steps == 500
        assert result.algorithm == "MAPPO"

    def test_yaml_syntax_error_exits(self, tmp_path):
        """YAML 문법 오류 시 sys.exit(1)이 호출되는지 확인한다."""
        filepath = tmp_path / "bad.yaml"
        filepath.write_text("network:\n  mode: [invalid\n  unclosed", encoding="utf-8")

        manager = ConfigManager(str(filepath))
        with pytest.raises(SystemExit) as exc_info:
            manager.load()
        assert exc_info.value.code == 1

    def test_file_not_found_exits(self, tmp_path):
        """파일 미존재 시 sys.exit(1)이 호출되는지 확인한다."""
        manager = ConfigManager(str(tmp_path / "nonexistent.yaml"))
        with pytest.raises(SystemExit) as exc_info:
            manager.load()
        assert exc_info.value.code == 1

    def test_out_of_range_exits(self, tmp_path):
        """범위 초과 시 sys.exit(1)이 호출되는지 확인한다."""
        config = {
            "network": {"mode": "random", "num_nodes": 300, "density": 0.5},
            "agents": {"num_police": 3},
            "environment": {"max_steps": 500},
            "training": {
                "algorithm": "MAPPO",
                "learning_rate": 0.0003,
                "discount_factor": 0.99,
                "epsilon": 0.1,
                "batch_size": 64,
                "replay_buffer_size": 100000,
                "max_episodes": 10000,
                "checkpoint_interval": 100,
            },
            "rewards": {
                "terminal_reward": 1.0,
                "shaping_scale": 0.1,
                "cooperation_bonus": 0.05,
                "invalid_action_penalty": -0.01,
            },
        }
        filepath = tmp_path / "bad_range.yaml"
        filepath.write_text(yaml.dump(config), encoding="utf-8")

        manager = ConfigManager(str(filepath))
        with pytest.raises(SystemExit) as exc_info:
            manager.load()
        assert exc_info.value.code == 1

    def test_fixed_edges_loaded(self, tmp_path):
        """fixed_edges가 튜플 리스트로 변환되는지 확인한다."""
        config = {
            "network": {
                "mode": "fixed",
                "num_nodes": 5,
                "density": 0.5,
                "fixed_edges": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]],
            },
            "agents": {"num_police": 3},
            "environment": {"max_steps": 100},
            "training": {
                "algorithm": "MAPPO",
                "learning_rate": 0.0003,
                "discount_factor": 0.99,
                "epsilon": 0.1,
                "batch_size": 64,
                "replay_buffer_size": 100000,
                "max_episodes": 10000,
                "checkpoint_interval": 100,
            },
            "rewards": {
                "terminal_reward": 1.0,
                "shaping_scale": 0.1,
                "cooperation_bonus": 0.05,
                "invalid_action_penalty": -0.01,
            },
        }
        filepath = tmp_path / "fixed.yaml"
        filepath.write_text(yaml.dump(config), encoding="utf-8")

        manager = ConfigManager(str(filepath))
        result = manager.load()
        assert result.fixed_edges == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]


class TestConfigManagerValidate:
    """ConfigManager.validate() 테스트."""

    def test_valid_config_no_errors(self):
        """유효한 설정은 오류 없음."""
        config = SimulationConfig()
        manager = ConfigManager("dummy")
        errors = manager.validate(config)
        assert errors == []

    def test_invalid_network_mode(self):
        """잘못된 network_mode 검증."""
        config = SimulationConfig(network_mode="invalid")
        manager = ConfigManager("dummy")
        errors = manager.validate(config)
        assert len(errors) == 1
        assert "network_mode" in errors[0]

    def test_num_nodes_too_small(self):
        """num_nodes < 4 검증."""
        config = SimulationConfig(num_nodes=2)
        manager = ConfigManager("dummy")
        errors = manager.validate(config)
        assert any("num_nodes" in e for e in errors)

    def test_num_nodes_too_large(self):
        """num_nodes > 200 검증."""
        config = SimulationConfig(num_nodes=201)
        manager = ConfigManager("dummy")
        errors = manager.validate(config)
        assert any("num_nodes" in e for e in errors)

    def test_density_zero(self):
        """density == 0 검증."""
        config = SimulationConfig(density=0.0)
        manager = ConfigManager("dummy")
        errors = manager.validate(config)
        assert any("density" in e for e in errors)

    def test_density_above_one(self):
        """density > 1.0 검증."""
        config = SimulationConfig(density=1.5)
        manager = ConfigManager("dummy")
        errors = manager.validate(config)
        assert any("density" in e for e in errors)

    def test_negative_learning_rate(self):
        """learning_rate <= 0 검증."""
        config = SimulationConfig(learning_rate=-0.001)
        manager = ConfigManager("dummy")
        errors = manager.validate(config)
        assert any("learning_rate" in e for e in errors)

    def test_discount_factor_out_of_range(self):
        """discount_factor > 1 검증."""
        config = SimulationConfig(discount_factor=1.5)
        manager = ConfigManager("dummy")
        errors = manager.validate(config)
        assert any("discount_factor" in e for e in errors)

    def test_negative_epsilon(self):
        """epsilon < 0 검증."""
        config = SimulationConfig(epsilon=-0.1)
        manager = ConfigManager("dummy")
        errors = manager.validate(config)
        assert any("epsilon" in e for e in errors)

    def test_batch_size_zero(self):
        """batch_size < 1 검증."""
        config = SimulationConfig(batch_size=0)
        manager = ConfigManager("dummy")
        errors = manager.validate(config)
        assert any("batch_size" in e for e in errors)

    def test_num_police_too_few(self):
        """num_police < 3 검증."""
        config = SimulationConfig(num_police=1)
        manager = ConfigManager("dummy")
        errors = manager.validate(config)
        assert any("num_police" in e for e in errors)

    def test_num_police_too_many(self):
        """num_police > 20 검증."""
        config = SimulationConfig(num_police=25)
        manager = ConfigManager("dummy")
        errors = manager.validate(config)
        assert any("num_police" in e for e in errors)


class TestConfigManagerSerialize:
    """ConfigManager.serialize() 테스트."""

    def test_serialize_produces_valid_yaml(self):
        """직렬화 결과가 유효한 YAML인지 확인한다."""
        config = SimulationConfig()
        manager = ConfigManager("dummy")
        yaml_str = manager.serialize(config)
        parsed = yaml.safe_load(yaml_str)
        assert "network" in parsed
        assert "agents" in parsed
        assert "environment" in parsed
        assert "training" in parsed
        assert "rewards" in parsed

    def test_serialize_preserves_values(self):
        """직렬화가 값을 보존하는지 확인한다."""
        config = SimulationConfig(num_nodes=50, density=0.7, num_police=10)
        manager = ConfigManager("dummy")
        yaml_str = manager.serialize(config)
        parsed = yaml.safe_load(yaml_str)
        assert parsed["network"]["num_nodes"] == 50
        assert parsed["network"]["density"] == 0.7
        assert parsed["agents"]["num_police"] == 10


class TestConfigManagerRoundTrip:
    """ConfigManager.round_trip() 테스트."""

    def test_round_trip_default_config(self):
        """기본 설정의 round-trip 동일성 확인."""
        config = SimulationConfig()
        manager = ConfigManager("dummy")
        result = manager.round_trip(config)
        assert result == config

    def test_round_trip_custom_config(self):
        """커스텀 설정의 round-trip 동일성 확인."""
        config = SimulationConfig(
            network_mode="fixed",
            num_nodes=10,
            density=0.8,
            fixed_edges=[(0, 1), (1, 2), (2, 3)],
            num_police=5,
            max_steps=200,
            algorithm="MADDPG",
            learning_rate=0.01,
            discount_factor=0.9,
            epsilon=0.5,
            batch_size=32,
            replay_buffer_size=10000,
            max_episodes=1000,
            checkpoint_interval=50,
            terminal_reward=2.0,
            shaping_scale=0.05,
            cooperation_bonus=0.1,
            invalid_action_penalty=-0.02,
        )
        manager = ConfigManager("dummy")
        result = manager.round_trip(config)
        assert result == config
