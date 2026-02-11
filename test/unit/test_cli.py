import argparse
import pytest

from netqmpi.runtime.cli import NetQASMConfig


class TestNetQASMConfigDefaults:
    """Verify that NetQASMConfig has sensible default values."""

    def test_network_config_defaults_to_none(self):
        cfg = NetQASMConfig()
        assert cfg.network_config is None

    def test_post_function_defaults_to_none(self):
        cfg = NetQASMConfig()
        assert cfg.post_function is None

    def test_formalism_defaults_to_ket(self):
        from netqasm.runtime.settings import Formalism
        cfg = NetQASMConfig()
        assert cfg.formalism == Formalism.KET

    def test_num_rounds_defaults_to_one(self):
        cfg = NetQASMConfig()
        assert cfg.num_rounds == 1

    def test_log_cfg_defaults_to_none(self):
        cfg = NetQASMConfig()
        assert cfg.log_cfg is None

    def test_use_app_config_defaults_to_true(self):
        cfg = NetQASMConfig()
        assert cfg.use_app_config is True

    def test_enable_logging_defaults_to_true(self):
        cfg = NetQASMConfig()
        assert cfg.enable_logging is True

    def test_hardware_defaults_to_generic(self):
        cfg = NetQASMConfig()
        assert cfg.hardware == "generic"


class TestNetQASMConfigCustomValues:
    """Verify custom values are stored properly."""

    def test_custom_network_config(self):
        from unittest.mock import MagicMock
        nc = MagicMock()
        cfg = NetQASMConfig(network_config=nc)
        assert cfg.network_config is nc

    def test_custom_post_function(self):
        fn = lambda: None
        cfg = NetQASMConfig(post_function=fn)
        assert cfg.post_function is fn


class TestCLIArgumentParsing:
    """Test the CLI argument parser structure (mirrors cli.py:main parser)."""

    @pytest.fixture
    def parser(self):
        """Replicate the argument parser from cli.py."""
        p = argparse.ArgumentParser()
        p.add_argument("-np", "--num-procs", type=int, required=True)
        p.add_argument("script", type=str)
        p.add_argument("script_args", nargs=argparse.REMAINDER)
        return p

    def test_parses_basic_invocation(self, parser):
        args = parser.parse_args(["-np", "4", "my_script.py"])
        assert args.num_procs == 4
        assert args.script == "my_script.py"
        assert args.script_args == []

    def test_parses_script_args(self, parser):
        args = parser.parse_args(["-np", "2", "script.py", "--foo", "bar"])
        assert args.script_args == ["--foo", "bar"]

    def test_np_is_required(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["script.py"])

    def test_script_is_required(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["-np", "2"])

    def test_num_procs_must_be_int(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["-np", "abc", "script.py"])
