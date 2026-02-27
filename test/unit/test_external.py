import os
import tempfile

import pytest
from unittest.mock import patch, MagicMock


class TestImportModuleFromPath:
    """Tests for the dynamic module loader."""

    def test_imports_valid_python_file(self):
        from netqmpi.sdk.external import import_module_from_path

        with tempfile.NamedTemporaryFile(suffix=".py", mode='w', delete=False) as f:
            f.write("VALUE = 42\ndef main(): pass\n")
            f.flush()
            path = f.name
        try:
            module = import_module_from_path(path)
            assert module.VALUE == 42
            assert callable(module.main)
        finally:
            os.unlink(path)

    def test_raises_on_nonexistent_file(self):
        from netqmpi.sdk.external import import_module_from_path

        with pytest.raises((ImportError, FileNotFoundError, OSError)):
            import_module_from_path("/nonexistent/path/module.py")

    def test_module_registered_in_sys_modules(self):
        import sys
        from netqmpi.sdk.external import import_module_from_path

        with tempfile.NamedTemporaryFile(suffix=".py", mode='w', delete=False, prefix="test_mod_") as f:
            f.write("X = 1\n")
            f.flush()
            path = f.name
            module_name = os.path.splitext(os.path.basename(path))[0]
        try:
            import_module_from_path(path)
            assert module_name in sys.modules
        finally:
            os.unlink(path)
            sys.modules.pop(module_name, None)


class TestAppInstanceFromFileValidation:
    """Tests for input validation in app_instance_from_file."""

    def test_raises_if_file_is_none(self):
        from netqmpi.sdk.external import app_instance_from_file

        with pytest.raises(ValueError, match="file must be provided"):
            app_instance_from_file(file=None)

    def test_raises_if_not_py_extension(self):
        from netqmpi.sdk.external import app_instance_from_file

        with pytest.raises(ValueError, match="must be a .py file"):
            app_instance_from_file(file="script.txt")

    def test_raises_for_yaml_extension(self):
        from netqmpi.sdk.external import app_instance_from_file

        with pytest.raises(ValueError, match="must be a .py file"):
            app_instance_from_file(file="config.yaml")


class TestAppInstanceFromFileCreation:
    """Tests for correct ApplicationInstance creation."""

    def _create_temp_script(self):
        """Helper to create a temp Python file with a main function."""
        f = tempfile.NamedTemporaryFile(suffix=".py", mode='w', delete=False)
        f.write("def main(**kwargs): pass\n")
        f.flush()
        f.close()
        return f.name

    def test_creates_correct_number_of_programs(self):
        from netqmpi.sdk.external import app_instance_from_file
        path = self._create_temp_script()
        try:
            with patch('netqmpi.sdk.external.env.load_roles_config', return_value=None):
                app_instance = app_instance_from_file(path, num_processes=3)
                assert len(app_instance.app.programs) == 3
        finally:
            os.unlink(path)

    def test_program_parties_named_by_rank(self):
        from netqmpi.sdk.external import app_instance_from_file
        path = self._create_temp_script()
        try:
            with patch('netqmpi.sdk.external.env.load_roles_config', return_value=None):
                app_instance = app_instance_from_file(path, num_processes=3)
                parties = [p.party for p in app_instance.app.programs]
                assert parties == ["rank_0", "rank_1", "rank_2"]
        finally:
            os.unlink(path)

    def test_argv_does_not_contain_rank_or_size(self):
        from netqmpi.sdk.external import app_instance_from_file
        path = self._create_temp_script()
        try:
            with patch('netqmpi.sdk.external.env.load_roles_config', return_value=None):
                app_instance = app_instance_from_file(path, num_processes=2)
                inputs = app_instance.program_inputs
                assert "rank" not in inputs["rank_0"]
                assert "size" not in inputs["rank_0"]
                assert "rank" not in inputs["rank_1"]
                assert "size" not in inputs["rank_1"]
        finally:
            os.unlink(path)

    def test_party_alloc_maps_rank_to_self_when_no_roles_file(self):
        from netqmpi.sdk.external import app_instance_from_file
        path = self._create_temp_script()
        try:
            with patch('netqmpi.sdk.external.env.load_roles_config', return_value=None):
                app_instance = app_instance_from_file(path, num_processes=2)
                assert app_instance.party_alloc == {
                    "rank_0": "rank_0",
                    "rank_1": "rank_1",
                }
        finally:
            os.unlink(path)

    def test_single_process(self):
        from netqmpi.sdk.external import app_instance_from_file
        path = self._create_temp_script()
        try:
            with patch('netqmpi.sdk.external.env.load_roles_config', return_value=None):
                app_instance = app_instance_from_file(path, num_processes=1)
                assert len(app_instance.app.programs) == 1
                assert "rank" not in app_instance.program_inputs["rank_0"]
                assert "size" not in app_instance.program_inputs["rank_0"]
        finally:
            os.unlink(path)


class TestCommunicatorInjector:
    """Tests for the _make_communicator_injector wrapper."""

    def test_wrapper_creates_communicator_and_injects_as_comm(self):
        from netqmpi.sdk.external import _make_communicator_injector
        from unittest.mock import MagicMock, patch

        received = {}

        def fake_main(comm=None):
            received['comm'] = comm

        with patch('netqmpi.sdk.external.QMPICommunicator') as MockComm:
            mock_instance = MagicMock()
            MockComm.return_value = mock_instance
            mock_app_config = MagicMock()

            wrapper = _make_communicator_injector(fake_main, rank=1, size=3)
            wrapper(app_config=mock_app_config)

            MockComm.assert_called_once_with(1, 3, mock_app_config)
            assert received['comm'] is mock_instance

    def test_wrapper_captures_rank_and_size_in_closure(self):
        from netqmpi.sdk.external import _make_communicator_injector
        from unittest.mock import MagicMock, patch

        with patch('netqmpi.sdk.external.QMPICommunicator') as MockComm:
            MockComm.return_value = MagicMock()
            wrapper_0 = _make_communicator_injector(lambda comm=None: None, rank=0, size=4)
            wrapper_2 = _make_communicator_injector(lambda comm=None: None, rank=2, size=4)

            wrapper_0(app_config=None)
            wrapper_2(app_config=None)

            calls = MockComm.call_args_list
            assert calls[0][0] == (0, 4, None)
            assert calls[1][0] == (2, 4, None)

    def test_entry_points_are_wrapped(self):
        from netqmpi.sdk.external import app_instance_from_file
        import tempfile, os
        from unittest.mock import patch

        f = tempfile.NamedTemporaryFile(suffix=".py", mode='w', delete=False)
        f.write("def main(comm=None): pass\n")
        f.flush()
        f.close()
        try:
            with patch('netqmpi.sdk.external.env.load_roles_config', return_value=None):
                app_instance = app_instance_from_file(f.name, num_processes=2)
                # The entry is the wrapper, not the original main
                for prog in app_instance.app.programs:
                    assert prog.entry.__name__ == "wrapper"
        finally:
            os.unlink(f.name)
