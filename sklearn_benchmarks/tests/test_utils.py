import tempfile
from pathlib import Path

from sklearn_benchmarks.utils import save_libraries_version, load_libraries_version

def test_save_load_libraries_version():
    libraries = ["numpy", "scipy", "scikit-learn"]

    with tempfile.NamedTemporaryFile() as json_dump:
        json_dump_posix_path = Path(json_dump.name)
        save_libraries_version(version_file=json_dump_posix_path, libraries=libraries)
        versions = load_libraries_version(version_file=json_dump_posix_path)

    assert all(lib in versions for lib in libraries)

    assert versions['random_package'] == 'unknown'

