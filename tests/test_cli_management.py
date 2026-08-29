from tars.cli import build_parser


def test_management_cli_parsing():
    parser = build_parser()
    assert parser.parse_args(["model", "assign", "general", "qwen"]).model_command == "assign"
    assert parser.parse_args(["model", "unassign", "oracle"]).model_command == "unassign"
    assert parser.parse_args(["model", "swap", "builder", "coder"]).model_command == "swap"
    args = parser.parse_args(["role", "profile", "general", "compact"])
    assert (args.role_command, args.profile) == ("profile", "compact")
    assert parser.parse_args(["start"]).command == "start"
    assert parser.parse_args(["stop"]).command == "stop"
    assert parser.parse_args(["logs", "--lines", "20"]).lines == 20


def test_model_lifecycle_cli_parsing():
    parser = build_parser()
    pull = parser.parse_args(["model", "pull", "org/repo", "--filename", "m.gguf", "--alias", "m"])
    assert pull.source == "org/repo" and pull.alias == "m"
    imported = parser.parse_args(["model", "import", "/tmp/m.gguf", "--alias", "m"])
    assert imported.path == "/tmp/m.gguf"
    assert parser.parse_args(["model", "verify", "m"]).alias == "m"


def test_calibration_cli_parsing():
    parser = build_parser()
    default = parser.parse_args(["calibrate"])
    assert default.models == [] and not default.mid and not default.max
    mid = parser.parse_args(["calibrate", "one", "two", "--mid"])
    assert mid.models == ["one", "two"] and mid.mid
    maximum = parser.parse_args(["calibrate", "one", "--max", "--fresh"])
    assert maximum.max and maximum.fresh


def test_runtime_backend_cli_parsing():
    parser = build_parser()
    assert parser.parse_args(["backend", "list"]).backend_command == "list"
    status = parser.parse_args(["backend", "status", "llama.cpp"])
    assert status.backend == "llama.cpp"
