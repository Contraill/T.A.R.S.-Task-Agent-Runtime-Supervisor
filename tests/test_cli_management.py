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


def test_memory_cli_parsing():
    parser = build_parser()
    remember = parser.parse_args([
        "memory", "remember", "prefers concise answers", "--scope", "profile",
        "--tag", "preference",
    ])
    assert remember.memory_command == "remember" and remember.tag == ["preference"]
    search = parser.parse_args(["memory", "search", "concise", "--limit", "3"])
    assert search.query == "concise" and search.limit == 3


def test_context_epoch_cli_parsing():
    parser = build_parser()
    epochs = parser.parse_args(["context", "epochs", "task-one"])
    assert epochs.task_id == "task-one"
    search = parser.parse_args(["context", "search", "conv-one", "needle"])
    assert search.conversation_id == "conv-one" and search.query == "needle"


def test_temporary_cli_parsing():
    args = build_parser().parse_args(["temporary", "--role", "general"])
    assert args.command == "temporary" and args.role == "general"


def test_memory_maintenance_cli_parsing():
    args = build_parser().parse_args([
        "memory", "maintain", "--trigger", "scheduled", "--apply",
    ])
    assert args.memory_command == "maintain" and args.trigger == "scheduled" and args.apply


def test_scheduler_cli_surface():
    parser = build_parser()
    add = parser.parse_args([
        "schedule", "add", "task-one", "recurring", "every 10m",
        "--missed", "catch-up", "--max-catch-up", "3",
    ])
    assert add.schedule_command == "add" and add.max_catch_up == 3
    assert parser.parse_args(["schedule", "pause", "sch-one"]).schedule_id == "sch-one"
    assert parser.parse_args(["schedule", "run-due"]).schedule_command == "run-due"


def test_policy_approval_and_audit_cli_parsing():
    parser = build_parser()
    scope = parser.parse_args([
        "scope", "explain", "fs.write", "write", "/tmp/work/file",
        "--allow-path", "/tmp/work",
    ])
    assert scope.scope_command == "explain" and scope.allow_path == ["/tmp/work"]
    approval = parser.parse_args(["approvals", "--approve", "approval-one"])
    assert approval.approve == "approval-one"
    audit = parser.parse_args(["audit", "--state", "denied"])
    assert audit.state == "denied"
    backend = parser.parse_args(["execution-backend", "container"])
    assert backend.backend == "container"
