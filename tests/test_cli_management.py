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
