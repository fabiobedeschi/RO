schema = {
    "type": "object",
    "properties": {
        "directed": {"type": "boolean"},
        "edges": {
            "type": "object",
            "patternProperties": {
                "^[a-zA-Z0-9_]+$": {
                    "type": "object",
                    "patternProperties": {"^[a-zA-Z0-9_]+$": {"type": "number"}},
                },
            },
        },
    },
}
