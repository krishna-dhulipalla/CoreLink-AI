import sys
import types


fastmcp_module = types.ModuleType("mcp.server.fastmcp")


class _FakeFastMCP:
    def __init__(self, *_args, **_kwargs):
        pass

    def tool(self):
        def _decorator(fn):
            return fn

        return _decorator


fastmcp_module.FastMCP = _FakeFastMCP
server_module = types.ModuleType("mcp.server")
mcp_module = types.ModuleType("mcp")
sys.modules.setdefault("mcp", mcp_module)
sys.modules.setdefault("mcp.server", server_module)
sys.modules.setdefault("mcp.server.fastmcp", fastmcp_module)

from mcp_servers.file_handler.server import _is_probably_binary, _sniff_format


def test_file_handler_sniffs_pdf_from_magic_bytes_even_without_extension():
    raw = b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog >>\n"

    fmt = _sniff_format("https://example.com/download?id=1945", "text/plain", raw)

    assert fmt == "pdf"


def test_file_handler_detects_unknown_binary_payloads():
    raw = bytes([0, 159, 146, 150, 0, 200, 210, 33, 0, 14, 255, 128]) * 32

    assert _is_probably_binary(raw) is True
