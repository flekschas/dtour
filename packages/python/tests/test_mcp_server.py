"""Tests for the MCP server: the tools a host and the viewer call, and the HTTP server."""

import base64
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import arro3.io
import numpy as np
import pytest

Client = pytest.importorskip("mcp").Client
pl = pytest.importorskip("polars")

if not (Path(__file__).parents[1] / "src/dtour/static/mcp-app.js").exists():
    pytest.skip("Build the viewer first with `pnpm build:widget`", allow_module_level=True)

from dtour import mcp_server  # noqa: E402
from dtour.data import _read_table, from_numpy  # noqa: E402
from dtour.spec import read_spec_from_parquet  # noqa: E402

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def csv(tmp_path):
    rng = np.random.default_rng(0)
    n = 300
    labels = rng.integers(0, 3, n)
    X = rng.normal(0, 1, (n, 4)) + labels[:, None] * 3
    df = pl.DataFrame(
        {
            "id": np.arange(n),
            **{name: X[:, i] for i, name in enumerate("abcd")},
            "species": [f"s{label}" for label in labels],
        }
    ).with_columns(
        # One row misses a tour column, and another column misses a value
        pl.when(pl.int_range(n) == 5).then(None).otherwise(pl.col("a")).alias("a"),
        pl.when(pl.int_range(n) == 7).then(None).otherwise(pl.col("a") * 2).alias("extra"),
    )
    path = tmp_path / "data.csv"
    df.write_csv(path)
    return path


async def visualize(client, **arguments):
    result = await client.call_tool("visualize", arguments)
    assert not result.is_error, result.content[0].text
    data = urllib.request.urlopen(result.structured_content["views"][0]["url"]).read()
    return result, data


async def test_tools_link_to_the_viewer():
    async with Client(mcp_server.create_server()) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        uri = tools["visualize"].meta["ui"]["resourceUri"]
        assert uri.startswith("ui://dtour/viewer-")
        assert tools["read_data"].meta["ui"]["visibility"] == ["app"]

        resource = await client.read_resource(uri)
        content = resource.contents[0]
        assert content.mime_type == "text/html;profile=mcp-app"
        assert content.meta["ui"]["csp"]["connectDomains"][0].startswith("http://127.0.0.1:")


async def test_each_server_names_its_viewer_uniquely():
    # Hosts cache the viewer by URI, and its CSP names the server's port
    uris = []
    for _ in range(2):
        async with Client(mcp_server.create_server()) as client:
            tools = (await client.list_tools()).tools
            uris.append(tools[0].meta["ui"]["resourceUri"])
    assert uris[0] != uris[1]


async def test_visualize_embeds_the_tour_and_settings(csv, tmp_path):
    async with Client(mcp_server.create_server()) as client:
        result, data = await visualize(
            client,
            path=str(csv),
            columns=["a", "b", "c", "d"],
            settings={"point_color_by": "species"},
        )

    text = result.content[0].text
    assert "a pca tour over 4 columns (a, b, c, d) for 299 rows" in text
    assert "Dropped 1 of 300 rows" in text
    assert "Dropped columns with missing values: ['extra']" in text

    path = tmp_path / "view.parquet"
    path.write_bytes(data)
    spec = read_spec_from_parquet(path)
    assert spec["pointColorBy"] == "species"
    assert spec["tour"]["dimensions"] == ["a", "b", "c", "d"]
    assert _read_table(data).num_rows == 299


async def test_embedding_tours_add_their_columns(csv):
    async with Client(mcp_server.create_server()) as client:
        _, data = await visualize(client, path=str(csv), tour="le", columns=["a", "b", "c", "d"])

    table = _read_table(data)
    dims = read_spec_from_parquet(table)["tour"]["dimensions"]
    assert len(dims) > 2
    assert table.column_names[: len(dims)] == dims


async def test_csv_missing_value_markers(tmp_path):
    path = tmp_path / "na.csv"
    path.write_text("a,b,c\n1,2,x\nNA,3,y\n4,NA,z\n5,6,x\n7,8,y\n")
    async with Client(mcp_server.create_server()) as client:
        result, data = await visualize(client, path=str(path))

    assert "a static scatter of a (x) and b (y)" in result.content[0].text
    assert _read_table(data).num_rows == 3


@pytest.mark.parametrize("write", [arro3.io.write_ipc, arro3.io.write_ipc_stream])
async def test_arrow_files_and_streams(tmp_path, write):
    path = tmp_path / "data.arrow"
    write(_read_table(from_numpy(np.random.default_rng(0).normal(size=(50, 3)))), str(path))
    async with Client(mcp_server.create_server()) as client:
        result, data = await visualize(client, path=str(path))

    assert "a pca tour over 3 columns (dim_0, dim_1, dim_2) for 50 rows" in result.content[0].text
    assert _read_table(data).num_rows == 50


async def test_decimal_columns_are_numeric(tmp_path):
    path = tmp_path / "data.parquet"
    rng = np.random.default_rng(0)
    pl.DataFrame({"amount": rng.normal(size=20), "other": rng.normal(size=20)}).with_columns(
        pl.col("amount").cast(pl.Decimal(10, 2))
    ).write_parquet(path)
    async with Client(mcp_server.create_server()) as client:
        result, data = await visualize(client, path=str(path))

    assert "a static scatter of amount (x) and other (y)" in result.content[0].text
    assert _read_table(data).num_rows == 20


async def test_boolean_columns_become_zeros_and_ones(tmp_path):
    # The viewer reads boolean values as zeros
    path = tmp_path / "data.parquet"
    rng = np.random.default_rng(0)
    flags = [True, False] * 10
    pl.DataFrame({"a": rng.normal(size=20), "b": rng.normal(size=20), "flag": flags}).write_parquet(
        path
    )
    async with Client(mcp_server.create_server()) as client:
        result, data = await visualize(client, path=str(path))

    assert "over 3 columns (a, b, flag)" in result.content[0].text
    flag = pl.read_parquet(data)["flag"]
    assert flag.dtype.is_integer()
    assert flag.to_list() == [int(f) for f in flags]


async def test_sample(csv):
    async with Client(mcp_server.create_server()) as client:
        result, data = await visualize(
            client, path=str(csv), columns=["a", "b", "c", "d"], sample=100
        )

    assert "Sampled 100 of 299 rows" in result.content[0].text
    assert _read_table(data).num_rows == 100


def test_read_loads_urls_from_hosts_that_reject_python(csv):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.headers["User-Agent"].startswith("Python-urllib"):
                self.send_error(403)
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(csv.read_bytes())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        df = mcp_server._read(f"http://127.0.0.1:{server.server_address[1]}/data.csv")
    finally:
        server.shutdown()
    assert df.equals(mcp_server._read(str(csv)))


async def test_read_data_returns_the_file_in_chunks(csv, monkeypatch):
    monkeypatch.setattr(mcp_server, "CHUNK_SIZE", 10_000)
    async with Client(mcp_server.create_server()) as client:
        result, data = await visualize(client, path=str(csv))
        token = result.structured_content["token"]

        chunks = []
        while sum(map(len, chunks)) < len(data):
            chunk = await client.call_tool(
                "read_data", {"token": token, "offset": sum(map(len, chunks))}
            )
            assert chunk.structured_content["size"] == len(data)
            chunks.append(base64.b64decode(chunk.structured_content["base64"]))

    assert len(chunks) > 1
    assert b"".join(chunks) == data


async def test_describe_selection_compares_selected_rows(csv):
    async with Client(mcp_server.create_server()) as client:
        result, data = await visualize(client, path=str(csv), columns=["a", "b", "c", "d"])

        # The viewer sends a bit mask in little-endian uint32 words, one bit per row
        species = pl.read_parquet(data)["species"].to_numpy()
        selected = np.flatnonzero(species == "s2")
        words = np.zeros((len(species) + 31) // 32, "<u4")
        np.bitwise_or.at(words, selected // 32, (1 << (selected % 32)).astype("<u4"))
        mask = base64.b64encode(words.tobytes()).decode()
        summary = await client.call_tool("describe_selection", {"token": "?", "mask": mask})
        assert summary.is_error
        summary = await client.call_tool(
            "describe_selection", {"token": result.structured_content["token"], "mask": mask}
        )

    text = summary.content[0].text
    assert text.startswith(f"{len(selected)} of {len(species)} rows selected")
    assert "- species: s2 100% (" in text
    # Species s2 sits 6 units above s0 in every column
    assert "+" in text.split("\n")[2]


async def test_hosts_without_apps_get_a_browser_link(csv):
    async with Client(mcp_server.create_server()) as client:
        result, _ = await visualize(client, path=str(csv))

    url = result.content[0].text.split("Open the viewer in a browser: ")[1]
    html = urllib.request.urlopen(url).read().decode()
    assert html.startswith("<!doctype html>")
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(url.rsplit("/", 1)[0] + "/unknown")


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"columns": ["a", "nope"]}, "No columns named ['nope']"),
        ({"columns": ["a", "species"]}, "must be numeric"),
        ({"columns": ["a"]}, "at least 2 numeric columns"),
        ({"views": [{}] * 5}, "at most 4 views"),
        ({"views": [{"colums": ["a", "b"]}]}, "colums"),
        ({"tour": "le", "dimensions": 2}, 'needs tour "pca" or "umap"'),
        ({"dimensions": 1}, "at least 2"),
        ({"views": [{"columns": ["a", "b"]}, {"columns": ["nope", "b"]}]}, "['nope']"),
        ({"settings": {"point_colour_by": "species"}}, "point_colour_by"),
    ],
)
async def test_visualize_reports_invalid_arguments(csv, arguments, message):
    async with Client(mcp_server.create_server()) as client:
        result = await client.call_tool("visualize", {"path": str(csv), **arguments})

    assert result.is_error
    assert message in result.content[0].text


def load_views(result):
    return [
        urllib.request.urlopen(view["url"]).read() for view in result.structured_content["views"]
    ]


async def test_views_show_the_same_rows_with_their_own_tours(csv):
    async with Client(mcp_server.create_server()) as client:
        result = await client.call_tool(
            "visualize",
            {
                "path": str(csv),
                "settings": {"point_color_by": "species"},
                "views": [
                    {"columns": ["a", "b", "c"]},
                    {"columns": ["c", "d"], "settings": {"point_size": 3}},
                ],
            },
        )
        assert not result.is_error, result.content[0].text
        tour, scatter = load_views(result)

    text = result.content[0].text
    assert "Showing 2 views of the same 299 rows side by side" in text
    assert "- View 1: a pca tour over 3 columns (a, b, c)" in text
    assert "- View 2: a static scatter of c (x) and d (y)" in text
    # Rows missing a value in any view's columns are dropped from all views
    assert "Dropped 1 of 300 rows" in text

    tour_spec, scatter_spec = (
        read_spec_from_parquet(_read_table(data)) for data in (tour, scatter)
    )
    assert tour_spec["tour"]["dimensions"] == ["a", "b", "c"]
    assert "tour" not in scatter_spec
    assert scatter_spec["tourDimensions"] == ["c", "d"]
    assert tour_spec["pointColorBy"] == scatter_spec["pointColorBy"] == "species"
    assert scatter_spec["pointSize"] == 3
    assert "pointSize" not in tour_spec
    assert tour_spec["showLegend"] is scatter_spec["showLegend"] is False
    assert pl.read_parquet(tour)["id"].equals(pl.read_parquet(scatter)["id"])


async def test_two_dimensions_show_a_static_embedding(csv):
    async with Client(mcp_server.create_server()) as client:
        result, data = await visualize(
            client, path=str(csv), columns=["a", "b", "c", "d"], dimensions=2
        )

    assert "a static 2D pca embedding of 4 columns" in result.content[0].text
    table = _read_table(data)
    spec = read_spec_from_parquet(table)
    assert "tour" not in spec
    assert spec["tourDimensions"] == ["PC1", "PC2"]
    assert table.column_names[:2] == ["PC1", "PC2"]


@pytest.mark.parametrize("view", [{"columns": ["a", "b"]}, {"dimensions": 2}])
async def test_static_views_ignore_an_inherited_pca_tour(csv, view):
    # A PCA tour uses all numeric columns, so the viewer would tour instead
    async with Client(mcp_server.create_server()) as client:
        result = await client.call_tool(
            "visualize",
            {"path": str(csv), "settings": {"tour_by": "pca"}, "views": [{}, view]},
        )
        assert not result.is_error, result.content[0].text
        tour, scatter = load_views(result)

    assert read_spec_from_parquet(_read_table(tour))["tourBy"] == "pca"
    assert read_spec_from_parquet(_read_table(scatter))["tourBy"] == "dimensions"


async def test_a_umap_tour_next_to_a_2d_umap(csv):
    pytest.importorskip("umap")
    async with Client(mcp_server.create_server()) as client:
        result = await client.call_tool(
            "visualize",
            {
                "path": str(csv),
                "tour": "umap",
                "columns": ["a", "b", "c", "d"],
                "views": [{"dimensions": 4}, {"dimensions": 2}],
            },
        )
        assert not result.is_error, result.content[0].text
        tour, scatter = load_views(result)

    assert "View 1: a umap tour over a 4D embedding of 4 columns" in result.content[0].text
    assert "View 2: a static 2D umap embedding of 4 columns" in result.content[0].text
    assert read_spec_from_parquet(_read_table(tour))["tour"]["dimensions"] == [
        "UMAP1",
        "UMAP2",
        "UMAP3",
        "UMAP4",
    ]
    assert read_spec_from_parquet(_read_table(scatter))["tourDimensions"] == ["UMAP1", "UMAP2"]


async def test_the_manifest_and_read_data_cover_every_view(csv):
    async with Client(mcp_server.create_server()) as client:
        result = await client.call_tool(
            "visualize", {"path": str(csv), "views": [{}, {"columns": ["a", "b"]}]}
        )
        token = result.structured_content["token"]
        chunk = await client.call_tool("read_data", {"token": token, "view": 1})
        missing = await client.call_tool("read_data", {"token": token, "view": 2})

    views = result.structured_content["views"]
    origin = views[0]["url"].split("/data/")[0]
    manifest = json.loads(urllib.request.urlopen(f"{origin}/data/{token}").read())
    assert manifest == result.structured_content
    assert base64.b64decode(chunk.structured_content["base64"]) == load_views(result)[1]
    assert missing.is_error
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(f"{origin}/data/{token}/2")
