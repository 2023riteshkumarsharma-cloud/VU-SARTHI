import mimetypes

import pytest
from fastapi.testclient import TestClient

from advisor import config, server


@pytest.mark.parametrize("source_key", ["semester_spread", "minors"])
def test_workbook_source_download_has_excel_filename_and_content(source_key):
    path = config.SOURCES[source_key]
    response = TestClient(server.app).get(f"/api/source/{source_key}")

    assert response.status_code == 200
    assert response.headers["content-type"] == mimetypes.guess_type(path.name)[0]
    assert response.headers["content-disposition"] == f'attachment; filename="{path.name}"'
    assert response.content == path.read_bytes()
