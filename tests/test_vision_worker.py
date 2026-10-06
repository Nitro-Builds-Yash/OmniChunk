"""Tests for image and video vision processing with Gemini and local fallback."""

import json
from unittest.mock import MagicMock, patch

from cleave.workers.vision_worker import process_image_file


def test_image_processing_offline_fallback(tmp_path):
    img_file = tmp_path / "diagram.png"
    img_file.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR...")

    with patch("cleave.workers.vision_worker.settings") as mock_settings:
        mock_settings.return_value = MagicMock(gemini_api_key="")
        res = process_image_file(img_file)
        assert len(res.elements) >= 1
        assert res.elements[0].kind == "figure"
        assert "diagram.png" in res.elements[0].text


def test_image_processing_mock_gemini(tmp_path):
    img_file = tmp_path / "chart.png"
    img_file.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR...")

    mock_gemini_payload = {
        "title": "Quarterly Revenue Chart",
        "visual_description": "A bar chart showing revenue growth from Q1 to Q4.",
        "ocr_text": "Revenue Growth 2026",
        "entities": ["Revenue", "Q1", "Q4"],
        "elements": [
            {"kind": "heading", "text": "Revenue Growth 2026", "bbox": [0.1, 0.1, 0.2, 0.9]},
            {
                "kind": "figure",
                "text": "Bar chart showing revenue growth",
                "bbox": [0.2, 0.1, 0.8, 0.9],
            },
            {
                "kind": "caption",
                "text": "Figure 1: Quarterly breakdown",
                "bbox": [0.8, 0.1, 0.9, 0.9],
            },
        ],
    }

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": json.dumps(mock_gemini_payload)}],
                },
            }
        ],
    }

    with (
        patch("cleave.workers.vision_worker.settings") as mock_settings,
        patch("cleave.workers.vision_worker.request_with_retry", return_value=mock_response),
    ):
        mock_settings.return_value = MagicMock(
            gemini_api_key="mock_key", gemini_model="gemini-2.5", llm_timeout_s=30
        )
        res = process_image_file(img_file)
        assert res.title == "Quarterly Revenue Chart"
        assert len(res.elements) == 3
        assert res.elements[0].kind == "heading"
        assert res.elements[1].kind == "figure"
        assert res.elements[2].kind == "caption"
