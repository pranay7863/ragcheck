"""Tests for HTML report generation."""

from ragcheck.reports.html_report import generate_report


class TestGenerateReport:
    def test_generates_html(self):
        html = generate_report(
            project_name="test-project",
            overall_score=0.65,
            retrieval_score=0.70,
            faithfulness_score=0.60,
            tests_passed=35,
            tests_total=50,
            failures=[],
            recommendations=[],
            current_score=0.65,
            predicted_score=0.85,
            chunk_strategy="recursive",
            num_chunks=42,
            chunk_lengths=[400, 512, 300, 450],
            heatmap_data=[{"z": [[0]], "type": "heatmap", "colorscale": "RdYlGn"}],
        )
        assert "ragcheck Report" in html
        assert "test-project" in html

    def test_score_color_classes(self):
        html = generate_report(
            project_name="test",
            overall_score=0.95,
            retrieval_score=0.75,
            faithfulness_score=0.45,
            tests_passed=10,
            tests_total=10,
            failures=[],
            recommendations=[],
            current_score=0.95,
            predicted_score=1.0,
            chunk_strategy="fixed",
            num_chunks=5,
            chunk_lengths=[100, 200],
            heatmap_data=[{"z": [[0]], "type": "heatmap", "colorscale": "RdYlGn"}],
        )
        assert "score-excellent" in html
        assert "score-good" in html
        assert "score-poor" in html

    def test_recommendations_rendered(self):
        html = generate_report(
            project_name="test",
            overall_score=0.5,
            retrieval_score=0.5,
            faithfulness_score=0.5,
            tests_passed=5,
            tests_total=10,
            failures=[],
            recommendations=[{
                "title": "Fix chunking",
                "description": "Use semantic chunking",
                "expected_improvement": 15,
                "tradeoffs": "None",
                "difficulty": "easy",
                "code_example": "strategy = 'semantic'",
            }],
            current_score=0.5,
            predicted_score=0.65,
            chunk_strategy="fixed",
            num_chunks=5,
            chunk_lengths=[100],
            heatmap_data=[{"z": [[0]], "type": "heatmap", "colorscale": "RdYlGn"}],
        )
        assert "Fix chunking" in html
        assert "Use semantic chunking" in html
