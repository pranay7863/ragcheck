"""Generate beautiful HTML reports."""

from typing import Any

from jinja2 import Template

REPORT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ragcheck Report — {{ project_name }}</title>
    <style>
        :root {
            --primary: #e94560;
            --secondary: #1a1a2e;
            --success: #4caf50;
            --warning: #ff9800;
            --danger: #f44336;
            --bg: #f5f5f5;
            --card: #ffffff;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--secondary);
            line-height: 1.6;
        }
        .header {
            background: var(--secondary);
            color: white;
            padding: 40px;
            text-align: center;
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header .subtitle { opacity: 0.8; font-size: 1.1em; }
        .scorecards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            max-width: 1200px;
            margin: -30px auto 0;
        }
        .scorecard {
            background: var(--card);
            border-radius: 12px;
            padding: 25px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .scorecard:hover { transform: translateY(-5px); }
        .score-value {
            font-size: 3em;
            font-weight: bold;
            margin: 10px 0;
        }
        .score-excellent { color: var(--success); }
        .score-good { color: #8bc34a; }
        .score-fair { color: var(--warning); }
        .score-bad { color: var(--danger); }
        .score-na { color: #999; }
        .score-label {
            color: #666; font-size: 0.9em;
            text-transform: uppercase; letter-spacing: 1px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 30px;
        }
        .section {
            background: var(--card);
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .section h2 {
            color: var(--secondary);
            border-bottom: 2px solid var(--primary);
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .failure-card {
            border-left: 4px solid;
            padding: 15px;
            margin: 10px 0;
            background: #fafafa;
            border-radius: 0 6px 6px 0;
        }
        .failure-retrieval { border-color: var(--danger); }
        .failure-overload { border-color: var(--warning); }
        .failure-hallucination { border-color: #9c27b0; }
        .failure-boundary { border-color: #2196f3; }
        .failure-none { border-color: var(--success); }
        .recommendation {
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 15px;
            margin: 10px 0;
            border-radius: 0 6px 6px 0;
        }
        .recommendation .title { font-weight: bold; color: #1565c0; }
        .recommendation .expected { color: var(--success); font-weight: bold; }
        .heatmap-container { overflow-x: auto; padding-bottom: 10px; }
        .heatmap-grid {
            display: grid;
            gap: 2px;
            margin: 20px 0;
        }
        .heatmap-cell {
            width: 24px;
            height: 24px;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.6em;
            font-weight: bold;
        }
        .cell-dead { background: #ffcdd2; color: #c62828; }
        .cell-good { background: #c8e6c9; color: #2e7d32; }
        .cell-dominant { background: #fff9c4; color: #f57f17; }
        .bar-chart { max-width: 600px; margin: 20px 0; }
        .bar-row {
            display: flex;
            align-items: center;
            margin: 12px 0;
            gap: 12px;
        }
        .bar-label { width: 80px; font-weight: bold; color: #555; }
        .bar-track {
            flex: 1;
            background: #eee;
            border-radius: 6px;
            height: 28px;
            overflow: hidden;
        }
        .bar-fill {
            height: 100%;
            border-radius: 6px;
            transition: width 0.5s ease;
        }
        .bar-current { background: var(--danger); }
        .bar-predicted { background: var(--success); }
        .bar-value { width: 50px; text-align: right; font-weight: bold; }
        .histogram { max-width: 600px; margin: 20px 0; }
        .hist-row {
            display: flex;
            align-items: center;
            margin: 6px 0;
            gap: 10px;
        }
        .hist-label { width: 100px; font-size: 0.85em; color: #666; text-align: right; }
        .hist-track {
            flex: 1;
            background: #eee;
            border-radius: 4px;
            height: 20px;
            overflow: hidden;
        }
        .hist-bar {
            height: 100%;
            background: var(--primary);
            border-radius: 4px;
            transition: width 0.5s ease;
        }
        .hist-value { width: 40px; font-size: 0.85em; color: #555; text-align: right; }
        .chunk-stat-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .chunk-stat-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        .chunk-stat-value { font-size: 1.8em; font-weight: bold; color: var(--primary); }
        .chunk-stat-label { color: #666; font-size: 0.85em; margin-top: 4px; }
        .chunk-detail {
            border-left: 4px solid var(--primary);
            padding: 12px 15px;
            margin: 8px 0;
            background: #fafafa;
            border-radius: 0 6px 6px 0;
            font-size: 0.9em;
        }
        .chunk-detail pre {
            white-space: pre-wrap;
            word-break: break-word;
            margin-top: 6px;
            color: #444;
        }
        .footer {
            text-align: center;
            padding: 40px;
            color: #999;
            font-size: 0.9em;
        }
        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
            text-transform: uppercase;
        }
        .badge-easy { background: #c8e6c9; color: #2e7d32; }
        .badge-medium { background: #fff9c4; color: #f57f17; }
        .badge-hard { background: #ffcdd2; color: #c62828; }
        .na-badge {
            display: inline-block;
            background: #e0e0e0;
            color: #666;
            padding: 2px 8px;
            border-radius: 8px;
            font-size: 0.7em;
            margin-left: 6px;
            vertical-align: middle;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>ragcheck Report</h1>
        <p class="subtitle">{{ project_name }} | Generated {{ timestamp }}</p>
    </div>

    <div class="scorecards">
        <div class="scorecard">
            <div class="score-label">Overall Score</div>
            {% if overall_score is none %}<div class="score-value score-na">N/A</div>{% else %}<div class="score-value {{ overall_class }}">{{ overall_score }}<span>%</span></div>{% endif %}
        </div>
        <div class="scorecard">
            <div class="score-label">Retrieval Accuracy</div>
            {% if retrieval_score is none %}<div class="score-value score-na">N/A</div>{% else %}<div class="score-value {{ retrieval_class }}">{{ retrieval_score }}<span>%</span></div>{% endif %}
        </div>
        <div class="scorecard">
            <div class="score-label">Answer Faithfulness</div>
            {% if faithfulness_score is none %}
            <div class="score-value score-na">N/A <span class="na-badge">--generate-answers</span></div>
            {% else %}
            <div class="score-value {{ faithfulness_class }}">{{ faithfulness_score }}<span>%</span></div>
            {% endif %}
        </div>
        <div class="scorecard">
            <div class="score-label">Tests Passed</div>
            <div class="score-value">{{ tests_passed }}/{{ tests_total }}</div>
        </div>
    </div>

    <div class="container">
        <div class="section">
            <h2>Retrieval Heatmap</h2>
            <p>Green = retrieved | Red = not retrieved | Rows = chunks, Cols = queries</p>
            <div class="heatmap-container">
                <div class="heatmap-grid" style="grid-template-columns: repeat({{ heatmap_data[0].x|length }}, 24px);">
                    {% for row in heatmap_data[0].z %}
                        {% for cell in row %}
                            <div class="heatmap-cell {% if cell %}cell-good{% else %}cell-dead{% endif %}">{{ cell }}</div>
                        {% endfor %}
                    {% endfor %}
                </div>
            </div>
            <p style="font-size: 0.85em; color: #666; margin-top: 10px;">
                {{ heatmap_data[0].y|length }} chunks &times; {{ heatmap_data[0].x|length }} queries
            </p>
        </div>

        <div class="section">
            <h2>Failure Analysis</h2>
            {% for failure in failures %}
            <div class="failure-card failure-{{ failure.mode }}">
                <strong>{{ failure.mode|title }}</strong>
                (Confidence: {{ failure.confidence }}%)
                <p>{{ failure.explanation }}</p>
                <p><strong>Query:</strong> {{ failure.question }}</p>
            </div>
            {% endfor %}
        </div>

        <div class="section">
            <h2>Recommendations</h2>
            <p>Apply these fixes to improve your RAG system:</p>
            {% for rec in recommendations %}
            <div class="recommendation">
                <div class="title">{{ rec.title }}</div>
                <p>{{ rec.description }}</p>
                <p>
                    Expected improvement:
                    <span class="expected">+{{ rec.expected_improvement }}%</span>
                </p>
                <p>Tradeoffs: {{ rec.tradeoffs }}</p>
                <span class="badge badge-{{ rec.difficulty }}">{{ rec.difficulty }}</span>
                {% if rec.code_example %}
                <pre><code>{{ rec.code_example }}</code></pre>
                {% endif %}
            </div>
            {% endfor %}
        </div>

        <div class="section">
            <h2>Before / After Prediction</h2>
            <div class="bar-chart">
                <div class="bar-row">
                    <span class="bar-label">Current</span>
                    <div class="bar-track">
                        <div class="bar-fill bar-current" style="width: {% if current_score is none %}0{% else %}{{ (current_score * 100)|int }}{% endif %}%;"></div>
                    </div>
                    <span class="bar-value">{% if current_score is none %}N/A{% else %}{{ "%.0f"|format(current_score * 100) }}%{% endif %}</span>
                </div>
                <div class="bar-row">
                    <span class="bar-label">Predicted</span>
                    <div class="bar-track">
                        <div class="bar-fill bar-predicted" style="width: {% if predicted_score is none %}0{% else %}{{ (predicted_score * 100)|int }}{% endif %}%;"></div>
                    </div>
                    <span class="bar-value">{% if predicted_score is none %}N/A{% else %}{{ "%.0f"|format(predicted_score * 100) }}%{% endif %}</span>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Chunk Analysis</h2>
            <div class="chunk-stat-grid">
                <div class="chunk-stat-card">
                    <div class="chunk-stat-value">{{ num_chunks }}</div>
                    <div class="chunk-stat-label">Total Chunks</div>
                </div>
                <div class="chunk-stat-card">
                    <div class="chunk-stat-value">{{ avg_chunk_length|int }}</div>
                    <div class="chunk-stat-label">Avg Length (chars)</div>
                </div>
                <div class="chunk-stat-card">
                    <div class="chunk-stat-value">{{ context_loss_score|round(2) }}</div>
                    <div class="chunk-stat-label">Context Loss</div>
                </div>
                <div class="chunk-stat-card">
                    <div class="chunk-stat-value">{{ chunk_strategy }}</div>
                    <div class="chunk-stat-label">Strategy</div>
                </div>
            </div>
            <h3>Chunk Length Distribution</h3>
            <div class="histogram">
                {% for bin in chunk_histogram %}
                <div class="hist-row">
                    <span class="hist-label">{{ bin.range }}</span>
                    <div class="hist-track">
                        <div class="hist-bar" style="width: {{ bin.pct|int }}%;"></div>
                    </div>
                    <span class="hist-value">{{ bin.count }}</span>
                </div>
                {% endfor %}
            </div>
            <h3>Chunk Details</h3>
            {% for chunk in chunk_details %}
            <div class="chunk-detail">
                <strong>Chunk {{ loop.index }}</strong> ({{ chunk.start }}&ndash;{{ chunk.end }}) &mdash; {{ chunk.length }} chars
                <pre>{{ chunk.preview }}</pre>
            </div>
            {% endfor %}
        </div>
    </div>

    <div class="footer">
        <p>Generated by ragcheck &mdash; Lighthouse for RAG systems</p>
        <p><a href="https://github.com/pranay7863/ragcheck">Star on GitHub</a></p>
    </div>
</body>
</html>
"""


def generate_report(
    project_name: str,
    overall_score: float | None,
    retrieval_score: float | None,
    faithfulness_score: float | None,
    tests_passed: int,
    tests_total: int,
    failures: list[dict[str, Any]],
    recommendations: list[dict[str, Any]],
    current_score: float,
    predicted_score: float,
    chunk_strategy: str,
    num_chunks: int,
    chunk_histogram: list[dict[str, Any]],
    chunk_details: list[dict[str, Any]],
    avg_chunk_length: float,
    context_loss_score: float,
    heatmap_data: list[dict[str, Any]],
) -> str:
    """Generate a complete HTML report.

    Args:
        project_name: Name of the project
        overall_score: Overall RAG score (0-1)
        retrieval_score: Retrieval accuracy score (0-1)
        faithfulness_score: Answer faithfulness score (0-1), or None if not measured
        tests_passed: Number of tests passed
        tests_total: Total number of tests
        failures: List of failure dictionaries
        recommendations: List of recommendation dictionaries
        current_score: Current overall score
        predicted_score: Predicted score after recommendations
        chunk_strategy: Chunking strategy used
        num_chunks: Number of chunks
        chunk_histogram: Binned chunk length data for CSS histogram
        chunk_details: List of chunk detail dicts
        avg_chunk_length: Average chunk length in characters
        context_loss_score: Ratio of chunks ending mid-sentence
        heatmap_data: Plotly-compatible heatmap data structure

    Returns:
        Complete HTML report string
    """
    from datetime import datetime

    def score_class(score: float | None) -> str:
        if score is None:
            return "score-na"
        if score >= 0.9:
            return "score-excellent"
        elif score >= 0.7:
            return "score-good"
        elif score >= 0.5:
            return "score-fair"
        else:
            return "score-bad"

    template = Template(REPORT_TEMPLATE)
    return template.render(
        project_name=project_name,
        timestamp=datetime.now().strftime("%d %b %Y, %H:%M"),
        overall_score=int(overall_score * 100) if overall_score is not None else None,
        overall_class=score_class(overall_score),
        retrieval_score=int(retrieval_score * 100) if retrieval_score is not None else None,
        retrieval_class=score_class(retrieval_score),
        faithfulness_score=int(faithfulness_score * 100) if faithfulness_score is not None else None,
        faithfulness_class=score_class(faithfulness_score),
        tests_passed=tests_passed,
        tests_total=tests_total,
        failures=failures,
        recommendations=recommendations,
        current_score=current_score,
        predicted_score=predicted_score,
        chunk_strategy=chunk_strategy,
        num_chunks=num_chunks,
        chunk_histogram=chunk_histogram,
        chunk_details=chunk_details,
        avg_chunk_length=avg_chunk_length,
        context_loss_score=context_loss_score,
        heatmap_data=heatmap_data,
    )