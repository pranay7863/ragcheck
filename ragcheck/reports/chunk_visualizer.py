"""Generate HTML chunk visualization."""


from jinja2 import Template

from ragcheck.analyzers.chunkers import Chunk

CHUNK_VIZ_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>ragcheck — Chunk Visualization</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0; padding: 20px; background: #f5f5f5;
        }
        .container {
            max-width: 1200px; margin: 0 auto; background: white;
            padding: 30px; border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        h1 { color: #1a1a2e; border-bottom: 3px solid #e94560; padding-bottom: 10px; }
        .chunk { margin: 8px 0; padding: 12px; border-radius: 6px; border-left: 4px solid; }
        .chunk-fixed { background: #e3f2fd; border-color: #2196f3; }
        .chunk-semantic { background: #e8f5e9; border-color: #4caf50; }
        .chunk-recursive { background: #fff3e0; border-color: #ff9800; }
        .overlap { background: #fce4ec; border-color: #e91e63; font-style: italic; }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px; margin: 20px 0;
        }
        .stat-card { background: #f8f9fa; padding: 15px; border-radius: 6px; text-align: center; }
        .stat-value { font-size: 2em; font-weight: bold; color: #e94560; }
        .stat-label { color: #666; font-size: 0.9em; }
        #chart { width: 100%; height: 400px; }
        .document-text {
            background: #fafafa; padding: 15px; border-radius: 6px;
            line-height: 1.8; font-size: 14px;
        }
        .highlight { padding: 2px 4px; border-radius: 3px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>ragcheck — Chunk Visualization</h1>
        <p><strong>Source:</strong> {{ source }} | <strong>Strategy:</strong> {{ strategy }}</p>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{{ num_chunks }}</div>
                <div class="stat-label">Total Chunks</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ avg_length|int }}</div>
                <div class="stat-label">Avg Length</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ context_loss_score|round(2) }}</div>
                <div class="stat-label">Context Loss</div>
            </div>
        </div>

        <h2>Chunk Length Distribution</h2>
        <div id="chart"></div>

        <h2>Document with Chunk Boundaries</h2>
        <div class="document-text">{{ document_html|safe }}</div>

        <h2>Chunk Details</h2>
        {% for chunk in chunks %}
        <div class="chunk chunk-{{ strategy }}">
            <strong>Chunk {{ loop.index }}</strong>
            ({{ chunk.start }}–{{ chunk.end }}):
            {{ chunk.text[:200] }}
            {% if chunk.text|length > 200 %}...{% endif %}
        </div>
        {% endfor %}
    </div>

    <script>
        const lengths = {{ lengths|tojson }};
        Plotly.newPlot('chart', [{
            x: lengths,
            type: 'histogram',
            nbinsx: 20,
            marker: { color: '#e94560' }
        }], {
            title: 'Chunk Length Distribution',
            xaxis: { title: 'Chunk Length (characters)' },
            yaxis: { title: 'Frequency' }
        });
    </script>
</body>
</html>
"""


def generate_chunk_viz(chunks: list[Chunk], source: str, strategy: str, document_text: str) -> str:
    """Generate HTML chunk visualization.

    Args:
        chunks: List of Chunk objects
        source: Source document path
        strategy: Chunking strategy name
        document_text: Original document text

    Returns:
        HTML string
    """
    lengths = [len(c.text) for c in chunks]
    avg_length = sum(lengths) / len(lengths) if lengths else 0

    context_loss = sum(1 for c in chunks if c.text and c.text[-1] not in ".!?:")
    context_loss_score = context_loss / len(chunks) if chunks else 0

    document_html = document_text[:500] + "..." if len(document_text) > 500 else document_text

    template = Template(CHUNK_VIZ_TEMPLATE)
    return template.render(
        source=source,
        strategy=strategy,
        num_chunks=len(chunks),
        avg_length=avg_length,
        context_loss_score=context_loss_score,
        lengths=lengths,
        chunks=chunks,
        document_html=document_html,
    )
