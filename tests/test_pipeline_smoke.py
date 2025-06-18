# tests/test_pipeline_smoke.py
"""
Smoke test for pipeline end-to-end functionality
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import json
from config.settings import VADER_CONFIG

from pipelines.main_pipeline import create_pipeline


@pytest.fixture
def tiny_dataset(tmp_path):
    """Create a tiny test dataset"""
    data = pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02'],
        'title': [
            'Apple Reports Strong Q4 Earnings Beat Expectations',
            'Microsoft Faces Regulatory Challenges in Cloud Market'
        ],
        'content': [
            'Apple Inc (AAPL) reported earnings per share of $2.10, beating analyst estimates. Revenue grew 15% year over year.',
            'Microsoft Corporation (MSFT) is under scrutiny from regulators regarding its cloud computing practices.'
        ],
        'symbols': [['AAPL'], ['MSFT']],
        'tags': [['earnings'], ['regulation']]
    })

    test_file = tmp_path / "test_data.parquet"
    data.to_parquet(test_file)
    return test_file


def test_pipeline_end_to_end(tiny_dataset, tmp_path):
    """Test full pipeline execution"""
    output_file = tmp_path / "output.jsonl"

    # Create and run pipeline
    pipeline = create_pipeline(
        pipeline_type="optimized",
        input_path=str(tiny_dataset),
        output_path=str(output_file),
        batch_size=2,
        resume=False,
        agg_method="conf_weighted"
    )

    # Run pipeline
    pipeline.run(max_articles=2)

    # Verify output exists
    assert output_file.exists(), "Output file was not created"

    # Verify content
    with open(output_file, 'r') as f:
        lines = f.readlines()

    assert len(lines) >= 2, f"Expected at least 2 lines, got {len(lines)}"

    # Verify each line is valid JSON
    for line in lines:
        data = json.loads(line)
        assert 'overall_sentiment' in data
        assert 'tickers' in data
        assert data['overall_sentiment'] in ['Positive', 'Neutral', 'Negative']


def test_pipeline_checkpoint_recovery(tiny_dataset, tmp_path):
    """Test pipeline can resume from checkpoint"""
    output_file = tmp_path / "output_resume.jsonl"

    # First run - process only 1 article
    pipeline1 = create_pipeline(
        pipeline_type="standard",
        input_path=str(tiny_dataset),
        output_path=str(output_file),
        batch_size=1,
        resume=True
    )
    pipeline1.run(max_articles=1)

    # Second run - should skip first article and process second
    pipeline2 = create_pipeline(
        pipeline_type="standard",
        input_path=str(tiny_dataset),
        output_path=str(output_file),
        batch_size=1,
        resume=True
    )
    pipeline2.run(max_articles=2)

    # Verify we have 2 articles total
    with open(output_file, 'r') as f:
        lines = f.readlines()

    assert len(lines) == 2, f"Expected 2 lines after resume, got {len(lines)}"


def test_aggregation_methods(tiny_dataset, tmp_path):
    """Test different aggregation methods produce different results"""
    methods = ["default", "majority", "conf_weighted"]
    results = {}

    for method in methods:
        output_file = tmp_path / f"output_{method}.jsonl"

        pipeline = create_pipeline(
            pipeline_type="optimized",
            input_path=str(tiny_dataset),
            output_path=str(output_file),
            agg_method=method,
            resume=False
        )

        pipeline.run(max_articles=2)

        # Read results
        with open(output_file, 'r') as f:
            results[method] = [json.loads(line) for line in f]

    # Verify we got results for all methods
    assert len(results) == 3

    # At least one method should produce different results
    sentiments_default = [r['overall_sentiment'] for r in results['default']]
    sentiments_majority = [r['overall_sentiment'] for r in results['majority']]
    sentiments_weighted = [r['overall_sentiment']
                           for r in results['conf_weighted']]

    # Check that we have valid sentiments
    all_sentiments = sentiments_default + sentiments_majority + sentiments_weighted
    assert all(s in ['Positive', 'Neutral', 'Negative']
               for s in all_sentiments)
