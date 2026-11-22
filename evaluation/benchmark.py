"""
Performance and Latency Benchmarking module.
Measures execution speed, memory footprint, and stage-by-stage timings.
"""
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

from app.core.pipeline import DocumentIntelligencePipeline
from app.utils.sample_generator import SampleGenerator


class BenchmarkRunner:
    """Executes throughput and latency benchmarks across PDF and Image documents."""

    def __init__(self):
        self.pipeline = DocumentIntelligencePipeline()
        self.sample_gen = SampleGenerator()

    def run_full_benchmark(self, iterations_per_doc: int = 2) -> Dict[str, Any]:
        """
        Benchmarks all generated sample documents and collects latency statistics.
        """
        samples = self.sample_gen.generate_all_samples()
        benchmark_results = []
        all_total_latencies = []
        step_latencies: Dict[str, List[float]] = {}

        for category, files in samples.items():
            pdf_path = Path(files["pdf"])
            pdf_bytes = pdf_path.read_bytes()

            for i in range(iterations_per_doc):
                t0 = time.perf_counter()
                res = self.pipeline.process_document(pdf_bytes, pdf_path.name)
                total_ms = (time.perf_counter() - t0) * 1000
                all_total_latencies.append(total_ms)

                # Collect step durations
                for step in res.processing_steps:
                    if step.step_name not in step_latencies:
                        step_latencies[step.step_name] = []
                    step_latencies[step.step_name].append(step.duration_ms)

                if i == 0:
                    benchmark_results.append({
                        "category": category,
                        "filename": pdf_path.name,
                        "predicted_category": res.classification.predicted_category,
                        "confidence": res.classification.confidence,
                        "total_latency_ms": round(total_ms, 2),
                        "validation_status": res.validation.status,
                        "extracted_fields_count": len(res.extracted_fields),
                        "page_count": res.metadata.page_count,
                        "words_count": res.metadata.num_words
                    })

        avg_step_latencies = {
            step_name: round(float(np.mean(times)), 2)
            for step_name, times in step_latencies.items()
        }

        return {
            "total_benchmark_runs": len(all_total_latencies),
            "mean_latency_ms": round(float(np.mean(all_total_latencies)), 2),
            "median_latency_ms": round(float(np.median(all_total_latencies)), 2),
            "p95_latency_ms": round(float(np.percentile(all_total_latencies, 95)), 2),
            "min_latency_ms": round(float(np.min(all_total_latencies)), 2),
            "max_latency_ms": round(float(np.max(all_total_latencies)), 2),
            "step_latency_breakdown": avg_step_latencies,
            "document_benchmarks": benchmark_results
        }


if __name__ == "__main__":
    runner = BenchmarkRunner()
    results = runner.run_full_benchmark()
    print("=== Benchmark Summary ===")
    print(f"Mean Latency: {results['mean_latency_ms']} ms")
    print(f"P95 Latency:  {results['p95_latency_ms']} ms")
    print("Step Breakdown:")
    for step, ms in results["step_latency_breakdown"].items():
        print(f"  - {step}: {ms} ms")
