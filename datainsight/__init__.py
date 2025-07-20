from .core import dataset_summary, describe_numerical, detect_outliers
from .visualizations import plot_correlation_heatmap, plot_missing_values
from .report_generator import generate_html_report


def quick_report(df, output="report.html"):
    summary = dataset_summary(df)
    description = describe_numerical(df).to_string()
    outliers = detect_outliers(df)

    heatmap_path = "heatmap.png"
    missing_path = "missing.png"
    plot_missing_values(df, missing_path)
    plot_correlation_heatmap(df, heatmap_path)

    generate_html_report(summary, description, outliers, heatmap_path, missing_path, output)
    print(f"Report generated: {output}")