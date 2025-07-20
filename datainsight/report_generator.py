from jinja2 import Environment, FileSystemLoader


def generate_html_report(summary, description, outliers, heatmap_path, missing_path, output_file):
    env = Environment(loader=FileSystemLoader('datainsight/templates'))
    template = env.get_template('report.html')
    html_content = template.render(
        summary=summary,
        description=description,
        outliers=outliers,
        heatmap_img=heatmap_path,
        missing_path=missing_path
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)