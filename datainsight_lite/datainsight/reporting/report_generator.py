from jinja2 import Environment, FileSystemLoader
import pkg_resources
import os

def generate_html_report(report, output_file):

    current_dir=os.path.dirname(__file__)
    template_path=os.path.join(current_dir,'..','templates')
    env = Environment(loader=FileSystemLoader(template_path))
    template = env.get_template('report.html')
    html_content = template.render(
        summary=report['summary'],
        description=report['description'],
        outliers=report['outliers'],
        heatmap_img=report['heatmap_path'],
        missing_path=report['missing_path']
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)