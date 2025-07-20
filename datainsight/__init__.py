from .reporter import Reporter
from .report_generator import generate_html_report
from utils import datatypes


def quick_report(file_path: datatypes.Str, sheet_name: datatypes.Str = None, output: datatypes.Str = "report.html"):
    report = Reporter(file_path, sheet_name).report()
    generate_html_report(report, output)
