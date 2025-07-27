import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
from .reporting.reporter import Reporter
from .reporting.report_generator import generate_html_report
from utils import datatypes


def quick_report(
                file_path: datatypes.Str, sheet_name: datatypes.Str = None, 
                spec_cat_col:datatypes.List[datatypes.Str]=[], 
                spec_num_col:datatypes.List[datatypes.Str]=[], 
                output: datatypes.Str = "report.html"
                ):
    report = Reporter(
                    file_path, 
                    sheet_name, 
                    cols_to_treat_as_cat=spec_cat_col,
                    cols_to_treat_as_num=spec_num_col
                    ).report()
    generate_html_report(report, output)
