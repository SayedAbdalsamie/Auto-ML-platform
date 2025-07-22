import sweetviz as sv
import pandas as pd


def generate_report(df, report_path):
    report = sv.analyze(df)
    report.show_html(filepath=report_path, open_browser=False)
    return report_path
