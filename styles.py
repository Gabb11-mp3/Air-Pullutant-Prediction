"""Qt stylesheet for the AirScope interface."""

from PyQt5.QtWidgets import QWidget


APP_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #F1F5F9;
    color: #0F172A;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 10pt;
}

QFrame#HeaderCard {
    background-color: #0F172A;
    border-radius: 10px;
}

QFrame#Card {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
}

QLabel#AppTitle {
    background: transparent;
    color: #F8FAFC;
    font-size: 22pt;
    font-weight: 700;
}

QLabel#AppSubtitle {
    background: transparent;
    color: #CBD5E1;
    font-size: 10pt;
}

QLabel#FieldLabel,
QLabel#SectionTitle {
    background: transparent;
    color: #0F172A;
    font-weight: 700;
}

QLabel#SectionTitle {
    font-size: 13pt;
}

QLabel#PathLabel,
QLabel#MutedLabel {
    background: transparent;
    color: #64748B;
}

QLabel#InfoNote {
    background-color: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 6px;
    color: #1E40AF;
    padding: 9px;
}

QPushButton {
    border-radius: 6px;
    font-weight: 600;
    min-height: 20px;
    padding: 7px 14px;
}

QPushButton#PrimaryButton {
    background-color: #2563EB;
    border: 1px solid #2563EB;
    color: #FFFFFF;
}

QPushButton#PrimaryButton:hover {
    background-color: #1D4ED8;
}

QPushButton#SecondaryButton {
    background-color: #FFFFFF;
    border: 1px solid #CBD5E1;
    color: #334155;
}

QPushButton#SecondaryButton:hover {
    background-color: #F8FAFC;
    border-color: #94A3B8;
}

QPushButton:disabled {
    background-color: #E2E8F0;
    border-color: #E2E8F0;
    color: #94A3B8;
}

QSpinBox {
    background-color: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 6px;
    padding: 6px;
}

QTableWidget {
    background-color: #FFFFFF;
    alternate-background-color: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 6px;
    gridline-color: #E2E8F0;
    selection-background-color: #DBEAFE;
    selection-color: #1E3A8A;
}

QHeaderView::section {
    background-color: #F8FAFC;
    border: none;
    border-bottom: 1px solid #E2E8F0;
    color: #475569;
    font-weight: 700;
    padding: 8px;
}

QScrollArea#ChartScroll,
QToolBar#ChartToolbar {
    background-color: #F8FAFC;
    border: none;
}

QStatusBar {
    background-color: #F8FAFC;
    color: #475569;
}
"""


def apply_styles(widget: QWidget) -> None:
    """Apply the shared stylesheet."""
    widget.setStyleSheet(APP_STYLESHEET)