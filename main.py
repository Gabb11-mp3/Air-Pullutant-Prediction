"""AirScope application entry point."""

import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from error_handling import (
    show_unhandled_exception,
)
from main_window import AirPollutionWindow


def main() -> int:
    """Configure and run AirScope."""
    QApplication.setAttribute(
        Qt.AA_EnableHighDpiScaling,
        True,
    )

    QApplication.setAttribute(
        Qt.AA_UseHighDpiPixmaps,
        True,
    )

    application = QApplication(
        sys.argv
    )

    application.setApplicationName(
        "AirScope"
    )

    application.setStyle("Fusion")

    sys.excepthook = (
        show_unhandled_exception
    )

    window = AirPollutionWindow()
    window.showMaximized()

    return application.exec_()


if __name__ == "__main__":
    sys.exit(main())