"""Application-level unexpected-error handling."""

import traceback
from types import TracebackType
from typing import Type

from PyQt5.QtWidgets import QMessageBox


def show_unhandled_exception(
    exception_type: Type[BaseException],
    exception: BaseException,
    exception_traceback: TracebackType,
) -> None:
    """Show unexpected errors inside a message box."""
    dialog = QMessageBox()
    dialog.setIcon(QMessageBox.Critical)
    dialog.setWindowTitle("Unexpected application error")
    dialog.setText(
        "The application encountered an unexpected error."
    )
    dialog.setInformativeText(str(exception))
    dialog.setDetailedText(
        "".join(
            traceback.format_exception(
                exception_type,
                exception,
                exception_traceback,
            )
        )
    )
    dialog.exec_()