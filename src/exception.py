import sys
from src.logger import logging


def error_message_details(error, error_detail: sys):
    """
    Generate an error message with detailed information about the error.

    Args:
        error (Exception): The original exception object.
        error_detail (sys): The sys module containing traceback information.

    Returns:
        str: Error message with details.

    """
    _, _, exc_tb = error_detail.exc_info()  # Retrieving the exception information
    file_name = (
        exc_tb.tb_frame.f_code.co_filename
    )  # Extracting the filename from the traceback object
    error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name,
        exc_tb.tb_lineno,
        str(error),  # Generating the error message with details
        )

    return error_message


class CustomException(Exception):
    """
    Custom exception class that includes detailed error information.

    Args:
        error_message (str): The error message.
        error_detail (sys): The sys module containing traceback information.

    Attributes:
        error_message (str): The error message with details.

    """
    def __init__(self, error_message, error_detail: sys):
        super().__init__(
            error_message
        )  # Initializing the base Exception class with the error message
        self.error_message = error_message_details(
            error_message,
            error_detail=error_detail,  # Generating the error message with details
        )

    def __str__(self):
        """
        Return the error message.

        Returns:
            str: The error message with details.

        """
        return self.error_message  # Returning the error message


if __name__ == "__main__":
    try:
        a = 1 / 0  # Performing a division by zero to raise an exception
    except Exception as e:
        logging.info("Divide by zero")
        raise CustomException(
            e, sys
        )  # Raising a CustomException with the original exception and sys module as error details
