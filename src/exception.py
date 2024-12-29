import sys  # Importing sys module to work with system-specific parameters and functions
from src.logger import logging  # Importing a custom logging module

# Function to extract detailed error information
def error_message_detail(error, error_detail: sys):
    """
    Extracts and formats detailed information about an error.

    Args:
        error (Exception): The exception object that caused the error.
        error_detail (sys): The sys module, used to access detailed error information.

    Returns:
        str: A formatted string containing:
             - The file name where the error occurred.
             - The line number of the error.
             - The error message itself.
    """
    # Fetch traceback object from the error details
    _, _, exc_tb = error_detail.exc_info()
    
    # Extract the name of the file where the error occurred
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    # Format the error message with file name, line number, and error message
    error_message = (
        f"Error occurred in python script name [{file_name}] "
        f"line number [{exc_tb.tb_lineno}] error message [{str(error)}]"
    )
    
    return error_message


# Custom exception class to provide detailed error messages
class CustomException(Exception):
    """
    A custom exception class that extends Python's built-in Exception class.

    Purpose:
        - To provide detailed and formatted error messages for debugging.
    
    Args:
        error_message (str): A short description of the error.
        error_detail (sys): The sys module, used to fetch detailed error information.

    Methods:
        - __str__: Returns the detailed error message when the exception is printed.
    """
    def __init__(self, error_message, error_detail: sys):
        """
        Constructor to initialize the CustomException class.

        - Calls the parent Exception class constructor.
        - Generates a detailed error message using the error_message_detail function.

        Args:
            error_message (str): The original error message.
            error_detail (sys): The sys module to fetch error details.
        """
        # Initialize the parent Exception class with the error message
        super().__init__(error_message)
        
        # Generate a detailed error message using the helper function
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
    
    def __str__(self):
        """
        Converts the exception object to a string.

        Returns:
            str: The detailed error message.
        """
        return self.error_message