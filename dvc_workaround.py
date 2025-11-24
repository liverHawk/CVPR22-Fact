import sys
import io


def fix_stdout_buffer():
    if hasattr(sys.stdout, "__class__"):
        class_name = str(sys.stdout.__class__)
        if "LoggingProxy"  in class_name and not hasattr(sys.stdout, "buffer"):
            if hasattr(sys.__stdout__, "buffer"):
                sys.stdout.buffer = sys.__stdout__.buffer
            else:
                sys.stdout.buffer = io.open(sys.__stdout__.fileno(), "wb", closefd=False)


# fix_stdout_buffer()