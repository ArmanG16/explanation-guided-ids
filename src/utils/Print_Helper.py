import time

def MyPrint(file_name, message, line_num=-1, error=False, success=False):
    line_num_msg = ""
    if (line_num >= 0):
        line_num_msg = "Line: " + str(line_num) + " "

    color_change = ""
    if (error):
        color_change = "\033[91m"
    elif (success):
        color_change = "\033[92m"

    timestamp = time.strftime("%H:%M:%S")

    print(color_change + timestamp + ", " + file_name + ": " + line_num_msg + message + "\033[0m" if color_change == "" else "")