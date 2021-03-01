from datetime import timedelta
import ntpath

class Utilities:
    # Correctly format seconds to HH:MM:SS
    def format_seconds(self, s):
        return str(timedelta(seconds=s))

    def get_file_name(self, path):
        head, tail = ntpath.split(path)
        return tail or ntpath.basename(head)