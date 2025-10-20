import os


class HtmlFinder(object):
    def __init__(self, specific_name=None):
        self.queue = []
        self.specific_name = specific_name

    def find_html(self, path):
        try:
            if not os.path.isdir(path):
                return
            if self.queue:
                del self.queue[0]
            for dir in os.listdir(path):
                dir_path = os.path.join(path, dir)
                if os.path.isdir(dir_path):
                    self.queue.append(dir_path)
                elif self.specific_name is not None and dir_path.endswith(self.specific_name):
                    return dir_path
                elif dir_path.endswith(".html"):
                    html_path = dir_path
                    return html_path
                else: continue
            html_path = self.find_html(self.queue[0])
            if html_path is not None:
                return html_path
        except Exception as e:
            print(f"Error appears when finding {path}, error: {str(e)}")

    def reset_queue(self):
        self.queue = []