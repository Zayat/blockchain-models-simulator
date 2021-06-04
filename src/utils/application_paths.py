# This class summarizes all important directories of the application template,
# it's good to avoid possible mistakes when handling files.

import os


class ApplicationPaths:

    @staticmethod
    def application_root(current_path=""):
        return os.path.abspath(os.path.dirname(__file__) + os.sep + "../../") + current_path

    @staticmethod
    def config(current_path=""):
        return ApplicationPaths.application_root() + os.sep + "config/" + current_path

    @staticmethod
    def logs(current_path=""):
        return ApplicationPaths.application_root() + os.sep + "logs/" + current_path

    @staticmethod
    def experiment_results(current_path=""):
        return ApplicationPaths.application_root() + os.sep + "experiment_results" + os.sep + current_path

    @staticmethod
    def evaluation_results(current_path=""):
        return ApplicationPaths.application_root() + os.sep + "evaluation_results" + os.sep + current_path

    @staticmethod
    def makedirs():
        os.makedirs(name=ApplicationPaths.logs(), exist_ok=True)
        os.makedirs(name=ApplicationPaths.experiment_results(), exist_ok=True)
        os.makedirs(name=ApplicationPaths.evaluation_results(), exist_ok=True)
