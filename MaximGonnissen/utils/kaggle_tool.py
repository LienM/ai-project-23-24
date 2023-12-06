import kaggle
from kaggle.models.kaggle_models_extended import Submission
from datetime import datetime
import json


class SubmissionWrapper:
    """
    Wrapper for Kaggle submission
    """
    def __init__(self, submission: Submission):
        self.submission = submission

    def get_description(self) -> str:
        return self.submission.__dict__.get('description')

    def get_filename(self) -> str:
        return self.submission.__dict__.get('fileName')

    def get_date(self) -> datetime:
        return self.submission.__dict__.get('date')

    def get_ref(self) -> str:
        return self.submission.__dict__.get('ref')

    def get_public_score(self) -> float:
        return self.submission.__dict__.get('publicScore')

    def get_private_score(self) -> float:
        return self.submission.__dict__.get('privateScore')

    def get_status(self) -> str:
        return self.submission.__dict__.get('status')

    def get_submitted_by(self) -> str:
        return self.submission.__dict__.get('submittedBy')

    def get_team_name(self) -> str:
        return self.submission.__dict__.get('teamName')

    def get_url(self) -> str:
        return self.submission.__dict__.get('url')

    def get_size(self) -> str:
        return self.submission.__dict__.get('size')

    def parse_json_description(self) -> dict:
        return json.loads(self.get_description())


class KaggleTool:
    """
    Utility class for working with Kaggle
    """
    is_authenticated = False

    def __init__(self, competition: str):
        self.api = kaggle.api
        self.competition = competition

        if not self.is_authenticated:
            self.api.authenticate()

    @PendingDeprecationWarning
    def get_all_submissions(self, competition: str = None):
        # TODO: pagination doesn't work? Deprecated for now
        submissions = []
        results = []
        page = 0
        while results not in [[], None]:
            results = self.api.process_response(self.api.competitions_submissions_list(id=competition, page=page))
            submissions.extend([Submission(s) for s in results])

            print(f'Page {page}: {len(results)} submissions')

            page += 1

        return submissions

    def list_submissions(self, competition: str = None):
        return self.api.competition_submissions(competition or self.competition)

    def list_submissions_wrapped(self, competition: str = None):
        return [SubmissionWrapper(submission) for submission in self.list_submissions(competition)]

    def upload_submission(self, file: str, message: str = None, competition: str = None, quiet: bool = True, metadata: dict = None):
        if metadata is not None:
            metadata['message'] = message
            message = json.dumps(metadata)

        self.api.competition_submit(file, message, competition or self.competition, quiet)
